import argparse
import logging
import random
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
import codecs

from numpy.core.multiarray import _reconstruct
from numpy import ndarray
from torch.serialization import add_safe_globals
add_safe_globals([
    _reconstruct,
    ndarray,
    codecs.encode,
    np.dtype,
    np.dtype('uint32').__class__,
])

from datasets import Audio, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, set_seed
import torchaudio
import torchaudio.transforms as T

# Data augmentation functions
def add_noise(audio, noise_level=0.005):
    noise = torch.randn_like(audio) * noise_level
    return audio + noise

def change_speed(audio, orig_sr, factor=1.1):
    new_sr = int(orig_sr * factor)
    return torchaudio.functional.resample(audio, orig_sr, new_sr)

def random_volume(audio, gain_db_range=(-5, 5)):
    gain_db = random.uniform(*gain_db_range)
    return audio * (10.0 ** (gain_db / 20.0))

# Logging and seeding
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

# Path resolution for CSV file
def resolve_paths(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    base = csv_path.parent
    df["path"] = df["path"].map(
        lambda p: str((base / p).resolve()) if not Path(p).is_absolute() else str(Path(p))
    )
    return df

# Dataset preprocessing: augment & extract features
def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess_fn(example):
        # Load raw waveform
        audio = example["path"]["array"]
        audio = torch.tensor(audio)
        
        # Basic augmentations (noise, volume, speed)
        if random.random() < 0.5:
            audio = add_noise(audio)
            audio = random_volume(audio)
            audio = change_speed(audio, orig_sr=16000, factor=random.choice([0.9, 1.1]))
        
        # Feature extraction (raw audio -> model inputs)
        audio_inputs = processor.feature_extractor(
            audio, sampling_rate=16000, return_attention_mask=False
        )
        input_values = audio_inputs["input_values"]
        if isinstance(input_values, list) and len(input_values) == 1:
            input_values = input_values[0]

        # Tokenize labels (remove apostrophes)
        text = example["text"].replace("'", " ")
        with processor.as_target_processor():
            labels = processor.tokenizer(text, add_special_tokens=False).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
            "reference_text": example["text"].lower()
        }
    return dataset.map(preprocess_fn, remove_columns=["path", "text"])

# Checkpoint utilities
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved at {filepath}")

def load_checkpoint(model, optimizer, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, ckpt['epoch'], ckpt['loss']

# WER/CER computation
def compute_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    return float(d[len(r)][len(h)])/len(r) if r else 1.0

def compute_cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    return float(d[len(r)][len(h)])/len(r) if r else 1.0

# Training and evaluation loops
def train(model, processor, train_loader, eval_dataset, device, args, logger, checkpoint_path, patience=3, min_delta=0.001):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.to(device)

    start_epoch = 0
    if checkpoint_path:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        logger.info(f"Resuming from epoch {start_epoch+1}")

    best_wer, counter = float('inf'), 0
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_values=batch['input_values'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        logger.info(f"Train Loss: {avg_loss:.4f}")

        wer, _ = evaluate(model, processor, eval_dataset, device, logger, num_samples=5)
        if best_wer - wer > min_delta:
            best_wer, counter = wer, 0
            save_checkpoint(model, optimizer, epoch+1, avg_loss, f"{args.output_dir}/best_model.pth")
        else:
            counter += 1
            logger.info(f"No improvement. Patience {counter}/{patience}")
            if counter>=patience:
                logger.info("Early stopping.")
                break
        # remove old checkpoints
        for ck in glob.glob(f"{args.output_dir}/checkpoint_epoch_*.pth"): os.remove(ck)
        save_checkpoint(model, optimizer, epoch+1, avg_loss, f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pth")
    return model

def evaluate(model, processor, dataset, device, logger, num_samples=None):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm): m.eval()
    
    idxs = random.sample(range(len(dataset)), num_samples) if num_samples else range(len(dataset))
    wers, cers = [], []
    with torch.no_grad():
        for i in idxs:
            s = dataset[i]
            iv = torch.tensor(s['input_values']).unsqueeze(0).to(device)
            mask = (iv!=0.0).long()
            logits = model(iv, attention_mask=mask).logits
            preds = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(preds, skip_special_tokens=True)[0].lower().strip()
            text = text.replace('<unk>','').replace('  ',' ').strip()
            ref = s['reference_text']
            w = compute_wer(ref, text); c = compute_cer(ref, text)
            wers.append(w); cers.append(c)
            logger.info(f"Ref: {ref}")
            logger.info(f"Pred: {text}")
            logger.info(f"WER: {w:.2f}, CER: {c:.2f}")
    logger.info(f"Average WER: {np.mean(wers):.4f}, CER: {np.mean(cers):.4f}")
    return float(np.mean(wers)), float(np.mean(cers))

# Argument parsing and main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Manual fine-tuning of Wav2Vec2.")
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--csv', type=str, default="transcript.csv")
    parser.add_argument('--model-name', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--output-dir', type=str, default="./output")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    model.freeze_feature_encoder()
    model.config.hidden_dropout = 0.1
    model.config.attention_dropout = 0.1

    # Load & preprocess data
    csv_path = Path(args.data_path) / args.csv
    df = pd.read_csv(csv_path, sep=";")
    df = resolve_paths(df, csv_path)
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("path", Audio(sampling_rate=16000))
    proc_ds = preprocess_dataset(ds, processor)

    # Split train / val
    val_len = int(0.2 * len(proc_ds))
    train_ds, val_ds = random_split(
        proc_ds,
        [len(proc_ds)-val_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # DataLoader
    pad = processor.tokenizer.pad_token_id
    def collate_fn(batch):
        iv = pad_sequence([torch.tensor(b['input_values']) for b in batch], batch_first=True, padding_value=0.0)
        am = (iv!=0.0).long()
        lbl = pad_sequence([torch.tensor(b['labels']) for b in batch], batch_first=True, padding_value=pad)
        lbl[lbl==pad] = -100
        return {'input_values': iv, 'attention_mask': am, 'labels': lbl}

    loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Train & eval
    trained = train(model, processor, loader, val_ds, device, args, logger, args.checkpoint_path)
    logger.info("Training complete.")
    evaluate(trained, processor, val_ds, device, logger)

    # Save final model & processor
    trained.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    logger.info("Done.")
