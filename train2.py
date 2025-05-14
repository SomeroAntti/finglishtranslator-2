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
from torch.serialization import add_safe_globals
import codecs

from numpy.core.multiarray import _reconstruct
from numpy import ndarray
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

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def resolve_paths(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    base = csv_path.parent
    df["path"] = df["path"].map(lambda p: str((base / p).resolve()) if not Path(p).is_absolute() else str(Path(p)))
    return df


# For data augmentation
def add_noise(audio, noise_level=0.005):
    noise = torch.randn_like(audio) * noise_level
    return audio + noise

def change_speed(audio, orig_sr, factor=1.1):
    new_sr = int(orig_sr * factor)
    return torchaudio.functional.resample(audio, orig_sr, new_sr)

def random_volume(audio, gain_db_range=(-5, 5)):
    gain_db = random.uniform(*gain_db_range)
    return audio * (10.0 ** (gain_db / 20.0))


def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess(example):
        audio = example["path"]["array"]
        audio = torch.tensor(audio)

        # Augmentation
        if random.random() < 0.5:
            audio = add_noise(audio)
            audio = random_volume(audio)
            audio = change_speed(audio, orig_sr=16000, factor=random.choice([0.9, 1.1]))

        audio_inputs = processor.feature_extractor(audio, sampling_rate=16000, return_attention_mask=False)
        input_values = audio_inputs["input_values"]
        if isinstance(input_values, list) and len(input_values) == 1:
            input_values = input_values[0]

        with processor.as_target_processor():
            labels = processor.tokenizer(example["text"]).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
            "reference_text": example["text"]
            }

    return dataset.map(preprocess, remove_columns=["path", "text"])


def collate_fn(batch):
    input_tensors = [torch.tensor(b["input_values"], dtype=torch.float32) for b in batch]
    input_values = pad_sequence(input_tensors, batch_first=True, padding_value=0.0)
    attention_mask = (input_values != 0.0).long()

    label_tensors = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
    labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels
    }


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
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train(model, processor, train_loader, eval_dataset, device, args, logger, checkpoint_path, patience=2, min_delta=0.001):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.to(device)

    if checkpoint_path:
        model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
        logger.info(f"Resuming from epoch {start_epoch + 1} with loss {loss:.4f}")
    else:
        start_epoch = 0

    best_wer = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0.0

        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Clone labels to safely replace -100
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = processor.tokenizer.pad_token_id

            optimizer.zero_grad()
            outputs = model(input_values=input_values,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Train Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        current_wer, _ = evaluate(model, processor, eval_dataset, device, logger, num_samples=5)

        # Early stopping check
        if best_wer - current_wer > min_delta:
            best_wer = current_wer
            patience_counter = 0
            # Save best model
            best_model_path = f"{args.output_dir}/best_model.pth"
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, best_model_path)
            logger.info("Best model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in WER. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break        
        
        # Remove previous checkpoints
        prev_checkpoints = glob.glob(f"{args.output_dir}/checkpoint_epoch_*.pth")
        for ckpt in prev_checkpoints:
            os.remove(ckpt)

        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_path)

    return model


def evaluate(model, processor, dataset, device, logger, num_samples=None):
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.eval()
    wer_scores = []
    cer_scores = []

    sampled_indices = (
        range(len(dataset)) if num_samples is None else
        random.sample(range(len(dataset)), num_samples)
    )

    with torch.no_grad():
        for idx in sampled_indices:
            sample = dataset[idx]
            input_values = sample["input_values"]
            input_values = torch.tensor(input_values).unsqueeze(0).to(device)

            attention_mask = sample.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

            transcription = processor.batch_decode(pred_ids)[0].lower()
            reference = sample["reference_text"].lower()

            wer = compute_wer(reference, transcription)
            wer_scores.append(wer)

            cer = compute_cer(reference, transcription)
            cer_scores.append(cer)

            logger.info(f"Ref:  {reference}")
            logger.info(f"Pred: {transcription}")
            logger.info(f"WER:  {wer:.2f}")
            logger.info(f"CER:  {cer:.2f}\n")

    avg_wer = np.mean(wer_scores)
    avg_cer = np.mean(cer_scores)
    logger.info(f"Average WER: {avg_wer:.4f}")
    logger.info(f"Average CER: {avg_cer:.4f}")

    return avg_wer, avg_cer


def compute_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r)) if r else 1.0


def compute_cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r)) if r else 1.0


def main(args):
    logger = setup_logging()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    #Tweaking the model for small dataset: freezing the feature encoder
    model.freeze_feature_encoder()
    model.config.hidden_dropout = 0.1
    model.config.attention_dropout = 0.1

    logger.info("Preparing dataset...")
    csv_path = Path(args.data_path, args.csv)
    df = pd.read_csv(csv_path, sep=";")
    df = resolve_paths(df, Path(csv_path))
    original_dataset = Dataset.from_pandas(df)
    original_dataset = original_dataset.cast_column("path", Audio(sampling_rate=16000))
    processed_dataset = preprocess_dataset(original_dataset, processor)

    logger.info("Splitting dataset...")
    val_size = int(0.2 * len(processed_dataset))
    train_dataset, eval_dataset  = random_split(processed_dataset, [len(processed_dataset) - val_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # Check if we are resuming from a checkpoint
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else None

    logger.info("Starting training...")
    #trained_model = train(model, processor, train_loader, original_dataset, device, args, logger, checkpoint_path)
    trained_model = train(model, processor, train_loader, processed_dataset, device, args, logger, checkpoint_path)
    logger.info("Training complete.")

    logger.info("Final Evaluation on Entire Validation Set")
    evaluate(trained_model, processor, eval_dataset, device, logger, num_samples=None)

    logger.info("Saving model...")
    trained_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    torch.save(model.state_dict(), "model.pt")

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual fine-tuning of Wav2Vec2.")
    #default_csv = Path(__file__).resolve().parent / "transcript.csv"
    #parser.add_argument("--csv", type=str, default=str(default_csv))
    parser.add_argument('--data_path', type=str,default="")
    parser.add_argument("--csv", type=str, default="transcript.csv")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base-960h")
    #parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-large-960h-lv60-self")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()
    main(args)