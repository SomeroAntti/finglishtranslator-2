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
from datasets import Audio, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, set_seed
import torchaudio


# ------------- Setup and Utilities ------------------ #

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


# ------------- Data Preprocessing ------------------ #

def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess(example):
        audio = example["path"]["array"]
        audio = torch.tensor(audio)

        # --- Debug: check raw input ---
        if torch.all(audio == 0):
            print("Warning: Input audio is all zeros.")

        # --- Optional: augmentations (disabled for now) ---
        # if random.random() < 0.5:
        #     audio = add_noise(audio)
        #     audio = change_speed(audio, orig_sr=16000, factor=random.choice([0.9, 1.1]))
        #     audio = random_volume(audio)

        # --- Feature extraction (raw waveform) ---
        audio_inputs = processor(audio, sampling_rate=16000, return_attention_mask=False)
        input_values = audio_inputs["input_values"]
        if isinstance(input_values, list) and len(input_values) == 1:
            input_values = input_values[0]

        # --- Tokenize labels ---
        text = example["text"].lower().replace("'", "").strip()
        with processor.as_target_processor():
            labels = processor.tokenizer(text, add_special_tokens=False).input_ids

        if not labels:
            print(f"Warning: Empty label from text: '{text}'")

        return {
            "input_values": input_values,
            "labels": labels,
            "reference_text": text
        }

    return dataset.map(preprocess, remove_columns=["path", "text"])


# ------------- Evaluation ------------------ #

def compute_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    return float(d[len(r)][len(h)]) / len(r) if r else 1.0

def compute_cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = min(d[i-1][j-1]+1, d[i][j-1]+1, d[i-1][j]+1)
    return float(d[len(r)][len(h)]) / len(r) if r else 1.0

def evaluate(model, processor, dataset, device, logger, num_samples=None):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.eval()

    wer_scores, cer_scores = [], []
    indices = range(len(dataset)) if num_samples is None else random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for i in indices:
            sample = dataset[i]
            input_values = torch.tensor(sample["input_values"]).unsqueeze(0).to(device)
            attention_mask = (input_values != 0.0).long()

            logits = model(input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)

            # Debug prints
            logger.debug(f"Logits shape: {logits.shape}")
            logger.debug(f"Predicted IDs: {pred_ids.tolist()}")

            raw_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
            transcription = raw_preds[0].lower().strip().replace("<unk>", "").replace("  ", " ").strip()
            reference = sample["reference_text"].lower().strip()

            wer = compute_wer(reference, transcription)
            cer = compute_cer(reference, transcription)
            wer_scores.append(wer)
            cer_scores.append(cer)

            logger.info(f"Ref:  {reference}")
            logger.info(f"Pred: {transcription}")
            logger.info(f"WER:  {wer:.2f}, CER: {cer:.2f}")

    avg_wer = np.mean(wer_scores)
    avg_cer = np.mean(cer_scores)
    logger.info(f"Average WER: {avg_wer:.4f}")
    logger.info(f"Average CER: {avg_cer:.4f}")
    return avg_wer, avg_cer


# ------------- Training ------------------ #

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def train(model, processor, train_loader, eval_dataset, device, args, logger, checkpoint_path, patience=3, min_delta=0.001):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.to(device)

    start_epoch = 0
    if checkpoint_path:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    best_wer = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0.0

        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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

        current_wer, _ = evaluate(model, processor, eval_dataset, device, logger, num_samples=5)

        if best_wer - current_wer > min_delta:
            best_wer = current_wer
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, f"{args.output_dir}/best_model.pth")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping.")
                break

    return model


# ------------- Main Script ------------------ #

def main(args):
    logger = setup_logging()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    model.freeze_feature_encoder()
    model.config.hidden_dropout = 0.1
    model.config.attention_dropout = 0.1

    pad_id = processor.tokenizer.pad_token_id

    def collate_fn(batch):
        input_tensors = [torch.tensor(b["input_values"], dtype=torch.float32) for b in batch]
        input_values = pad_sequence(input_tensors, batch_first=True, padding_value=0.0)
        attention_mask = (input_values != 0.0).long()

        label_tensors = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
        labels = pad_sequence(label_tensors, batch_first=True, padding_value=pad_id)
        labels[labels == pad_id] = -100

        return {"input_values": input_values,
                "attention_mask": attention_mask,
                "labels": labels}

    logger.info("Preparing dataset...")
    csv_path = Path(args.data_path, args.csv)
    df = pd.read_csv(csv_path, sep=";")
    df = resolve_paths(df, csv_path)
    original_dataset = Dataset.from_pandas(df)
    original_dataset = original_dataset.cast_column("path", Audio(sampling_rate=16000))
    processed_dataset = preprocess_dataset(original_dataset, processor)

    logger.info("Splitting dataset...")
    val_size = int(0.2 * len(processed_dataset))
    train_dataset, eval_dataset = random_split(
        processed_dataset,
        [len(processed_dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    logger.info("Starting training...")
    trained_model = train(
        model, processor, train_loader, eval_dataset,
        device, args, logger, args.checkpoint_path
    )

    logger.info("Final evaluation...")
    evaluate(trained_model, processor, eval_dataset, device, logger, num_samples=None)

    logger.info("Saving model...")
    trained_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--csv', type=str, default="transcript.csv")
    parser.add_argument('--model-name', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--output-dir', type=str, default="./output")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    args = parser.parse_args()
    main(args)
