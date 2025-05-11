import argparse
import logging
import random
import os
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


def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess(example):
        audio = example["path"]["array"]
        audio_inputs = processor.feature_extractor(audio, sampling_rate=16000, return_attention_mask=False)
        input_values = audio_inputs["input_values"]
        if isinstance(input_values, list) and len(input_values) == 1:
            input_values = input_values[0]

        with processor.as_target_processor():
            labels = processor.tokenizer(example["text"]).input_ids

        return {"input_values": input_values, "labels": labels}

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
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved at {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train(model, processor, train_loader, val_dataset, device, args, logger, checkpoint_path=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    ctc_loss = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    if checkpoint_path:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        logger.info(f"Resuming training from epoch {start_epoch + 1}...")
    else:
        start_epoch = 0
        logger.info("Starting training from scratch...")

    model.to(device)
    model.train()

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0.0

        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits

            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            input_lengths = torch.full(
                size=(logits.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long
            )

            labels[labels == -100] = processor.tokenizer.pad_token_id
            target_lengths = (labels != processor.tokenizer.pad_token_id).sum(-1)

            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        logger.info(f"Train Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, processor, val_dataset, device, logger)

        # Save checkpoint after each epoch
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(model, optimizer, epoch, total_loss / len(train_loader), checkpoint_path)

    return model


def evaluate(model, processor, dataset, device, logger):
    model.eval()
    wer_scores = []

    def decode(pred_ids):
        return processor.tokenizer.decode(pred_ids, skip_special_tokens=True).lower()

    sample_indices = random.sample(range(len(dataset)), min(6, len(dataset)))
    samples = dataset.select(sample_indices)

    with torch.no_grad():
        for sample in samples:
            audio = sample["path"]["array"]
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = processor.batch_decode(pred_ids)[0].lower()
            ref_text = sample["text"].lower()

            wer = compute_wer(ref_text, pred_text)
            wer_scores.append(wer)

            logger.info(f"Ref:  {ref_text}")
            logger.info(f"Pred: {pred_text}")
            logger.info(f"WER:  {wer:.2f}\n")

    logger.info(f"Average WER: {np.mean(wer_scores):.4f}")
    model.train()


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


def main(args):
    logger = setup_logging()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)

    logger.info("Preparing dataset...")
    df = pd.read_csv(args.csv, sep=";")
    df = resolve_paths(df, Path(args.csv))
    original_dataset = Dataset.from_pandas(df)
    original_dataset = original_dataset.cast_column("path", Audio(sampling_rate=16000))
    processed_dataset = preprocess_dataset(original_dataset, processor)

    logger.info("Splitting dataset...")
    val_size = int(0.2 * len(processed_dataset))
    train_dataset, _ = random_split(processed_dataset, [len(processed_dataset) - val_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # Check if we are resuming from a checkpoint
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else None

    logger.info("Starting training...")
    trained_model = train(model, processor, train_loader, original_dataset, device, args, logger, checkpoint_path)

    logger.info("Saving model...")
    trained_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual fine-tuning of Wav2Vec2.")
    default_csv = Path(__file__).resolve().parent / "transcript.csv"
    parser.add_argument("--csv", type=str, default=str(default_csv))
    #parser.add_argument("--csv", type=str, default="transcript.csv")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-large-960h-lv60-self")
    parser.add_argument("--output-dir", type=str, default="./wav2vec2-manual")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()
    main(args)
