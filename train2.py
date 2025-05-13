import os
import torch
import torchaudio
import logging
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Audio
import pandas as pd
import numpy as np

# --------- Logging Setup ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Custom WER/CER ---------
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

# ------- Argument Parser -------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="",
                    help="Path to dataset folder containing transcript.csv and audio files")
parser.add_argument("--csv", type=str, default="C:\\Users\\Siidu\\Desktop\Gitit\\finglishtranslator-2\\transcript.csv", help="CSV filename in data_path")
parser.add_argument("--output_dir", type=str, default="./wav2vec2_finetuned")
parser.add_argument("--num_train_epochs", type=int, default=10)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
args = parser.parse_args()

# ------- Load CSV and Build Dataset -------
csv_file = os.path.join(args.data_path, args.csv)
logger.info(f"Loading metadata from {csv_file}")
df = pd.read_csv(csv_file, sep=";")
# Expect columns: path;text
base = args.data_path
df["path"] = df["path"].apply(lambda p: os.path.join(base, p) if not os.path.isabs(p) else p)

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
dataset = dataset.train_test_split(test_size=0.2)

# ------- Load Processor & Model -------
logger.info("Loading processor and model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# ------- Preprocessing Function -------
def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess(example):
        audio = example["path"]["array"]
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_attention_mask=False)
        text = example.get("text", "").lower().strip()
        with processor.as_target_processor():
            labels = processor.tokenizer(text).input_ids
        return {"input_values": inputs["input_values"], "labels": labels}
    return dataset.map(preprocess, remove_columns=["path", "text"])

logger.info("Preprocessing dataset...")
processed = DatasetDict({
    "train": preprocess_dataset(dataset["train"], processor),
    "test": preprocess_dataset(dataset["test"], processor)
})

# ------- Metric Computation -------
def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(logits), dim=-1).tolist()
    label_ids = pred.label_ids.tolist()

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(
        [[tok if tok != -100 else processor.tokenizer.pad_token_id for tok in seq] for seq in label_ids],
        skip_special_tokens=True
    )
    wers, cers = [], []
    for ref, hyp in zip(label_str, pred_str):
        wers.append(compute_wer(ref.lower(), hyp.lower()))
        cers.append(compute_cer(ref.lower(), hyp.lower()))
    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)
    logger.info(f"➡️  WER: {avg_wer:.4f}, CER: {avg_cer:.4f}")
    for ref, hyp in zip(label_str, pred_str)[:3]:
        logger.info(f"   REF: {ref}")
        logger.info(f"   HYP: {hyp}")
    return {"wer": avg_wer, "cer": avg_cer}

# ------- Training Arguments & Trainer -------
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    learning_rate=args.learning_rate,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed["train"],
    eval_dataset=processed["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

# ------- Run Training -------
logger.info("Starting training...")
trainer.train()
logger.info("Training complete. Saving...")
trainer.save_model(args.output_dir)
processor.save_pretrained(args.output_dir)
logger.info("Done.")
