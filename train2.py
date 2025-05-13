import torch
import torchaudio
import logging
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Dataset
import evaluate
from typing import Dict, List, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Preprocessing
def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess(example):
        audio = example["audio"]["array"]
        inputs = processor(audio, sampling_rate=16000)
        with processor.as_target_processor():
            labels = processor(example["text"]).input_ids
        return {
            "input_values": inputs["input_values"],
            "labels": labels,
            "reference_text": example["text"]
        }

    return dataset.map(preprocess, remove_columns=dataset.column_names)

# Compute metrics for evaluation
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)

    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids

    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    logger.info(f"WER: {wer:.4f}, CER: {cer:.4f}")
    for ref, hyp in zip(label_str, pred_str):
        logger.info(f"Ref:  {ref}")
        logger.info(f"Pred: {hyp}\n")
    return {"wer": wer, "cer": cer}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="", help="Path to audio dataset folder or HF dataset name")
parser.add_argument("--output_dir", type=str, default="./wav2vec2_finetuned")
parser.add_argument("--num_train_epochs", type=int, default=10)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
args = parser.parse_args()

# Load dataset
logger.info("Loading dataset...")
dataset = load_dataset("audiofolder", data_dir=args.data_path)
dataset = dataset["train"].train_test_split(test_size=0.2)

# Load model and processor
logger.info("Loading Wav2Vec2 processor and model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Preprocess
logger.info("Preprocessing dataset...")
processed_dataset = DatasetDict({
    "train": preprocess_dataset(dataset["train"], processor),
    "test": preprocess_dataset(dataset["test"], processor),
})

# TrainingArguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    group_by_length=True,
    per_device_train_batch_size=args.per_device_train_batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{args.output_dir}/logs",
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.005,
    warmup_steps=500,
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False,
)

# Trainer
logger.info("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
logger.info("Training complete. Saving model...")
trainer.save_model(args.output_dir)
processor.save_pretrained(args.output_dir)
logger.info("Done.")
