#!/usr/bin/env python3
"""
optimize_model.py

1. Ensure data directory exists
2. Load Wav2Vec2ForCTC model from Hugging Face
3. Baseline eval (size, latency, WER)
4. Dynamic & Static INT8 quantization
5. Ablation logging (ablation.csv)
6. Export best model: ExecuTorch if available, else TorchScript (logits-only)
"""

import os
import time
import csv
import torch
import torchaudio
from torch import nn
from torch.quantization import (
    quantize_dynamic,
    get_default_qconfig,
    prepare,
    convert,
)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer

# === CONFIGURATION ===
MODEL_CHECKPOINT = "./model.pt"          # local fine-tuned checkpoint path
BASE_MODEL = "facebook/wav2vec2-base-960h"
DATA_DIR = "./data"
CALIBRATION_SAMPLES = 200
EVAL_SAMPLES = 100
DEVICE = torch.device("cpu")
ABLATION_CSV = "ablation.csv"
MAX_AUDIO_LENGTH = 16000 * 10

# === UTILITIES ===

def measure_size(path):
    return round(os.path.getsize(path) / (1024**2), 2)


def measure_latency(model, input_values, runs=50):
    model.to(DEVICE).eval()
    with torch.no_grad():
        _ = model(input_values)
        start = time.time()
        for _ in range(runs):
            _ = model(input_values)
    return (time.time() - start) / runs * 1000  # ms


def compute_wer(model, processor, dataset, num_samples=100):
    model.to(DEVICE).eval()
    total_wer = 0.0
    count = 0
    for waveform, sample_rate, transcript, *_ in dataset:
        if count >= num_samples:
            break
        audio = waveform.squeeze().numpy()
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(DEVICE)
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        total_wer += wer(transcript, pred_str)
        count += 1
    return round(total_wer / count, 4)


def log_ablation(row):
    header = ["variant", "size_mb", "latency_ms", "wer"]
    exists = os.path.isfile(ABLATION_CSV)
    with open(ABLATION_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

# === PREPARE DATA ===
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Data directory: {DATA_DIR}")
try:
    test_clean = torchaudio.datasets.LIBRISPEECH(DATA_DIR, url="test-clean", download=True)
except Exception as e:
    print("Failed to download LibriSpeech dataset:", e)
    exit(1)

# === LOAD BASE MODEL & PROCESSOR ===
print("Loading base Wav2Vec2ForCTC…")
processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
model_base = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL).to(DEVICE)
# load fine-tuned weights if available (ignore missing)
if os.path.exists(MODEL_CHECKPOINT):
    state = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model_base.load_state_dict(state, strict=False)
model_base.eval()

# dummy input
silence = torch.zeros(1, MAX_AUDIO_LENGTH)
inputs = processor(silence.numpy().squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
input_values = inputs.input_values.to(DEVICE)

# === 1. Baseline ===
print("\n=== Baseline FP32 ===")
size_fp32 = measure_size(MODEL_CHECKPOINT) if os.path.exists(MODEL_CHECKPOINT) else None
lat_fp32 = measure_latency(model_base, input_values)
wer_fp32 = compute_wer(model_base, processor, test_clean, num_samples=EVAL_SAMPLES)
print(f"Size: {size_fp32 or 'N/A'} MB | Latency: {lat_fp32:.1f} ms | WER: {wer_fp32}")
log_ablation(["fp32", size_fp32, f"{lat_fp32:.1f}", wer_fp32])

# === 2A. Dynamic INT8 ===
print("\n=== Dynamic INT8 ===")
model_dyn = quantize_dynamic(model_base, {nn.Linear}, dtype=torch.qint8).eval()
# save state_dict
torch.save(model_dyn.state_dict(), "model_dyn.pth")
size_dyn = measure_size("model_dyn.pth")
lat_dyn = measure_latency(model_dyn, input_values)
wer_dyn = compute_wer(model_dyn, processor, test_clean, num_samples=EVAL_SAMPLES)
print(f"Size: {size_dyn} MB | Latency: {lat_dyn:.1f} ms | WER: {wer_dyn}")
log_ablation(["dynamic_int8", size_dyn, f"{lat_dyn:.1f}", wer_dyn])

# === 2B. Static INT8 (optional) ===
print("\n=== Static INT8 ===")
try:
    model_stat = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL).to(DEVICE)
    if os.path.exists(MODEL_CHECKPOINT):
        model_stat.load_state_dict(state, strict=False)
    model_stat.qconfig = get_default_qconfig("fbgemm")
    prepare(model_stat, inplace=True)
    for i, (waveform, sr, *_ ) in enumerate(test_clean):
        inp = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True).input_values.to(DEVICE)
        model_stat(inp)
        if i >= CALIBRATION_SAMPLES:
            break
    model_stat = convert(model_stat, inplace=True).eval()
    torch.save(model_stat.state_dict(), "model_stat.pth")
    size_stat = measure_size("model_stat.pth")
    lat_stat = measure_latency(model_stat, input_values)
    wer_stat = compute_wer(model_stat, processor, test_clean, num_samples=EVAL_SAMPLES)
    print(f"Size: {size_stat} MB | Latency: {lat_stat:.1f} ms | WER: {wer_stat}")
    log_ablation(["static_int8", size_stat, f"{lat_stat:.1f}", wer_stat])
except Exception as e:
    print("Static quantization failed:", e)

# === 3. Export best model ===
# Choose quantized model object (dynamic or static) for export
quantized_model = model_dyn  # switch to model_stat if static is better

# Define a wrapper to only return logits (no dict output)
class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, input_values):
        # forward pass returns logits tensor directly
        return self.base(input_values).logits

# Export directly from the wrapped quantized model
wrapped = LogitsOnlyWrapper(quantized_model).eval()
try:
    from executorch.exir import export
    print("Exporting to ExecuTorch (.et)...")
    export(wrapped, (input_values,), output_dir="model_et")
    print("→ model_et directory created")
except (ImportError, AttributeError, ModuleNotFoundError) as e:
    print("ExecuTorch export failed — falling back to TorchScript:", e)
    traced = torch.jit.trace(wrapped, input_values)
    traced.save("model_ts.pt")
    print("→ model_ts.pt created")


print("Done. Ablation in", ABLATION_CSV)