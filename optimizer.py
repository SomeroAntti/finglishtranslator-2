#!/usr/bin/env python3
"""
optimize_model.py

1. Ensure data directory exists
2. Load Wav2Vec2ForCTC model
3. Baseline eval (size, latency, WER)
4. Global unstructured pruning + Dynamic INT8 quantization
5. Gzip-compress the dynamic-int8 state_dict to push under 30MB
6. Ablation logging
7. Export best model: TorchScript logits-only
"""

import os
import time
import csv
import gzip
import shutil
import torch
import torchaudio
from torch import nn
import torch.nn.utils.prune as prune
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer

# === CONFIGURATION ===
MODEL_CHECKPOINT = "./model.pt"          # your checkpoint path
BASE_MODEL = "facebook/wav2vec2-base-960h"
DATA_DIR = "./data"
EVAL_SAMPLES = 100
DEVICE = torch.device("cpu")
ABLATION_CSV = "ablation.csv"
MAX_AUDIO_LENGTH = 16000 * 5  # 5s
PRUNE_AMOUNT = 0.8  # fraction of weights to prune globally

# === UTILITIES ===
def measure_size(path): return round(os.path.getsize(path) / (1024**2), 2)

def measure_latency(model, input_values, runs=50):
    model.to(DEVICE).eval()
    with torch.no_grad(): _ = model(input_values)
    start = time.time()
    for _ in range(runs): _ = model(input_values)
    return (time.time()-start)/runs*1000

def compute_wer(model, processor, dataset, num_samples=100):
    model.to(DEVICE).eval(); total, cnt = 0.0, 0
    for wav, sr, transcript, *_ in dataset:
        if cnt>=num_samples: break
        inp = processor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True).input_values.to(DEVICE)
        with torch.no_grad(): logits = model(inp).logits
        pred = processor.batch_decode(torch.argmax(logits, -1))[0]
        total += wer(transcript, pred); cnt+=1
    return round(total/cnt,4)

def log_ablation(row):
    hdr=["variant","size_mb","latency_ms","wer"]
    exists=os.path.isfile(ABLATION_CSV)
    with open(ABLATION_CSV,'a',newline='') as f:
        w=csv.writer(f)
        if not exists: w.writerow(hdr)
        w.writerow(row)

# === PREP DATA ===
os.makedirs(DATA_DIR, exist_ok=True)
print("Downloading LibriSpeech test-clean…")
test_clean = torchaudio.datasets.LIBRISPEECH(DATA_DIR, url="test-clean", download=True)

# === LOAD MODEL & PROC ===
proc = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL).to(DEVICE)
if os.path.exists(MODEL_CHECKPOINT):
    sd = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
model.eval()

# dummy input
inp0 = torch.zeros(1, MAX_AUDIO_LENGTH)
input_values = proc(inp0.numpy().squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)

# === 1. Baseline FP32 ===
print("\n=== Baseline FP32 ===")
size_fp32 = measure_size(MODEL_CHECKPOINT) if os.path.exists(MODEL_CHECKPOINT) else None
lat_fp32 = measure_latency(model, input_values)
wer_fp32 = compute_wer(model, proc, test_clean, num_samples=EVAL_SAMPLES)
print(f"Size: {size_fp32} MB | Latency: {lat_fp32:.1f} ms | WER: {wer_fp32}")
log_ablation(["fp32", size_fp32, f"{lat_fp32:.1f}", wer_fp32])

# === 2. Pruning + Dynamic INT8 ===
print("\n=== Applying global unstructured pruning ===")
# Collect all Linear weight parameters
to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, nn.Linear)]
prune.global_unstructured(
    to_prune,
    pruning_method=prune.L1Unstructured,
    amount=PRUNE_AMOUNT
)
# Make pruning permanent
for m, _ in to_prune:
    prune.remove(m, 'weight')

print("\n=== Dynamic INT8 quantization ===")
model_dyn = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).eval()
# save and size
torch.save(
    model_dyn.state_dict(),
    "model_dyn.pth",
    _use_new_zipfile_serialization=True
)
size_dyn = measure_size("model_dyn.pth")
lat_dyn = measure_latency(model_dyn, input_values)
wer_dyn = compute_wer(model_dyn, proc, test_clean, num_samples=EVAL_SAMPLES)
print(f"Size after prune+quant: {size_dyn} MB | Latency: {lat_dyn:.1f} ms | WER: {wer_dyn}")
log_ablation(["prune80_dynamic_int8", size_dyn, f"{lat_dyn:.1f}", wer_dyn])

# === 3. Gzip-compress for maximal reduction ===
print("\n=== Gzip Compress State Dict ===")
with open("model_dyn.pth", "rb") as f_in:
    with gzip.open("model_dyn.pth.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
size_gz = measure_size("model_dyn.pth.gz")
print(f"Gzipped size: {size_gz} MB")
log_ablation(["prune80_dynamic_int8_gzip", size_gz, "N/A", "N/A"] )

# === 4. Export TorchScript logits-only ===
class Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).logits
wrapper = Wrapper(model_dyn).eval()
traced = torch.jit.trace(wrapper, input_values)
traced.save("model_ts.pt")
print("Exported TorchScript as model_ts.pt")

print("Done. Ablation logged to", ABLATION_CSV)
