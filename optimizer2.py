import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader
from jiwer import wer
import os

# Ensure reproducibility
torch.manual_seed(0)

# Constants
DATASET_PATH = "./data"
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():
    print("Loading LibriSpeech dataset...")
    dataset = LIBRISPEECH(DATASET_PATH, url="test-clean", download=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def load_model():
    print("Loading custom model from model.pt...")
    model = torch.load("model.pt", map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    # You must define or load your label mapping here
    # For example, if your model uses the same labels as Wav2Vec2:
    labels = list("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ|")  # Adjust as needed
    return model, labels

def greedy_decode(emissions, labels):
    # emissions: [batch, time, num_labels]
    indices = torch.argmax(emissions, dim=-1)[0].cpu().tolist()
    prev = None
    result = []
    for idx in indices:
        if idx != prev and idx != 0:
            result.append(labels[idx])
        prev = idx
    return "".join(result).replace("|", " ").strip()

def evaluate(model, data_loader, labels):
    print("Evaluating baseline model...")
    transcriptions, references = [], []

    for i, (waveform, _, transcript, *_ ) in enumerate(data_loader):
        waveform = waveform.to(DEVICE).squeeze(1)
        with torch.inference_mode():
            emissions, _ = model(waveform)
        transcript_pred = greedy_decode(emissions, labels)
        transcriptions.append(transcript_pred.lower())
        references.append(transcript[0].lower())
        if i >= 10:
            break

    return wer(references, transcriptions)

def quantize_model(model):
    print("Quantizing model...")
    # Move model to CPU for static eager quantization
    model_cpu = model.cpu()
    qconfig = get_default_qconfig("fbgemm")

    # Prepare and convert (eager mode quantization)
    prepared = prepare(model_cpu, {"": qconfig})
    q_model = convert(prepared)

    return q_model

class QuantizedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        return self.model(waveform)

def export_to_executorch(model):
    print("Exporting quantized model to ExecuTorch...")
    wrapper = QuantizedWrapper(model)
    wrapper.eval()
    dummy_input = torch.randn(1, 16000)
    from torch.export import export, save
    try:
        exported_model = export(wrapper, (dummy_input,))
        save(exported_model, "quantized_model.et")
        print("Saved quantized model to quantized_model.et")
    except Exception as e:
        print(f"Export failed: {e}")
        raise
    return exported_model


def main():
    data_loader = load_dataset()
    model, labels = load_model()

    # Evaluate baseline WER
    wer_score = evaluate(model, data_loader, labels)
    print(f"Baseline WER: {wer_score * 100:.2f}%")

    # Quantize model
    quantized_model = quantize_model(model)
    print("Skipping WER evaluation for quantized model.")

    # Export quantized model
    et_program = export_to_executorch(quantized_model)
    print("Export successful!")

if __name__ == "__main__":
    main()
