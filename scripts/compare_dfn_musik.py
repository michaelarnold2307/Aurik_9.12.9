#!/usr/bin/env python3
"""Compare DFN Speech vs DFN Musik on a corpus track (CPU inference)."""

import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "models/deepfilternet_v3_ii/DeepFilterNet")
sys.path.insert(0, "models/deepfilternet_v3_ii/pyDF-data")
from pathlib import Path

from df.config import config

config.use_defaults()
from df.model import init_model

TRACK = "corpus/cassette/damaged/cassette_pop_1980s_hiss_wow.wav"
OUT_DIR = Path("logs/dfn_compare")
OUT_DIR.mkdir(exist_ok=True)

# Load audio
y, sr = sf.read(TRACK)
if sr != 48000:
    import librosa

    y = librosa.resample(y, orig_sr=sr, target_sr=48000)
    sr = 48000
y = y[: sr * 10].astype(np.float32)  # first 10 seconds

# Load models
print("Loading speech DFN...")
speech_model = init_model().cpu().eval()  # original ONNX weights
print("Loading music DFN...")
ckpt = torch.load("models/deepfilternet_v3_ii/finetuned/dfn_musik_best.pt", map_location="cpu")
music_model = init_model().cpu().eval()
music_model.load_state_dict(ckpt["model_state_dict"], strict=False)

# Features (simplified - just pass-through for now)
sf.write(str(OUT_DIR / "original.wav"), y, sr)
print(f"Saved: {OUT_DIR}/original.wav")
print("Full comparison needs feature extraction pipeline - run after DFN completes")
