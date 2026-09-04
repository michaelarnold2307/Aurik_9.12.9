#!/usr/bin/env python3
"""
§v10.300: Harmonic Inpainting — DiT Fine-Tuning.

Nimmt das vortrainierte MIIPHER DiT (FlowMatchingDiT, 201M, Epoch 120)
und trainiert es spezifisch auf HARMONIC INPAINTING:

  - DFN entfernt Rauschen → dämpft Obertöne in betroffenen Bändern
  - DiT rekonstruiert NUR die gedämpften harmonischen Anteile
  - Loss ist maskiert: nur inpainting-Regionen werden optimiert

Vorteil: 20-30 Epochen statt 500+ — 95% weniger Training.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "miipher_dit"))

from dit_model import FlowMatchingDiT

SR = 48_000
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
BATCH_SIZE = 1
ACCUM_STEPS = 4  # Effective batch = 8
EPOCHS = 30
LR = 5e-5
EARLY_STOP_PATIENCE = 5  # Stop if val-loss doesn't improve for 5 epochs
MIN_VAL_LOSS_IMPROVEMENT = 1e-6  # Minimum improvement to count as progress

CHECKPOINT_DIR = _PROJECT / "models" / "harmonic_inpainting"
BEST_PT = CHECKPOINT_DIR / "inpainting_best.pt"
LATEST_PT = CHECKPOINT_DIR / "inpainting_latest.pt"
PRETRAINED_DIT = _PROJECT / "models" / "miipher_dit" / "checkpoint_best_epoch120.pt"


# ═════════════════════════════════════════════════════════════════════════════
# Harmonic Mask Generator
# ═════════════════════════════════════════════════════════════════════════════


class HarmonicMaskGenerator:
    """
    Simuliert DFN-Dämpfung: identifiziert harmonische Peaks und dämpft
    Bänder, die vom Denoiser betroffen wären.

    Die Maske definiert, WELCHE Samples das DiT rekonstruieren soll.
    """

    def __init__(self, n_fft: int = 2048, hop: int = 512):
        self.n_fft = n_fft
        self.hop = hop

    def generate(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generiert eine Inpainting-Maske basierend auf simulierter DFN-Dämpfung.

        Returns:
            (attenuated_audio, inpainting_mask) — beide [T]
            mask: 1.0 = muss rekonstruiert werden, 0.0 = unberührt
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop
        window = np.hanning(self.n_fft)
        specgram = np.zeros((n_frames, self.n_fft // 2 + 1), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop
            frame = audio[start : start + self.n_fft] * window
            specgram[i] = np.abs(np.fft.rfft(frame))

        # Identifiziere harmonische Peaks (oberhalb des spektralen Medians)
        median_spec = np.median(specgram, axis=0)
        harmonic_mask_freq = np.zeros(self.n_fft // 2 + 1, dtype=bool)

        for freq_bin in range(1, self.n_fft // 2 + 1):
            if median_spec[freq_bin] > np.median(median_spec) * 2.5:
                # Harmonischer Peak
                harmonic_mask_freq[max(0, freq_bin - 2) : freq_bin + 3] = True

        # Erzeuge zeitabhängige Dämpfung: nur bei lauten Frames dämpfen
        frame_energy = specgram.mean(axis=1)
        loud_frames = frame_energy > np.median(frame_energy)

        # Baue Waveform-Maske
        mask = np.zeros(len(audio), dtype=np.float32)
        for i in range(n_frames):
            if loud_frames[i]:
                start = i * self.hop
                end = min(start + self.n_fft, len(audio))
                # Dämpfe: 30-60% Reduktion in harmonischen Bändern
                attenuation = np.random.uniform(0.3, 0.6)
                mask[start:end] = attenuation * window[: end - start]

        # Attenuiere das Audio
        attenuated = audio * (1.0 - mask)

        # Inpainting-Maske: wo wurde gedämpft?
        inpainting_mask = np.clip(mask, 0.0, 1.0)

        return attenuated.astype(np.float32), inpainting_mask.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Inpainting Dataset
# ═════════════════════════════════════════════════════════════════════════════


class InpaintingDataset(Dataset):
    """Lädt saubere Musik, simuliert DFN-Dämpfung, erzeugt Inpainting-Targets."""

    def __init__(self, files: list[Path]):
        self.files = files
        self.mask_gen = HarmonicMaskGenerator()

    def __len__(self):
        return max(len(self.files), 100) * 5

    def _load_chunk(self, path: Path) -> np.ndarray:
        try:
            with sf.SoundFile(str(path)) as snd:
                sr = snd.samplerate
                chunk_native = min(int(3.0 * sr), snd.frames)
                max_start = max(0, snd.frames - chunk_native)
                start_frame = random.randint(0, max_start) if max_start > 0 else 0
                snd.seek(start_frame)
                y = snd.read(chunk_native, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            y = np.nan_to_num(y).astype(np.float32)
            if len(y) < CHUNK_SAMPLES:
                y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
            else:
                y = y[:CHUNK_SAMPLES]
            # Peak-normalize
            peak = np.abs(y).max() + 1e-10
            return (y / peak).astype(np.float32)
        except Exception:
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def __getitem__(self, idx):
        f = self.files[idx % len(self.files)]
        clean = self._load_chunk(f)

        # Simuliere DFN-Dämpfung
        attenuated, mask = self.mask_gen.generate(clean)

        return {
            "clean": torch.from_numpy(clean),
            "attenuated": torch.from_numpy(attenuated),
            "mask": torch.from_numpy(mask),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════


def train(
    epochs: int = EPOCHS,
    lr: float = LR,
    steps_per_epoch: int = 200,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
) -> dict[str, float]:
    """Run fine-tuning with convergence gate.

    Returns:
        Training summary dict with epochs_run, best_val_loss, production_ready flag.
    """
    device = torch.device("cuda")

    # Daten
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    fma = _PROJECT / "data" / "fma_small" / "fma_small"
    corpus = _PROJECT / "corpus"

    all_files = []
    if fma.is_dir():
        all_files.extend(sorted(fma.rglob("*.mp3"))[:2000])
    all_files.extend(sorted(musdb.rglob("*.wav"))[:500])
    if corpus.is_dir():
        all_files.extend(sorted(corpus.rglob("*.wav"))[:200])

    rng = random.Random(42)
    rng.shuffle(all_files)
    n_train = int(0.9 * len(all_files))
    train_files, val_files = all_files[:n_train], all_files[n_train:]

    print(f"Files: {len(all_files)} → Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = InpaintingDataset(train_files)
    val_ds = InpaintingDataset(val_files)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True, prefetch_factor=2
    )

    # Modell: vortrainiertes DiT laden
    print(f"Loading pretrained DiT from {PRETRAINED_DIT}...", flush=True)
    model = FlowMatchingDiT().to(device)
    ckpt = torch.load(str(PRETRAINED_DIT), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: FlowMatchingDiT ({n_params:.0f}M total, {trainable:.0f}M trainable)")

    # Nur die letzten 6 Layer fine-tunen (rest frozen)
    # Das spart VRAM und verhindert Overfitting
    frozen_params = 0
    for name, param in model.named_parameters():
        if not (any(f"blocks.{i}." in name for i in range(15, 18)) or "final_ada" in name or "output_proj" in name):
            param.requires_grad = False
            frozen_params += 1
        else:
            pass  # Fine-tune Block 18-23

    trainable_ft = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Fine-tuning: {trainable_ft:.0f}M parameters (blocks 15-17 + final layers, rest frozen)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience_counter = 0
    production_ready = False

    print(f"Epochs: {epochs} | LR: {lr} | Effective batch: {BATCH_SIZE * ACCUM_STEPS} | Patience: {early_stop_patience}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean = batch["clean"].to(device).unsqueeze(-1)  # [B, T, 1]
            attenuated = batch["attenuated"].to(device).unsqueeze(-1)
            mask = batch["mask"].to(device).unsqueeze(-1)  # [B, T, 1]

            # Flow-Matching: sample random t
            t_vals = torch.rand(BATCH_SIZE, device=device)

            # Target velocity: clean - attenuated (direction to restore)
            # But we ONLY care about the mask region
            target = clean - attenuated  # [B, T, 1]
            target = target * mask  # [B, T, 1] — nur Inpainting-Region

            # Interpolate noisy towards clean
            noise = torch.randn_like(attenuated)
            x_t = (1 - t_vals[:, None, None]) * attenuated + t_vals[:, None, None] * clean
            x_t = x_t + 0.01 * noise  # small noise for stability

            # Predict velocity
            velocity = model(x_t, t_vals)

            # Masked Loss: nur auf Inpainting-Regionen
            vel_loss = F.mse_loss(velocity * mask, target * mask)
            loss = vel_loss / ACCUM_STEPS
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            train_loss += vel_loss.item() * ACCUM_STEPS

            if (step + 1) % 50 == 0:
                e = time.time() - t0
                avg_l = train_loss / (step + 1)
                print(
                    f"  Ep {epoch + 1:3d}/{epochs} | St {step + 1:3d}/{len(train_loader)} | L {avg_l:.6f} | {e:.0f}s",
                    flush=True,
                )

            if step >= 200:  # Limit steps per epoch
                break

        scheduler.step()
        avg_train = train_loss / min(200, len(train_loader))

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20:
                    break
                clean_v = vb["clean"].to(device).unsqueeze(-1)
                atten_v = vb["attenuated"].to(device).unsqueeze(-1)
                mask_v = vb["mask"].to(device).unsqueeze(-1)

                t_val = torch.full((BATCH_SIZE,), 0.5, device=device)
                noise_v = torch.randn_like(atten_v) * 0.01
                x_t_v = 0.5 * atten_v + 0.5 * clean_v + noise_v

                vel_v = model(x_t_v, t_val)
                target_v = (clean_v - atten_v) * mask_v
                val_loss += F.mse_loss(vel_v * mask_v, target_v * mask_v).item()
                vn += 1

        avg_val = val_loss / max(vn, 1)

        print(
            f"Ep {epoch + 1:3d}/{epochs} | Tr {avg_train:.6f} | Val {avg_val:.6f} | "
            f"LR {optimizer.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s",
            flush=True,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": avg_val,
            },
            LATEST_PT,
        )

        if avg_val < best_val - MIN_VAL_LOSS_IMPROVEMENT:
            best_val = avg_val
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val,
                    "production_ready": False,  # Not yet — still training
                },
                BEST_PT,
            )
            print(f"  >> Best: {best_val:.6f}")
        else:
            patience_counter += 1

        # §v10.600: Production-Readiness Gate (Ep 30 + convergence)
        if epoch >= 29 and best_val < 1e-4:
            production_ready = True
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val,
                    "production_ready": True,
                },
                BEST_PT,
            )
            print(f"  >> PRODUCTION READY: Ep {epoch + 1}, val={best_val:.6f}")

        # Early stopping
        if patience_counter >= early_stop_patience and epoch >= 20:
            print(f"\nEarly stop at epoch {epoch + 1} (patience exhausted, best={best_val:.6f})")
            break

    print(f"\nDone. Best val: {best_val:.6f} | Production ready: {production_ready} | {BEST_PT}")

    return {
        "epochs_run": epoch + 1,
        "best_val_loss": best_val,
        "production_ready": production_ready,
    }


if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser(description="Harmonic Inpainting Fine-Tuning (§v10.300)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    args = p.parse_args()

    # Save config for reproducibility (§v10.600 SOTA)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "chunk_sec": CHUNK_SEC,
        "sr": SR,
        "pretrained_dit": str(PRETRAINED_DIT),
    }
    with open(CHECKPOINT_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = train(args.epochs, args.lr, args.steps_per_epoch)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
