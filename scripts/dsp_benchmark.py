#!/usr/bin/env python3
"""Objektiver DSP-Benchmark: veraltete vs. moderne Rauschunterdrückungs-Stufen.

Analog zum Pitch-Benchmark: synthetische Ground-Truth (deterministisch, §G5),
referenzbasierte Metriken — kein Hörtest nötig. Misst die verworfenen DSPs
(Spec 04: ~~Spectral Subtraction~~, ~~Wiener 1984~~) gegen ihre Referenz-
Implementierungen UND gegen Auriks operative Fallbacks
(plugins/deepfilternet_v3_ii_plugin.py: _spectral_gating_fallback,
_omlsa_primary_fallback) — inklusive des vorbestehenden scipy-NOLA-
Kantenspikes (edge_peak_ratio).

Metriken:
    ref_snr_db:    Referenz-SNR des Outputs gegen das Clean-Referenzsignal
    lsd_db:        Log-Spectral-Distance zum Clean-Referenzsignal
    edge_peak_ratio: max|out| an den Rändern / max|out| im Inneren
                     (>3 = Kantenartefakt, vorbestehender NOLA-Spike)

Usage:
    python scripts/dsp_benchmark.py [--out models/dsp_benchmark_report.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "models" / "dsp_benchmark_report.json"
_SR = 48000
_EDGE = 1024


def synth_degraded(sr: int = _SR, dur: float = 2.0, snr_db: float = 5.0, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Clean-Referenz (Harmonik + Vibrato) + degradiert (Rauschen + Crackles)."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(dur * sr)) / sr
    f0 = 220.0 + 6.0 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    clean = np.zeros_like(phase)
    norm = 0.0
    for k in range(1, 7):
        clean += (1.0 / k) * np.sin(k * phase)
        norm += 1.0 / k
    clean = clean / norm
    noise = rng.randn(len(t))
    noise *= np.sqrt(np.mean(clean**2) + 1e-12) / (np.sqrt(np.mean(noise**2) + 1e-12) * 10 ** (snr_db / 20))
    deg = clean + noise
    for _ in range(8):  # deterministische Crackles
        pos = int(rng.randint(0, len(deg) - 1))
        deg[pos] += float(rng.uniform(0.2, 0.5))
    return clean.astype(np.float32), np.clip(deg, -1.0, 1.0).astype(np.float32)


def _stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> tuple[np.ndarray, int, int]:
    from scipy.signal import stft

    _, _, Z = stft(x, fs=_SR, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    return Z, n_fft, hop


def _istft(Z: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    from scipy.signal import istft

    # boundary=True passt zur zero-padded STFT — das korrekte Paar.
    # (Das Plugin nutzt boundary=None/padded=False + boundary=False → NOLA-Spike.)
    _, x = istft(Z, fs=_SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    return x[:length]


def spectral_gating_ref(x: np.ndarray, sr: int, floor: float = 0.05, k: float = 1.5) -> np.ndarray:
    """Referenz: Boll-Spectral-Gating mit korrekten STFT-Rändern (kein NOLA-Spike)."""
    Z, n_fft, hop = _stft(x)
    mag = np.abs(Z)
    noise_est = np.percentile(mag[:, :10], 50, axis=1, keepdims=True)
    mask = np.clip((mag - k * noise_est) / (mag + 1e-10), floor, 1.0)
    return np.asarray(_istft(mask * mag * np.exp(1j * np.angle(Z)), n_fft, hop, len(x)), dtype=np.float32)


def wiener_ref(x: np.ndarray, sr: int, floor: float = 0.05) -> np.ndarray:
    """Referenz: Wiener-Filter (Power-Subtraktion) mit korrekten Rändern."""
    Z, n_fft, hop = _stft(x)
    psd = np.abs(Z) ** 2
    noise_psd = np.mean(psd[:, :10], axis=1, keepdims=True)
    gain = np.clip(np.maximum(psd - noise_psd, 0.0) / (psd + 1e-12), floor, 1.0)
    return np.asarray(_istft(gain * Z, n_fft, hop, len(x)), dtype=np.float32)


def aurik_spectral_gating(x: np.ndarray, sr: int) -> np.ndarray:
    """Aurik operativ: DFN-Plugin-Spectral-Gating-Fallback (mit NOLA-Kantenverhalten)."""
    from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

    return DeepFilterNetV3Plugin._spectral_gating_fallback(x, sr)


def aurik_omlsa(x: np.ndarray, sr: int) -> np.ndarray:
    """Aurik operativ: OMLSA-Primärfallback des DFN-Plugins."""
    from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

    return DeepFilterNetV3Plugin._omlsa_primary_fallback(x, sr)


def aurik_banquet(x: np.ndarray, sr: int) -> np.ndarray:
    """Aurik operativ: Banquet-Vinyl-ONNX (Decrackle-ML-Pfad phase_09)."""
    from plugins.banquet_vinyl_plugin import process_vinyl

    return process_vinyl(np.asarray(x, dtype=np.float32), sr)


def metrics(clean: np.ndarray, out: np.ndarray, sr: int, edge: int = _EDGE) -> tuple[float, float, float]:
    out = np.asarray(out, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    n = min(len(out), len(clean))
    out, clean = out[:n], clean[:n]
    err = out - clean
    ref_snr = float(10 * np.log10(np.mean(clean**2) / (np.mean(err**2) + 1e-12)))
    # Log-Spectral-Distance
    Zc = np.abs(np.fft.rfft(clean * np.hanning(n)))
    Zo = np.abs(np.fft.rfft(out * np.hanning(n)))
    lsd = float(np.mean(np.abs(20 * np.log10((Zo + 1e-12) / (Zc + 1e-12)))))
    if n > 2 * edge:
        interior = float(np.max(np.abs(out[edge:-edge])) + 1e-9)
        edges = float(np.max(np.abs(np.concatenate([out[:edge], out[-edge:]]))))
    else:
        interior = edges = float(np.max(np.abs(out)) + 1e-9)
    return ref_snr, lsd, float(edges / interior)


METHODS: dict[str, Any] = {
    "spectral_gating_ref": spectral_gating_ref,
    "wiener_ref": wiener_ref,
    "aurik_spectral_gating": aurik_spectral_gating,
    "aurik_omlsa": aurik_omlsa,
    "aurik_banquet": aurik_banquet,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSP-Benchmark")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    for snr_db in (5.0, 15.0):
        for seed in (11, 12):
            clean, deg = synth_degraded(snr_db=snr_db, seed=seed)
            for name in args.methods:
                t0 = time.time()
                try:
                    out = METHODS[name](deg, _SR)
                    ref_snr, lsd, edge_ratio = metrics(clean, out, _SR)
                    results.append({
                        "snr_db": snr_db, "seed": seed, "method": name,
                        "runtime_s": round(time.time() - t0, 2),
                        "ref_snr_db": round(ref_snr, 1),
                        "lsd_db": round(lsd, 1),
                        "edge_peak_ratio": round(edge_ratio, 1),
                    })
                except Exception as exc:
                    results.append({"snr_db": snr_db, "seed": seed, "method": name, "error": str(exc)[:120]})

    summary: dict[str, Any] = {}
    for name in args.methods:
        rows = [r for r in results if r.get("method") == name and "ref_snr_db" in r]
        if not rows:
            summary[name] = {"n": 0}
            continue
        summary[name] = {
            "n": len(rows),
            "ref_snr_db_mean": round(float(np.mean([r["ref_snr_db"] for r in rows])), 1),
            "lsd_db_mean": round(float(np.mean([r["lsd_db"] for r in rows])), 1),
            "edge_peak_ratio_max": round(float(np.max([r["edge_peak_ratio"] for r in rows])), 1),
            "runtime_s_mean": round(float(np.mean([r["runtime_s"] for r in rows])), 2),
        }
    report = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
