#!/usr/bin/env python3
"""Objektiver Pitch-Tracker-Benchmark: CREPE (2018) vs. FCPE (2023) vs. RMVPE (2023).

Pitch-Tracking ist der eine ML-Task mit objektiver Ground-Truth: synthetische
Signale mit bekannter F0-Trajektorie (deterministisch, §G5). Kein Hörtest nötig —
die Zahlen entscheiden die Challenger-Runde für die Wow/Flutter- und
Speed/Pitch-Pfade (CREPE ist dort 12-fach verdrahtet, Spec-verworfen,
Nachfolger FCPE/RMVPE sind im Haus).

Metriken (auf gemeinsamen Zeitraster, nur beidseitig voiced Frames):
    - cents_rmse:  F0-Fehler in Cents (ohne Oktav-/Grobe Fehler)
    - gpe_rate:    Gross Pitch Error (> 50 Cents, nicht-Oktave)
    - octave_rate: Oktavverwechslungen
    - voiced_f1:   Voicing-Erkennung

Usage:
    python scripts/pitch_tracker_benchmark.py [--out models/pitch_benchmark_report.json]
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
DEFAULT_OUT = ROOT / "models" / "pitch_benchmark_report.json"
_SR = 48000
_SEED = 7

CASES = ("steady", "vibrato", "glide", "low", "high")
CONDITIONS = ("clean", "snr10", "wow")


def _f0_of(case: str, t: np.ndarray, dur: float) -> np.ndarray:
    if case == "steady":
        return np.full_like(t, 220.0)
    if case == "vibrato":
        return 220.0 + 6.0 * np.sin(2 * np.pi * 5.5 * t)
    if case == "glide":
        return 150.0 + (400.0 - 150.0) * t / max(dur, 1e-9)
    if case == "low":
        return np.full_like(t, 80.0)
    return np.full_like(t, 800.0)  # high


def synth(case: str, condition: str, dur: float = 5.0, seed: int = _SEED) -> tuple[np.ndarray, np.ndarray]:
    """Signal + F0-Ground-Truth (Hz pro Sample). Determinismus: fester Seed."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(dur * _SR)) / _SR
    f0 = _f0_of(case, t, dur).astype(np.float64)
    if condition == "wow":
        # 0.5 % Zeit-Warp bei 2 Hz (Wow-Bereich): F0-Faktor = 1 + A·cos(2π·f_w·t)
        A, f_w = 0.005, 2.0
        t_warp = t + A * np.sin(2 * np.pi * f_w * t) / (2 * np.pi * f_w)
        f0 = f0 * (1.0 + A * np.cos(2 * np.pi * f_w * t))
    phase = 2 * np.pi * np.cumsum(f0) / _SR
    sig = np.zeros_like(phase)
    norm = 0.0
    for k in range(1, 7):
        sig += (1.0 / k) * np.sin(k * phase)
        norm += 1.0 / k
    sig = sig / norm
    if condition == "wow":
        sig = np.interp(t, t_warp, sig)
    if condition == "snr10":
        noise = rng.randn(len(sig))
        noise *= np.sqrt(np.mean(sig**2) + 1e-12) / (np.sqrt(np.mean(noise**2) + 1e-12) * 10 ** (10 / 20))
        sig = sig + noise
    return np.clip(sig, -1.0, 1.0).astype(np.float32), f0.astype(np.float64)


def _common_grid(times: np.ndarray, f0: np.ndarray, truth_t: np.ndarray, truth_f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpoliert Truth auf das Zeitraster des Trackers."""
    return times, np.interp(times, truth_t, truth_f0)


def _metrics(est_times: np.ndarray, est_f0: np.ndarray, est_voiced: np.ndarray, truth_f0: np.ndarray) -> dict[str, float] | None:
    both = est_voiced & np.isfinite(est_f0) & (truth_f0 > 0)
    if int(both.sum()) < 20:
        return None
    ratio = np.log2(est_f0[both] / truth_f0[both])
    oct_idx = np.round(ratio)
    # Oktave nur bei |oct_idx| >= 1 zählen — ratio ~ 0 ist der korrekte Frame!
    oct_mask = (np.abs(oct_idx) >= 1) & (np.abs(ratio - oct_idx) < 0.15)
    clean = (~oct_mask) & (np.abs(ratio) <= np.log2(2 ** (50 / 1200)))
    cents = 1200.0 * ratio[clean]
    truth_voiced = np.ones_like(est_voiced, dtype=bool)  # synthetisch: immer voiced
    tp = int(np.sum(est_voiced & truth_voiced))
    fp = int(np.sum(est_voiced & ~truth_voiced))
    fn = int(np.sum(~est_voiced & truth_voiced))
    return {
        "cents_rmse": float(np.sqrt(np.mean(cents**2))) if cents.size else float("nan"),
        "gpe_rate": float(np.mean((~oct_mask) & (np.abs(ratio) > np.log2(2 ** (50 / 1200))))) if both.sum() else 0.0,
        "octave_rate": float(np.mean(oct_mask)),
        "voiced_f1": float(2 * tp / (2 * tp + fp + fn + 1e-9)),
    }


def run_tracker(name: str, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if name == "crepe":
        from plugins.crepe_plugin import analyze_pitch

        r = analyze_pitch(audio, _SR)
        return r.times_s, r.f0_hz, r.voiced_prob >= 0.5, "crepe"
    if name == "fcpe":
        from plugins.fcpe_plugin import analyze_pitch

        r = analyze_pitch(audio, _SR)
        return r.times_s, r.f0_hz, r.voiced_prob >= 0.5, "fcpe"
    from plugins.rmvpe_plugin import analyze_pitch, get_rmvpe_plugin

    _inst = get_rmvpe_plugin()
    r = analyze_pitch(audio, _SR)
    times = getattr(r, "times", None)
    if times is None:
        times = np.arange(len(r.f0), dtype=np.float64) * 0.01
    voiced = np.isfinite(r.f0) & (r.f0 > 0)
    # §V6-Transparenz: ONNX kaputt → pYIN-Fallback explizit ausweisen
    model = str(getattr(r, "model_used", None) or "")
    if not model:
        model = "rmvpe" if getattr(_inst, "_onnx_loaded", False) else "rmvpe_pyin_fallback"
    return np.asarray(times, dtype=np.float64), r.f0, voiced, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pitch-Tracker-Benchmark")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trackers", nargs="+", default=["crepe", "fcpe", "rmvpe"])
    args = parser.parse_args(argv)

    t_all = np.arange(int(5.0 * _SR)) / _SR
    results: list[dict[str, Any]] = []
    for case in CASES:
        for cond in CONDITIONS:
            audio, truth_f0 = synth(case, cond)
            for tracker in args.trackers:
                t0 = time.time()
                try:
                    times, f0, voiced, model = run_tracker(tracker, audio)
                    grid, truth = _common_grid(np.asarray(times, dtype=np.float64), f0, t_all, truth_f0)
                    m = _metrics(grid, np.asarray(f0, dtype=np.float64), np.asarray(voiced, dtype=bool), truth)
                    results.append({
                        "case": case, "condition": cond, "tracker": tracker,
                        "model": model, "runtime_s": round(time.time() - t0, 2),
                        "metrics": m,
                    })
                except Exception as exc:
                    results.append({
                        "case": case, "condition": cond, "tracker": tracker,
                        "error": str(exc)[:120],
                    })

    # Zusammenfassung pro Tracker
    summary: dict[str, Any] = {}
    for tracker in args.trackers:
        rows = [r["metrics"] for r in results if r.get("tracker") == tracker and r.get("metrics")]
        if not rows:
            summary[tracker] = {"n": 0}
            continue
        summary[tracker] = {
            "n": len(rows),
            "cents_rmse_mean": round(float(np.nanmean([r["cents_rmse"] for r in rows])), 1),
            "gpe_rate_mean": round(float(np.mean([r["gpe_rate"] for r in rows])), 4),
            "octave_rate_mean": round(float(np.mean([r["octave_rate"] for r in rows])), 4),
            "voiced_f1_mean": round(float(np.mean([r["voiced_f1"] for r in rows])), 3),
            "runtime_s_mean": round(float(np.mean([r["runtime_s"] for r in results if r.get("tracker") == tracker and "runtime_s" in r])), 2),
        }

    report = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
