#!/usr/bin/env python3
"""Vokal-Challenger-Runde SNR<10 dB: Incumbent (MIIPHER-DiT) vs. Kandidat (SGMSE+).

Die MIIPHER-Stufe ist Auriks Pfad für stark degradierten Gesang (SNR < 10 dB,
Spec 04 „Last-Resort“). Incumbent ist der offene Flow-Matching-DiT
(plugins/miipher_dit_plugin.py, §v10.14 — Ersatz des proprietären Google-MIIPHER);
Kandidat ist SGMSE+ (plugins/sgmse_plugin.py, Richter et al. 2022, lokal
finetunet). Bewertung über die Hörrunde (challenger_round.py decide).

Deterministisch (§G5): fixe Seeds für Rausch-Mischung, feste Reihenfolge.
Fehlende Modelle werden als Probleme protokolliert (§V6), nie still ersetzt.

Usage:
    python scripts/prepare_vocal_snr_round.py [--out <dir>] [--snr-db 5.0]
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
VOCAL_DIR = ROOT / "tests" / "real_world_validation" / "test_library" / "vocals"
DEFAULT_OUT = ROOT / "audit" / "listening_study" / "round_2026-08-15" / "vocal_snr_round"
_SR = 48000
_SEED = 2026


def make_task_audio(audio: np.ndarray, sr: int, snr_db: float, seed: int) -> tuple[np.ndarray, float]:
    """Mischt deterministisches Rauschen auf Ziel-SNR und gibt (task, snr_ist) zurück."""
    rng = np.random.RandomState(seed)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != _SR:
        from scipy.signal import resample_poly

        g = int(np.gcd(sr, _SR))
        audio = resample_poly(audio, _SR // g, sr // g).astype(np.float32)
    sig_pow = float(np.mean(audio**2)) + 1e-12
    noise = rng.randn(len(audio)).astype(np.float32)
    noise_pow = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10)))
    noise = (noise * scale).astype(np.float32)
    task = np.clip(audio + noise, -1.0, 1.0).astype(np.float32)
    snr_ist = float(10 * np.log10(sig_pow / (np.mean(noise**2) + 1e-12)))
    return task, snr_ist


def make_codec_task(audio: np.ndarray, sr: int, seed: int) -> np.ndarray:
    """Codec-degradierter Vokal (mp3_low-Proxy, deklariert synthetisch — Corpus-Konvention)."""
    rng = np.random.RandomState(seed)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != _SR:
        from scipy.signal import resample_poly

        g = int(np.gcd(sr, _SR))
        audio = resample_poly(audio, _SR // g, sr // g).astype(np.float32)
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 8000.0, btype="low", fs=_SR, output="sos")
    audio = sosfiltfilt(sos, audio).astype(np.float32)
    audio = audio + rng.normal(0.0, 0.003, audio.shape).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def run_models(
    out_dir: Path,
    snr_db: float,
    skip_candidate: bool = False,
    task: str = "noise",
    material: str = "mp3_low",
    restorability: float = 20.0,
) -> dict[str, Any]:
    import soundfile as sf

    out_dir = Path(out_dir)
    (out_dir / "task").mkdir(parents=True, exist_ok=True)
    (out_dir / "incumbent_dit").mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_sgmse").mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    problems: list[str] = []

    dit = None
    sgmse = None
    try:
        from plugins.miipher_dit_plugin import get_miipher_dit

        dit = get_miipher_dit()
        if getattr(dit, "_fallback_active", True):
            problems.append("Incumbent DiT nicht geladen — Fallback aktiv (kein SOTA-Vergleich möglich)")
    except Exception as exc:
        problems.append(f"Incumbent DiT Import fehlgeschlagen: {exc}")
    try:
        from plugins.sgmse_plugin import get_sgmse_plus_plugin

        sgmse = get_sgmse_plus_plugin()
    except Exception as exc:
        problems.append(f"Kandidat SGMSE+ Import fehlgeschlagen: {exc}")

    for idx, vocal_path in enumerate(sorted(VOCAL_DIR.glob("*.wav"))):
        iid = vocal_path.stem
        audio, sr = sf.read(str(vocal_path), dtype="float32")
        if task == "codec":
            task_audio = make_codec_task(audio, sr, seed=_SEED + idx)
            snr_ist = 0.0
            task_label = "codec_mp3low_proxy"
        else:
            task_audio, snr_ist = make_task_audio(audio, sr, snr_db, seed=_SEED + idx)
            task_label = f"snr{snr_ist:.1f}db"
        task_path = out_dir / "task" / f"{iid}_{task_label}.wav"
        sf.write(str(task_path), task_audio, _SR)

        entry: dict[str, Any] = {
            "item_id": iid,
            "task": str(task_path),
            "snr_db_ist": round(snr_ist, 2),
            "incumbent": None,
            "candidate": None,
            "problems": [],
        }

        if dit is not None and not getattr(dit, "_fallback_active", True):
            t0 = time.time()
            try:
                res_dit = dit.enhance(
                    task_audio, _SR, material=material, restorability_score=restorability
                )
                out_dit = np.asarray(getattr(res_dit, "audio", res_dit), dtype=np.float32)
                p = out_dir / "incumbent_dit" / f"{iid}_dit.wav"
                sf.write(str(p), out_dit, _SR)
                entry["incumbent"] = str(p)
                entry["incumbent_runtime_s"] = round(time.time() - t0, 1)
                entry["incumbent_model_used"] = str(getattr(res_dit, "model_used", "?"))
            except Exception as exc:
                entry["problems"].append(f"DiT-Lauf fehlgeschlagen: {exc}")
        else:
            entry["problems"].append("Incumbent DiT nicht verfügbar")

        if sgmse is not None and not skip_candidate:
            t0 = time.time()
            try:
                res = sgmse.enhance(task_audio, _SR, sigma=0.5, max_runtime_s=600.0)
                out_s = np.asarray(getattr(res, "audio", res), dtype=np.float32)
                p = out_dir / "candidate_sgmse" / f"{iid}_sgmse.wav"
                sf.write(str(p), out_s, _SR)
                entry["candidate"] = str(p)
                entry["candidate_runtime_s"] = round(time.time() - t0, 1)
                entry["candidate_model_used"] = str(getattr(res, "model_used", "?"))
            except Exception as exc:
                entry["problems"].append(f"SGMSE+-Lauf fehlgeschlagen: {exc}")
        elif sgmse is None:
            entry["problems"].append("Kandidat SGMSE+ nicht verfügbar")

        items.append(entry)

    manifest = {
        "round": "vocal_snr_5db",
        "task": f"deterministisches Rauschen, SNR {snr_db} dB (Seed {_SEED}+idx)",
        "incumbent": "MIIPHER-DiT (Flow-Matching, §v10.14)",
        "candidate": "SGMSE+ (Richter et al. 2022, lokal)",
        "items": items,
        "problems": problems,
    }
    (out_dir / "round_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vokal-Challenger-Runde SNR<10 dB")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--skip-candidate", action="store_true", help="nur Incumbent (DiT) ausführen")
    parser.add_argument("--task", choices=["noise", "codec"], default="noise", help="Degradations-Art")
    parser.add_argument("--material", default="mp3_low", help="Material-Kontext für den DiT (Gate 1)")
    parser.add_argument("--restorability", type=float, default=20.0, help="Restorability-Kontext für den DiT")
    args = parser.parse_args(argv)

    manifest = run_models(
        args.out,
        args.snr_db,
        skip_candidate=args.skip_candidate,
        task=args.task,
        material=args.material,
        restorability=args.restorability,
    )
    inc = sum(1 for i in manifest["items"] if i["incumbent"])
    cand = sum(1 for i in manifest["items"] if i["candidate"])
    print(f"Runde: {inc} Incumbent, {cand} Kandidat (von {len(manifest['items'])} Items)")
    for p in manifest["problems"]:
        print(f"  PROBLEM: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
