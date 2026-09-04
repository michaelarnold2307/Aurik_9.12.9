#!/usr/bin/env python3
"""
Unified ONNX/TorchScript export pipeline for all music-trained models (§v10.14–17).

Exports fine-tuned checkpoints to production-ready ONNX/TorchScript files.
Replaces the speech-trained originals with music-adapted versions.

Usage:
    # Export all available music-trained models:
    python scripts/export_all_musik_models.py

    # Export specific model:
    python scripts/export_all_musik_models.py --model miipher_dit
    python scripts/export_all_musik_models.py --model dfn
    python scripts/export_all_musik_models.py --model sgmse

Models:
  miipher_dit    FlowMatchingDiT (201M) → models/miipher_dit/flow_matching_dit.onnx
  dfn            DeepFilterNet3 (2.4M)  → models/deepfilternet_v3_ii/{enc,dec,erb_dec}.onnx
  sgmse          SGMSE+ (65M)           → models/sgmse_plus/sgmse_musik.ts
  mp_senet       MP-SENet (~1M)         → models/mp_senet/mp_senet_musik.onnx
"""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = "python3"
_PROJECT = Path(__file__).resolve().parent.parent
MODELS_DIR = _PROJECT / "models"


def run(cmd: list[str], desc: str):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(_PROJECT))
    if result.returncode != 0:
        print(f"  ❌ Failed with code {result.returncode}")
    else:
        print("  ✅ Done")


def export_all(python: str = PYTHON, only: str | None = None):
    print(f"Export pipeline — Python: {python}")
    print(f"Models dir: {MODELS_DIR}")

    # ── MIIPHER-DiT (§v10.14) ─────────────────────────────────────────
    if only in (None, "miipher_dit", "dit"):
        ckpt = MODELS_DIR / "miipher_dit" / "checkpoint_best.pt"
        onnx_out = MODELS_DIR / "miipher_dit" / "flow_matching_dit.onnx"
        if ckpt.exists():
            run(
                [
                    PYTHON,
                    "scripts/export_miipher_dit_onnx.py",
                    "--checkpoint",
                    str(ckpt),
                    "--output",
                    str(onnx_out),
                ],
                "MIIPHER-DiT → ONNX",
            )
        else:
            print(f"\n  ⚠️  MIIPHER-DiT checkpoint not found: {ckpt}")

    # ── DeepFilterNet Musik (§v10.15) ─────────────────────────────────
    if only in (None, "dfn", "deepfilternet"):
        ckpt = MODELS_DIR / "deepfilternet_v3_ii" / "finetuned" / "dfn_musik_best.pt"
        if ckpt.exists():
            run(
                [
                    PYTHON,
                    "scripts/export_df_musik_onnx.py",
                    "--checkpoint",
                    str(ckpt),
                ],
                "DeepFilterNet Musik → ONNX (enc, dec, erb_dec)",
            )
        else:
            print(f"\n  ⚠️  DFN checkpoint not found: {ckpt}")

    # ── SGMSE+ Musik (§v10.16) ────────────────────────────────────────
    if only in (None, "sgmse", "sgmse_plus"):
        ckpt = MODELS_DIR / "sgmse_plus" / "finetuned" / "sgmse_musik_best.pt"
        ts_out = MODELS_DIR / "sgmse_plus" / "sgmse_musik.ts"
        if ckpt.exists():
            run(
                [
                    PYTHON,
                    "scripts/export_sgmse_musik_ts.py",
                    "--checkpoint",
                    str(ckpt),
                    "--output",
                    str(ts_out),
                ],
                "SGMSE+ TorchScript export",
            )
        else:
            print(f"\n  ⚠️  SGMSE+ checkpoint not found: {ckpt}")

    # ── MP-SENet Musik (§v10.17) ──────────────────────────────────────
    if only in (None, "mp_senet", "mpsenet"):
        ckpt = MODELS_DIR / "mp_senet" / "finetuned" / "mp_senet_musik_best.pt"
        onnx_out = MODELS_DIR / "mp_senet" / "mp_senet_musik.onnx"
        if ckpt.exists():
            print("\n  ⚠️  MP-SENet uses existing export script.")
            print(f"     Checkpoint: {ckpt}")
            print("     Run: python scripts/export_mp_senet_onnx.py")
        else:
            print(f"\n  ⚠️  MP-SENet checkpoint not found: {ckpt}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Export pipeline complete")
    print(f"{'=' * 60}")
    _print_status()


def _print_status():
    """Show which models have trained checkpoints vs ONNX exports."""
    models = [
        ("MIIPHER-DiT", "miipher_dit/checkpoint_best.pt", "miipher_dit/flow_matching_dit.onnx"),
        ("DFN Musik", "deepfilternet_v3_ii/finetuned/dfn_musik_best.pt", "deepfilternet_v3_ii/finetuned/enc.onnx"),
        ("SGMSE+ Musik", "sgmse_plus/finetuned/sgmse_musik_best.pt", "sgmse_plus/sgmse_musik.ts"),
        ("MP-SENet Musik", "mp_senet/finetuned/mp_senet_musik_best.pt", "mp_senet/mp_senet_musik.onnx"),
    ]
    print(f"\n  {'Model':<18} {'Checkpoint':<12} {'ONNX/TorchScript':<12}")
    print(f"  {'-' * 18} {'-' * 12} {'-' * 12}")
    for name, ckpt_rel, onnx_rel in models:
        ckpt_ok = "✅" if (MODELS_DIR / ckpt_rel).exists() else "❌"
        onnx_ok = "✅" if (MODELS_DIR / onnx_rel).exists() else "❌"
        print(f"  {name:<18} {ckpt_ok:<12} {onnx_ok:<12}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export all music-trained models")
    p.add_argument("--python", type=str, default="python3", help="Python interpreter")
    p.add_argument("--model", type=str, default=None, help="Export only this model (miipher_dit|dfn|sgmse|mp_senet)")
    args = p.parse_args()
    export_all(args.python, args.model)
    _print_status()
