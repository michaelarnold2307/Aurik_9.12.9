#!/usr/bin/env python3
"""
Export DeepFilterNet3 checkpoint → ONNX (enc, dec, erb_dec).

Converts the trained PyTorch model from DFN Musik Fine-Tuning (§v10.15)
into the three ONNX files expected by the Aurik plugin:
  - enc.onnx:  feat_erb [1,1,T,32] + feat_spec [1,2,T,96] → e0,e1,e2,e3,emb,c0
  - erb_dec.onnx: emb [1,T,512] + e3,e2,e1,e0 → mask [1,1,T,32]
  - dec.onnx:     emb [1,T,512] + c0 [1,64,T,96] → coefs [1,T,96,10]

Architecture: DeepFilterNet3 (2.4M params), 48kHz, FFT=960, Hop=480.

Usage:
    python scripts/export_df_musik_onnx.py [--checkpoint PATH]
"""

import sys
from pathlib import Path

import onnx
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "models" / "deepfilternet_v3_ii" / "DeepFilterNet"))
sys.path.insert(0, str(_PROJECT / "models" / "deepfilternet_v3_ii" / "pyDF-data"))

CHECKPOINT_PATH = _PROJECT / "models/deepfilternet_v3_ii/finetuned/dfn_musik_best.pt"
OUT_DIR = _PROJECT / "models/deepfilternet_v3_ii/finetuned"


def load_model(checkpoint_path: str, device="cpu"):
    """Load DeepFilterNet3 with fine-tuned weights."""
    from df.config import config

    config.use_defaults()
    from df.deepfilternet3 import init_model

    model = init_model().to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model


def export_enc(model, out_path: str):
    """Export encoder: feat_erb + feat_spec → e0,e1,e2,e3,emb,c0"""
    print("Exporting encoder (enc.onnx)...")
    dummy_erb = torch.randn(1, 1, 100, 32)  # [B, 1, T, 32]
    dummy_spec = torch.randn(1, 2, 100, 96)  # [B, 2, T, 96]

    # The encoder forward: enc(feat_erb, feat_spec) → e0,e1,e2,e3,emb,c0,lsnr
    torch.onnx.export(
        model.enc,
        (dummy_erb, dummy_spec),
        out_path,
        input_names=["feat_erb", "feat_spec"],
        output_names=["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
        dynamic_axes={
            "feat_erb": {0: "batch", 2: "time"},
            "feat_spec": {0: "batch", 3: "time"},
            "e0": {0: "batch", 2: "time"},
            "e1": {0: "batch", 2: "time"},
            "e2": {0: "batch", 2: "time"},
            "e3": {0: "batch", 2: "time"},
            "emb": {0: "batch", 1: "time"},
            "c0": {0: "batch", 2: "time"},
            "lsnr": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(out_path)
    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"  ✅ {out_path} ({size_mb:.1f} MB)")


def export_erb_dec(model, out_path: str):
    """Export ERB decoder: emb + e3,e2,e1,e0 → mask"""
    print("Exporting ERB decoder (erb_dec.onnx)...")
    # Correct shapes from encoder output (conv_ch=16)
    dummy_emb = torch.randn(1, 100, 128)  # [B, T, emb_dim=128]
    dummy_e3 = torch.randn(1, 16, 100, 8)  # [B, 16, T, F/4=8]
    dummy_e2 = torch.randn(1, 16, 100, 8)  # [B, 16, T, F/4=8]
    dummy_e1 = torch.randn(1, 16, 100, 16)  # [B, 16, T, F/2=16]
    dummy_e0 = torch.randn(1, 16, 100, 32)  # [B, 16, T, F=32]

    torch.onnx.export(
        model.erb_dec,
        (dummy_emb, dummy_e3, dummy_e2, dummy_e1, dummy_e0),
        out_path,
        input_names=["emb", "e3", "e2", "e1", "e0"],
        output_names=["erb_mask"],
        dynamic_axes={
            "emb": {0: "batch", 1: "time"},
            "e3": {0: "batch", 2: "time"},
            "e2": {0: "batch", 2: "time"},
            "e1": {0: "batch", 2: "time"},
            "e0": {0: "batch", 2: "time"},
            "erb_mask": {0: "batch", 2: "time"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(out_path)
    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"  ✅ {out_path} ({size_mb:.1f} MB)")


def export_dec(model, out_path: str):
    """Export DF decoder: emb + c0 → coefs (DFN3 ohne Alpha-Head).

    §P1-6 (2026-09-08): Der trainierte DeepFilterNet3-Forward wendet
    df_op(coefs) OHNE Alpha-Blend an (df_fc_a ist definiert, aber im
    Forward unbenutzt). Der Export liefert deshalb bewusst nur coefs;
    das Plugin behandelt fehlendes alpha als pure DF.
    """
    print("Exporting DF decoder (dec.onnx)...")
    dummy_emb = torch.randn(1, 100, 128)  # [B, T, emb_dim=128]
    dummy_c0 = torch.randn(1, 16, 100, 96)  # [B, conv_ch=16, T, 96]

    torch.onnx.export(
        model.df_dec,
        (dummy_emb, dummy_c0),
        out_path,
        input_names=["emb", "c0"],
        output_names=["coefs"],
        dynamic_axes={
            "emb": {0: "batch", 1: "time"},
            "c0": {0: "batch", 2: "time"},
            "coefs": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(out_path)
    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"  ✅ {out_path} ({size_mb:.1f} MB)")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Export DeepFilterNet3 → ONNX")
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return

    print(f"Loading checkpoint: {ckpt}")
    model = load_model(str(ckpt))

    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    export_enc(model, str(out / "enc.onnx"))
    export_erb_dec(model, str(out / "erb_dec.onnx"))
    export_dec(model, str(out / "dec.onnx"))

    print(f"\n✅ Done. ONNX models in {out}/")
    print("   To activate: copy over models/deepfilternet_v3_ii/{enc,dec,erb_dec}.onnx")


if __name__ == "__main__":
    main()
