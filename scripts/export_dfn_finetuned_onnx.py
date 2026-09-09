"""DFN-Finetune → ONNX-Export (Wartungswerkzeug) — enc/erb_dec/dec MIT Alpha-Head.

§Fix 2026-09-08 (User-Vorgabe: „finetuned muss erhalten bleiben, jedoch mit
Alpha-Head“): Der aktive finetuned dec.onnx war ein veralteter Export ohne
Alpha-Head (df_fc_a) — der Plugin degradierte auf pure-DF (WARNING §P1-6).
Der Finetune-Checkpoint `models/deepfilternet_v3_ii/finetuned/dfn_musik_best.pt`
enthält die trainierten df_fc_a-Gewichte; dieses Skript re-exportiert alle
drei Modelle aus dem Checkpoint nach der offiziellen Export-Konvention
(Rikorose/DeepFilterNet df/scripts/export.py) — dec zusätzlich mit dem
Alpha-Ausgang (DFN2-analog: alpha = Sigmoid(Linear(gru_state))).

Konfiguration (per strict-load gegen den Checkpoint verifiziert):
    sr 48000, fft_size 960, hop_size 480, nb_erb 32, nb_df 96, df_order 5,
    conv_ch 16, emb_hidden_dim 256, df_hidden_dim 256, df_num_layers 3,
    emb_num_layers 2, conv_lookahead 0, enc_lin_groups 1, lin_groups 1,
    group_shuffle False.

Nutzung (venv mit libdf + torch, z. B. .venv_aurik):
    python scripts/export_dfn_finetuned_onnx.py [--out models/deepfilternet_v3_ii/finetuned]

Die erzeugten ONNX-Dateien sind gitignored (models/); dieses Skript ist die
reproduzierbare Quelle der Wahrheit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "models" / "deepfilternet_v3_ii"
_DF_PKG = _MODEL_DIR / "DeepFilterNet"
_CHECKPOINT = _MODEL_DIR / "finetuned" / "dfn_musik_best.pt"

# Per strict-load verifizierte Finetune-Konfiguration.
_CONFIG: dict[str, tuple[object, type, str]] = {
    "sr": (48000, int, "df"),
    "fft_size": (960, int, "df"),
    "hop_size": (480, int, "df"),
    "nb_erb": (32, int, "df"),
    "nb_df": (96, int, "df"),
    "df_order": (5, int, "df"),
    "conv_ch": (16, int, "deepfilternet"),
    "emb_hidden_dim": (256, int, "deepfilternet"),
    "df_hidden_dim": (256, int, "deepfilternet"),
    "df_num_layers": (3, int, "deepfilternet"),
    "emb_num_layers": (2, int, "deepfilternet"),
    "conv_lookahead": (0, int, "deepfilternet"),
    "enc_linear_groups": (16, int, "deepfilternet"),
    "linear_groups": (1, int, "deepfilternet"),
    "group_shuffle": (False, bool, "deepfilternet"),
}


class DecWithAlpha(torch.nn.Module):
    """DFN3-DeepDecoder mit zusätzlichem Alpha-Ausgang (DFN2-analog).

    alpha = Sigmoid(Linear(gru_state)) — berechnet AUS dem GRU-Hidden-State
    vor dem df_out-Tanh, wie in DFN2 (df/deepfilternet.py). Die df_fc_a-
    Gewichte stammen aus dem Checkpoint.
    """

    def __init__(self, dec: torch.nn.Module) -> None:
        super().__init__()
        self.df_gru = dec.df_gru
        self.df_skip = dec.df_skip
        self.df_convp = dec.df_convp
        self.df_out = dec.df_out
        self.df_fc_a = dec.df_fc_a
        self.df_bins = dec.df_bins
        self.df_out_ch = dec.df_out_ch

    def forward(self, emb: torch.Tensor, c0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        alpha = self.df_fc_a(c)  # [B, T, 1]
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)
        c = self.df_out(c)
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0
        return c, alpha


def _export(path: Path, model: torch.nn.Module, inputs: tuple[torch.Tensor, ...],
            input_names: list[str], output_names: list[str],
            dynamic_axes: dict[str, dict[int, str]], opset: int = 14) -> None:
    model = model.eval().cpu()
    with torch.no_grad():
        torch.onnx.export(
            model=model,
            f=str(path),
            args=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            keep_initializers_as_inputs=False,
        )
    print(f"  exportiert: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=_MODEL_DIR / "finetuned")
    args = parser.parse_args()

    sys.path.insert(0, str(_DF_PKG))
    from df.config import config  # noqa: E402

    config.use_defaults()
    for option, (value, cast_type, section) in _CONFIG.items():
        config.set(option, value, cast_type, section)  # type: ignore[arg-type]

    from df.deepfilternet3 import init_model  # noqa: E402
    from libdf import DF  # noqa: E402

    model = init_model().eval().cpu()
    sd = torch.load(str(_CHECKPOINT), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"], strict=True)
    print(f"Checkpoint geladen: {_CHECKPOINT.name} (epoch {sd.get('epoch')}, val_loss {sd.get('val_loss')})")

    df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)

    from df.enhance import df_features  # noqa: E402

    with torch.no_grad():
        audio = torch.randn((1, 48000))
        spec, feat_erb, feat_spec = df_features(audio, df_state, 96, device="cpu")
        feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # [B,2,S,F'] (offizielle Konvention)

        # ── enc ─────────────────────────────────────────────────────────────
        e0, e1, e2, e3, emb, c0, lsnr = model.enc(feat_erb, feat_spec)
        _export(
            args.out / "enc.onnx",
            model.enc,
            (feat_erb, feat_spec),
            ["feat_erb", "feat_spec"],
            ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
            {
                "feat_erb": {2: "S"}, "feat_spec": {2: "S"},
                "e0": {2: "S"}, "e1": {2: "S"}, "e2": {2: "S"}, "e3": {2: "S"},
                "emb": {1: "S"}, "c0": {2: "S"}, "lsnr": {1: "S"},
            },
        )

        # ── erb_dec ──────────────────────────────────────────────────────────
        m_out = model.erb_dec(emb.clone(), e3, e2, e1, e0)
        _export(
            args.out / "erb_dec.onnx",
            model.erb_dec,
            (emb.clone(), e3, e2, e1, e0),
            ["emb", "e3", "e2", "e1", "e0"],
            ["m"],
            {"emb": {1: "S"}, "e3": {2: "S"}, "e2": {2: "S"}, "e1": {2: "S"}, "e0": {2: "S"}, "m": {2: "S"}},
        )

        # ── dec (MIT Alpha-Head) ────────────────────────────────────────────
        dec_alpha = DecWithAlpha(model.df_dec)
        coefs, alpha = dec_alpha(emb.clone(), c0)
        print(f"  dec-Referenz: coefs {tuple(coefs.shape)}, alpha {tuple(alpha.shape)}")
        _export(
            args.out / "dec.onnx",
            dec_alpha,
            (emb.clone(), c0),
            ["emb", "c0"],
            ["coefs", "alpha"],
            {"emb": {1: "S"}, "c0": {2: "S"}, "coefs": {1: "S"}, "alpha": {1: "S"}},
        )

    print("Fertig. Backup der alten Dateien nicht vergessen (z. B. *.stale).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
