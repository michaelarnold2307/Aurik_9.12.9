"""
Music Model Feature Flags (§v10.19)

Controls which model variant is active. Allows incremental deployment
and per-model rollback without code changes.

After §v10.19 training completion, set each flag to True to activate
the music-trained replacement. The original speech-trained model
remains as fallback.

Usage:
    from backend.core.music_model_flags import (
        use_df_musik, use_sgmse_musik, use_mp_senet_musik,
        use_miipher_dit, use_resemble_enhance,
        MUSIC_MODEL_PATHS,
    )

    model_path = MUSIC_MODEL_PATHS.get("dfn", fallback_path)
"""

# ── Feature Flags ───────────────────────────────────────────────────────────
# Set to True after training + ONNX export + A/B test passed.

use_df_musik: bool = True  # DFN Musik (§v10.15) replaces DeepFilterNet v3 (finetuned enc/dec/erb_dec)
use_sgmse_musik: bool = False  # SGMSE+ Musik (§v10.16) — kein Finetune-Checkpoint vorhanden
use_mp_senet_musik: bool = False  # MP-SENet Musik (§v10.17) — kein Finetune-Checkpoint vorhanden
use_miipher_dit: bool = True  # MIIPHER-DiT (§v10.14) replaces proprietary MIIPHER (flow_matching_dit.onnx)
use_bw_v5: bool = False  # BW-Reconstructor v5 — A1: HF-Gain-Gate nicht bestanden (0.73 < 1.02);
# redundant zu FlashSR/NVSR/DSP-SBR in Phase_06. Gated-Forschungsmodell.
use_harmonic_inpainting: bool = True  # Harmonic-Inpainting-DiT (§v10.300) — DiT-Finetune für gedämpfte Obertöne
use_whisper_denoiser: bool = False  # Whisper-Denoiser — DEPRECATED (Rev. 2026-08-16), nur A/B-Gate; NR trägt die Spec-04-Kette (DFN/SGMSE+/OMLSA)
use_resemble_enhance: bool = True  # Resemble Enhance — set to False after §v10.19

# ── Model Paths (relative to project root) ──────────────────────────────────

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent

MUSIC_MODEL_PATHS: dict[str, Path] = {
    # DFN Musik — three ONNX files
    "dfn_enc": _PROJECT_ROOT / "models" / "deepfilternet_v3_ii" / "finetuned" / "enc.onnx",
    "dfn_dec": _PROJECT_ROOT / "models" / "deepfilternet_v3_ii" / "finetuned" / "dec.onnx",
    "dfn_erb_dec": _PROJECT_ROOT / "models" / "deepfilternet_v3_ii" / "finetuned" / "erb_dec.onnx",
    # SGMSE+ Musik — TorchScript
    "sgmse": _PROJECT_ROOT / "models" / "sgmse_plus" / "finetuned" / "sgmse_musik.ts",
    # MP-SENet Musik — ONNX
    "mp_senet": _PROJECT_ROOT / "models" / "mp_senet" / "finetuned" / "mp_senet_musik.onnx",
    # MIIPHER-DiT — ONNX
    "miipher_dit": _PROJECT_ROOT / "models" / "miipher_dit" / "flow_matching_dit.onnx",
    # BW-Reconstructor v5 — bestes selbst trainiertes U-Net
    "bw": _PROJECT_ROOT / "models" / "bw_reconstructor" / "bw_reconstructor_v5.onnx",
    # Harmonic-Inpainting-DiT (§v10.300) — DiT-Finetune (FlowMatchingDiT state_dict)
    "harmonic_inpainting": _PROJECT_ROOT / "models" / "harmonic_inpainting" / "inpainting_best.pt",
    # Whisper-Denoiser (§v10.20) — Whisper-tiny (frozen) + 2M-Decoder (unet/decoder state_dicts)
    "whisper_denoiser": _PROJECT_ROOT / "models" / "miipher_dit" / "whisper_denoiser_best.pt",
    # Whisper-Encoder-Ersatz für MIIPHER-DiT (semantische Bedingung)
    "whisper_encoder": _PROJECT_ROOT / "models" / "whisper" / "whisper_tiny.onnx",
    # BigVGAN-Ersatz für MIIPHER-DiT-Vocoder
    "bigvgan": _PROJECT_ROOT / "models" / "bigvgan" / "bigvgan_v2.pth",
}

# ── Legacy paths (fallback) ─────────────────────────────────────────────────

LEGACY_MODEL_PATHS: dict[str, Path] = {
    "dfn_enc": _PROJECT_ROOT / "models" / "deepfilternet_v3_ii" / "enc.onnx",
    "dfn_dec": _PROJECT_ROOT / "models" / "deepfilternet_v3_ii" / "dec.onnx",
    "dfn_erb_dec": _PROJECT_ROOT / "models" / "deepfilternet_v3_ii" / "erb_dec.onnx",
    "sgmse": _PROJECT_ROOT / "models" / "sgmse_plus" / "sgmse_plus.ts",
    "mp_senet": _PROJECT_ROOT / "models" / "mp_senet" / "mp_senet.onnx",
    "miipher_dit": _PROJECT_ROOT / "models" / "miipher_dit" / "flow_matching_dit.onnx",
    # BW v1 als Fallback, falls v5 fehlt
    "bw": _PROJECT_ROOT / "models" / "bw_reconstructor" / "bw_reconstructor.onnx",
    # Harmonic-Inpainting: kein Legacy-Modell — DSP-Fallback (Phase_07 bestehende Synthese)
    # Whisper-Denoiser: kein Legacy-Modell — DFN bleibt primärer Denoiser
    # whisper_encoder/bigvgan: einziger bekannter lokaler Pfad — auch als Legacy hinterlegt
    "whisper_encoder": _PROJECT_ROOT / "models" / "whisper" / "whisper_tiny.onnx",
    "bigvgan": _PROJECT_ROOT / "models" / "bigvgan" / "bigvgan_v2.pth",
}


def resolve_model_path(model_key: str) -> Path | None:
    """Return active model path based on feature flags.

    Returns the music-trained path if the flag is set AND the file exists,
    otherwise falls back to the legacy path if it exists.
    Returns None if neither exists.
    """
    if model_key not in MUSIC_MODEL_PATHS:
        return None

    # Check which flag controls this model
    flag_map = {
        "dfn_enc": use_df_musik,
        "dfn_dec": use_df_musik,
        "dfn_erb_dec": use_df_musik,
        "sgmse": use_sgmse_musik,
        "mp_senet": use_mp_senet_musik,
        "miipher_dit": use_miipher_dit,
        "bw": use_bw_v5,
        "harmonic_inpainting": use_harmonic_inpainting,
        "whisper_denoiser": use_whisper_denoiser,
        # Ersatzpfade folgen dem DiT-Flag: nur aktiv wenn DiT selbst aktiv ist
        "whisper_encoder": use_miipher_dit,
        "bigvgan": use_miipher_dit,
    }
    use_music = flag_map.get(model_key, False)
    music_path = MUSIC_MODEL_PATHS.get(model_key)
    legacy_path = LEGACY_MODEL_PATHS.get(model_key)

    if use_music and music_path and music_path.exists():
        return music_path
    if legacy_path and legacy_path.exists():
        return legacy_path
    return None
