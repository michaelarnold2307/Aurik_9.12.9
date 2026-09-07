#!/usr/bin/env python3
"""
§v10.950: Model Zoo Registry — alle Modelle sichtbar, keins brachliegend.

Problem: 7 große SOTA-Modelle (AudioLDM2, CQTDiff, DiffWave, SGMSE+,
MP-SENet, MDX23C, MelBandRoformer) lagen ohne Referenzen in der Kette —
niemand wusste, was sie können, was sie brauchen, ob sie laden.

Lösung: Zentrale Registry mit:
  - Verifizierten ONNX-I/O-Shapes (gemessen, nicht geraten)
  - Zweck-Klassifikation: repair / generation / separation
  - Status: aktiv / verfügbar / benötigt-Kalibrierung
  - probe(): prüft Ladebarkeit ohne Inferenz

Damit ist jedes Modell SICHTBAR und die Aktivierungs-Entscheidung explizit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parent.parent


@dataclass
class ModelEntry:
    name: str
    path: str
    purpose: str  # "repair" | "generation" | "separation" | "vocoder"
    input_shapes: str  # gemessene I/O-Beschreibung
    status: str  # "active" | "available" | "needs_calibration" | "generation_only"
    integration: str | None = None  # wo/wie aktiviert
    notes: str = ""


# Verifizierte I/O-Shapes (gemessen via onnxruntime, 2026-08-13)
MODEL_ZOO: list[ModelEntry] = [
    ModelEntry(
        name="mp_senet",
        path="models/mp_senet/mp_senet.onnx",
        purpose="repair",
        input_shapes="IN (noisy_amp [B,201,T], noisy_pha [B,201,T]) → OUT denoised_amp [B,201,T]",
        status="available",
        integration="coordinated_repair._run_mp_senet_vocal (Opt-In use_mp_senet=True, §v10.994)",
        notes="Vokal-Denoising. n_fft=400 (201 Bins). Norm-Kalibrierung §v10.994: 99-Perzentil-Peak-Norm + Gain-Kompensation + Loudness-Guard.",
    ),
    ModelEntry(
        name="melbandroformer",
        path="models/melbandroformer/melbandroformer_optimized.onnx",
        purpose="repair",
        input_shapes="IN input [1, duration, 60, 384] → OUT 5-dim",
        status="active",
        notes="BS-Roformer (bs_roformer_plugin). Stem-aware Repair-Flow integriert (Rev. 2026-08-17): SLR-1-Pre-Phase (UV3) + Phase 66 Stem-Targeted-NR + StemRemixBalancer (Spec §1.4/§2.8, Quell-LUFS-korrekter Re-Mix mit Soft-Knee-Peak-Schutz).",
    ),
    ModelEntry(
        name="mdx23c",
        path="models/mdx23c/models/Kim_Vocal_2.onnx",
        purpose="separation",
        input_shapes="IN input [B,4,3072,256] → OUT [B,4,3072,256]",
        status="active",
        notes="Stem-Trennung (Vocals/drums/bass/other). In SotaVocalModelRouter als Separation-Kandidat verdrahtet (demucs-Primär, mdx23c-Fallback); StemRemixBalancer übernimmt den LUFS-korrekten Re-Mix.",
    ),
    ModelEntry(
        name="sgmse_plus",
        path="models/ (kein ONNX — .pth in Plugin)",
        purpose="repair",
        input_shapes="via plugins/sgmse_plugin.py",
        status="active",
        integration="coordinated_repair._run_denoise (Opt-In use_sgmse, kontextaktiviert bei vocal_confidence>0.5, §v10.994)",
        notes="Sprach-Enhancement-Diffusion (SGMSE+). Immer mit DSP-Fallback; nie stiller Ausfall.",
    ),
    ModelEntry(
        name="audioldm2",
        path="models/audioldm2/audioldm2.onnx",
        purpose="generation",
        input_shapes="Text-zu-Audio",
        status="generation_only",
        notes="Text-to-Audio-Generierung — KEIN Repair-Werkzeug. Korrekt NICHT in der Repair-Kette.",
    ),
    ModelEntry(
        name="cqtdiff",
        path="models/cqtdiff/score_network.pt",
        purpose="generation",
        input_shapes="Audio-Synthese",
        status="generation_only",
        notes="Diffusions-Synthese — KEIN Repair-Werkzeug.",
    ),
    ModelEntry(
        name="diffwave",
        path="models/diffwave/diffwave_model.onnx",
        purpose="vocoder",
        input_shapes="Mel → Waveform",
        status="generation_only",
        notes="Vocoder für Synthese; HiFi-GAN ist der aktiv genutzte Vocoder für Inpainting-Roadmap.",
    ),
]


def probe_models() -> dict[str, str]:
    """Prüft Ladebarkeit aller Modelle ohne Inferenz.

    Returns:
        {name: "ok" | "missing" | "load_error:<msg>"}
    """
    import onnxruntime as ort

    results: dict[str, str] = {}
    for entry in MODEL_ZOO:
        path = _PROJECT / entry.path
        if not path.exists() or path.suffix != ".onnx":
            results[entry.name] = "missing" if not path.exists() else "non_onnx"
            continue
        try:
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            results[entry.name] = "ok"
        except Exception as exc:
            results[entry.name] = f"load_error:{str(exc)[:60]}"
    return results


def get_model(name: str) -> ModelEntry | None:
    for entry in MODEL_ZOO:
        if entry.name == name:
            return entry
    return None


def report() -> str:
    """Text-Report für Logs/Startup."""
    lines = [f"Model Zoo: {len(MODEL_ZOO)} Modelle registriert"]
    by_status: dict[str, int] = {}
    for entry in MODEL_ZOO:
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
    lines.append("  Status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return "\n".join(lines)
