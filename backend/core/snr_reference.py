#!/usr/bin/env python3
"""backend/core/snr_reference.py — Kanonische SNR-Referenz-Definition.

SNR-Kanonisierung (Projekt 2026-08-22, testbegleitet): Die vier im Lauf
beobachteten SNR-Werte messen verschiedene physikalische Größen —
  Prognose 14.4 dB  (Pre-Analysis-Quell-Einschätzung),
  BIR 8.6 dB        (BlindInternalReference-Qualitätsproxy),
  GGB 26.9 dB       (Gain-Budget-Eingang),
  QualityAnalyzer 38.9 dB (Ergebnis-SNR des MQA-Gates).
Deshalb gilt ab jetzt die Benennungskonvention:
  source_snr_db   → SNR der QUELLE (degradierter Import), kanonisch aus
                    CalibrationContext.snr_db (Pre-Analysis).
  output_snr_db   → SNR des ERGEBNISSES, gemessen mit DIESER Referenz-
                    Definition (estimate_snr_db) — für Gates, die das
                    Ergebnis bewerten (baseline-relativ zum gleichen
                    Schätzer auf der Quelle).
  bir_snr_proxy   → BIR-Qualitätsproxy — explizit ADVISORY, nie Gate-Grundlage.
Gates vergleichen IMMER denselben Schätzer auf Quelle UND Ergebnis
(baseline-relativ, §0) — niemals source_snr gegen output_snr.

Referenz-Definition (aurik_snr_v1):
  Welch-PSD-Perzentil-SNR — robust auch für stationäre Signale (Töne):
    1. Mono-Mix layout-sicher (audio_layout.mono_mix)
    2. rfft(1024, Hann), Mittelung der PSD über alle Frames
    3. signal² = P90 der PSD-Bins, noise² = P10 der PSD-Bins
       (Ton-Energie konzentriert sich in wenige Bins → P90;
       Breitbandrauschen verteilt sich → P10)
    4. SNR_dB = 10·log10(P90 / P10)
  - Guards: Stille/konstantes Signal → 120 dB
  - Deterministisch (§G5 (GEBOTE.md)), layout-invariant, monoton im injizierten
    Rauschpegel (Test-gesichert).
"""

from __future__ import annotations

import numpy as np

from backend.core.audio_layout import mono_mix

SNR_DEFINITION_VERSION: str = "aurik_snr_v1"
FRAME_SIZE: int = 1024
SILENCE_SNR_DB: float = 120.0

# Benennungskonvention (Kanonische Schlüssel)
SOURCE_SNR_KEY: str = "source_snr_db"
OUTPUT_SNR_KEY: str = "output_snr_db"
BIR_SNR_PROXY_KEY: str = "bir_snr_proxy"


def estimate_snr_db(audio: np.ndarray, sr: int = 48000) -> float:
    """Referenz-SNR-Schätzer (aurik_snr_v1) — siehe Modul-Docstring.

    Args:
        audio: Mono oder Stereo ((N, C) und (C, N) — layout-invariant).
        sr:    Abtastrate (nur für die Dokumentation; Fenster ist fest 1024).

    Returns:
        float SNR in dB; 120.0 dB bei Stille/konstantem Signal.
    """
    mono = mono_mix(np.asarray(audio, dtype=np.float32))
    if mono.size < FRAME_SIZE:
        return SILENCE_SNR_DB
    n_frames = mono.size // FRAME_SIZE
    if n_frames < 3:
        return SILENCE_SNR_DB
    trimmed = mono[: n_frames * FRAME_SIZE].reshape(n_frames, FRAME_SIZE).astype(np.float64)
    window = np.hanning(FRAME_SIZE)
    psd = np.mean(np.abs(np.fft.rfft(trimmed * window, axis=1)) ** 2, axis=0)
    psd = np.maximum(psd, 1e-30)
    p10 = float(np.percentile(psd, 10))
    p90 = float(np.percentile(psd, 90))
    if p90 < 1e-25 or p10 <= 1e-30:
        return SILENCE_SNR_DB
    snr_db = float(10.0 * np.log10(p90 / p10))
    return float(np.clip(snr_db, 0.0, SILENCE_SNR_DB))


def format_snr_label(key: str, value_db: float) -> str:
    """Kanonisch beschriftete SNR-Zeile: '<key>=<wert> dB'."""
    return f"{key}={float(value_db):.1f} dB"


__all__ = [
    "SNR_DEFINITION_VERSION",
    "FRAME_SIZE",
    "SILENCE_SNR_DB",
    "SOURCE_SNR_KEY",
    "OUTPUT_SNR_KEY",
    "BIR_SNR_PROXY_KEY",
    "estimate_snr_db",
    "format_snr_label",
]
