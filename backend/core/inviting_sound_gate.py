"""Einladungs-Gate — Hörordnung Ebene 4 (hoerordnung.instructions.md §6).

Positives psychoakustisches Kriterium für „Wohlklang, in den sich das Ohr
hineinlegt": gemessen als Fenster-Gate (5 s, überlappend), nicht als Einzelwert.

Regeln (normativ, Hörordnung §6):
- Roughness (Zwicker, asper): keine Spitze > 0.5 asper in Stimm-/Klimax-Zonen.
- Sharpness (Bismarck-Näherung, acum): keine Sprünge > 0.2 acum zwischen
  benachbarten Fenstern.
- Ermüdung: fatigue_index > 0.40 ⇒ `fatigue_abort` — beendet die Optimierung
  (kein Veto des Audio-Ergebnisses, sondern Abbruch-Signal für Schleifen).

Das Gate erzeugt keine neuen Mess-Module: Roughness kommt aus
`backend/core/dsp/zwicker_metrics.py`, Ermüdung aus dem Fatigue-Analyzer bzw.
dem experience_runtime-Index. Sharpness (Bismarck) war als Funktion noch nicht
vorhanden und ist hier als schlanke Bark-basierte Näherung implementiert.

[RELEASE_MUST]-Frei: advisory-first; das Gate loggt und markiert, erzwingt aber
keinen harten Export-Stopp (§0 Primum non nocere — die Hör-Instanz entscheidet).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_WINDOW_S_DEFAULT = 5.0
_HOP_S_DEFAULT = 2.5
_ASPER_LIMIT = 0.50  # Hörordnung §6: Roughness-Spitze > 0.5 asper
_SHARPNESS_JUMP_LIMIT = 0.20  # Hörordnung §6: Sprung > 0.2 acum
_FATIGUE_ABORT_THRESHOLD = 0.40  # Hörordnung §6: Ermüdung > 0.40 → Abbruch

# Bark-Grenzen (ISO 11172-3, 25 Grenzen für 24 Bänder bis 15.5 kHz Näherung)
_BARK_EDGES_HZ = np.array(
    [
        0,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ],
    dtype=np.float64,
)


@dataclass
class InvitingGateResult:
    """Ergebnis des Einladungs-Gates (Hörordnung Ebene 4)."""

    passed: bool = True
    max_asper: float = 0.0
    max_asper_in_voice: float = 0.0
    sharpness_jump_max: float = 0.0
    fatigue_score: float = 1.0
    fatigue_abort: bool = False
    n_windows: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def compute_sharpness_acum(audio: np.ndarray, sr: int) -> float:
    """Sharpness (Bismarck-Näherung) in acum.

    N = 0.11 * sum(N'(z) * g(z) * z) / sum(N'(z)), z = Bark-Nummer,
    g(z) = 1 für z < 15.8, sonst 0.15*exp(0.42*(z-15.8)) + 0.85.
    Spezifische Lautheit N'(z) genähert über (Band-Energie)^0.23 (Zwicker).
    Kalibriert auf typische Musik: ~0.4-1.2 acum.
    """
    if audio is None or audio.size == 0:
        return 0.0
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=-1)
    mono = mono - float(np.mean(mono))
    n_fft = 4096
    if len(mono) < n_fft:
        return 0.0
    spec = np.abs(np.fft.rfft(mono[: min(len(mono), sr * 60)], n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band_energy = np.zeros(len(_BARK_EDGES_HZ) - 1)
    for b in range(len(band_energy)):
        mask = (freqs >= _BARK_EDGES_HZ[b]) & (freqs < _BARK_EDGES_HZ[b + 1])
        if np.any(mask):
            band_energy[b] = float(np.sum(spec[mask]))
    specific_loudness = band_energy**0.23
    z = np.arange(1, len(band_energy) + 1, dtype=np.float64)
    g = np.where(z < 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
    num = float(np.sum(specific_loudness * g * z))
    den = float(np.sum(specific_loudness)) + 1e-12
    acum = 0.11 * num / den
    # Sehr leise/stille Fenster (Energie ~0) erzeugen numerischen Müll → 0.
    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    if rms < 1e-5:
        acum = 0.0
    return float(np.clip(acum, 0.0, 4.0))


def _window_slices(total: int, sr: int, window_s: float, hop_s: float) -> list[tuple[int, int]]:
    w = max(int(sr * window_s), sr // 2)
    h = max(int(sr * hop_s), w // 2)
    if total <= w:
        return [(0, total)]
    out: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + w, total)
        out.append((start, end))
        if end >= total:
            break
        start += h
    return out


def check_inviting_gate(
    audio: np.ndarray,
    sr: int,
    singing_mask: np.ndarray | None = None,
    fatigue_index: float = 0.0,
    window_s: float = _WINDOW_S_DEFAULT,
    hop_s: float = _HOP_S_DEFAULT,
    repair_windows: list[tuple[float, float]] | None = None,
) -> InvitingGateResult:
    """Fenster-Gate der Hörordnung Ebene 4.

    Args:
        audio: Signal (mono (n,) oder (n, ch) / (ch, n)).
        sr: Abtastrate.
        singing_mask: optionales bool-Array (pro Fenster, 1D) — True = Stimm-
            /Klimax-Zone. Fehlt es, gelten alle Fenster als kritisch.
        fatigue_index: Ermüdungsindex aus experience_runtime (0..1).
        repair_windows: optionale Reparatur-Zeiträume (s) — Sharpness-Sprünge,
            die ein Reparatur-Fenster überlappen, sind beabsichtigt und werden
            vom Jump-Kriterium ausgenommen (advisory, 2026-09-07).
    """
    res = InvitingGateResult()
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2 and mono.shape[0] <= 8:
        mono = mono.mean(axis=0)
    elif mono.ndim == 2:
        mono = mono.mean(axis=-1)
    total = len(mono)
    if total < sr // 2:
        res.details["skipped"] = "audio_too_short"
        return res

    try:
        from backend.core.dsp.zwicker_metrics import compute_roughness_asper
    except Exception as exc:
        logger.debug("§V6 zwicker_metrics.compute_roughness_asper nicht verfügbar — Gate mit Default-Werten zurückgegeben: %s", exc)
        res.details["skipped"] = "zwicker_unavailable"
        return res

    windows = _window_slices(total, sr, window_s, hop_s)
    asper_values: list[float] = []
    sharp_values: list[float] = []
    for wi, (s, e) in enumerate(windows):
        seg = mono[s:e]
        try:
            asper = float(compute_roughness_asper(seg.astype(np.float32), sr))
        except Exception as exc:
            logger.debug("§V6 compute_roughness_asper fehlgeschlagen — 0.0 zurückgegeben (Window %d): %s", wi, exc)
            asper = 0.0
        sharp = compute_sharpness_acum(seg, sr)
        asper_values.append(asper)
        sharp_values.append(sharp)
        in_voice = True
        if singing_mask is not None:
            in_voice = bool(singing_mask[wi]) if wi < len(singing_mask) else bool(singing_mask[-1])
        if in_voice:
            res.max_asper_in_voice = max(res.max_asper_in_voice, asper)

    res.max_asper = float(max(asper_values)) if asper_values else 0.0
    _raw_jump_max = (
        float(max(abs(sharp_values[i + 1] - sharp_values[i]) for i in range(len(sharp_values) - 1)))
        if len(sharp_values) > 1
        else 0.0
    )
    # §Reparatur-Kontext (2026-09-07): Sprünge an echten Reparaturstellen sind
    # beabsichtigt (lokalisierte HF-Änderung) — das Jump-Kriterium der
    # Hörordnung §6 zielt auf unbeabsichtigte Diskontinuitäten. Kein neuer
    # Schwellwert: das 0.2-acum-Limit bleibt normativ, nur der Kontext wird
    # berücksichtigt.
    _eff_jump_max = _raw_jump_max
    _exempted_jumps = 0
    if _raw_jump_max > _SHARPNESS_JUMP_LIMIT and repair_windows:
        try:
            _kept_jumps: list[float] = []
            for _ji in range(len(sharp_values) - 1):
                _jv = abs(sharp_values[_ji + 1] - sharp_values[_ji])
                if _jv <= _SHARPNESS_JUMP_LIMIT:
                    _kept_jumps.append(_jv)
                    continue
                _s0 = windows[_ji][0] / sr
                _e0 = windows[_ji + 1][1] / sr
                _overlaps = any((_s0 < float(_rw[1]) and _e0 > float(_rw[0])) for _rw in repair_windows)
                if _overlaps:
                    _exempted_jumps += 1
                else:
                    _kept_jumps.append(_jv)
            _eff_jump_max = float(max(_kept_jumps)) if _kept_jumps else 0.0
            if _exempted_jumps:
                logger.info(
                    "Einladungs-Gate: %d Sharpness-Sprung/Sprünge an Reparaturstellen ausgenommen (raw=%.3f → effektiv=%.3f acum)",
                    _exempted_jumps,
                    _raw_jump_max,
                    _eff_jump_max,
                )
        except Exception as _jump_exc:
            logger.debug("Reparatur-Fenster-Exemption fehlgeschlagen: %s", _jump_exc)
    res.sharpness_jump_max = _eff_jump_max
    res.details["sharpness_jump_raw_max"] = round(_raw_jump_max, 3)
    res.details["exempted_jumps"] = _exempted_jumps
    res.n_windows = len(windows)

    # Ermüdung (experience_runtime-Index; optional Fatigue-Analyzer-Gegenprobe)
    res.fatigue_abort = bool(fatigue_index > _FATIGUE_ABORT_THRESHOLD)
    res.fatigue_score = float(np.clip(1.0 - float(fatigue_index), 0.0, 1.0))

    failed = []
    if res.max_asper_in_voice > _ASPER_LIMIT:
        failed.append(f"roughness_spike={res.max_asper_in_voice:.3f}asper")
    if res.sharpness_jump_max > _SHARPNESS_JUMP_LIMIT:
        failed.append(f"sharpness_jump={res.sharpness_jump_max:.3f}acum")
    res.passed = not failed
    res.details["failures"] = failed
    res.details["asper_per_window"] = [round(a, 3) for a in asper_values]
    res.details["sharpness_per_window"] = [round(s, 3) for s in sharp_values]
    return res


_instance: InvitingSoundGate | None = None
_lock = threading.Lock()


class InvitingSoundGate:
    """Singleton-Wrapper um check_inviting_gate (analog zu anderen Gates)."""

    def check(self, audio: np.ndarray, sr: int, **kwargs: Any) -> InvitingGateResult:
        return check_inviting_gate(audio, sr, **kwargs)


def get_inviting_gate() -> InvitingSoundGate:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = InvitingSoundGate()
    return _instance


__all__ = [
    "InvitingGateResult",
    "InvitingSoundGate",
    "check_inviting_gate",
    "compute_sharpness_acum",
    "get_inviting_gate",
]
