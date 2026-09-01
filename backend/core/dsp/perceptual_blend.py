"""
======================================================================

Ersetzt den skalaren Wet/Dry-Blend durch ein frequenzabhängiges,
psychoakustisch maskiertes Blend-Modell.

Prinzip: Nur hörbare Änderungen werden übernommen.
- In Frequenzbändern, wo die Änderung UNTER der Maskierungsschwelle liegt
  → Dry-Signal (Änderung ist unhörbar → kein Risiko von Artefakten)
- In Frequenzbändern, wo die Änderung ÜBER der Maskierungsschwelle liegt
  → Wet-Signal (hörbare Verbesserung)

Referenzen:
    ISO 11172-3: Simultane Maskierung (Bark-Skala)
    Zwicker & Fastl (1999): Psychoacoustics — Facts and Models
    Johnston (1988): Transform Coding of Audio Signals
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bark-Skala: 24 kritische Bänder des menschlichen Gehörs
# Grenzfrequenzen in Hz (Zwicker 1961)
# ---------------------------------------------------------------------------
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
    dtype=np.float32,
)


def _hz_to_bark(hz: np.ndarray) -> np.ndarray:
    """Konvertiert Hz → Bark (Zwicker-Formel)."""
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)  # type: ignore[no-any-return]


def perceptual_blend(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    scalar_wet: float = 1.0,
    *,
    min_gain: float = 0.05,
    silence_thresh_db: float = -60.0,
) -> np.ndarray:
    """Blendet wet und dry perzeptuell — nur hörbare Änderungen übernehmen.

    Args:
        dry:         Original-Audio (mono oder stereo) float32
        wet:         Verarbeitetes Audio, gleiche Shape wie dry
        sr:          Sample-Rate (48000)
        scalar_wet:  Globaler Wet-Skalar [0, 1] — skaliert die Stärke
        min_gain:    Minimaler Gain pro Band (Verhinderung von Totalauslöschung)
        silence_thresh_db: Schwellwert für Stille-Detektion

    Returns:
        Perzeptuell geblendetes Audio, gleiche Shape wie dry
    """
    if dry.shape != wet.shape:
        raise ValueError(f"Shape mismatch: dry {dry.shape} vs wet {wet.shape}")

    if scalar_wet <= 0.0:
        return dry.copy()
    if scalar_wet >= 1.0:
        return wet.copy()

    _N_FFT = 2048
    _HOP = 512

    is_stereo = dry.ndim == 2
    n_samples = dry.shape[0] if not is_stereo else dry.shape[1]

    # ── Mono-STFT für Maskierungsberechnung ──
    dry_mono = dry if not is_stereo else np.mean(dry, axis=0)
    wet_mono = wet if not is_stereo else np.mean(wet, axis=0)

    delta = wet_mono - dry_mono
    delta_rms = float(np.sqrt(np.mean(delta**2)) + 1e-12)
    if delta_rms < 1e-8:
        return dry.copy()  # Keine Änderung → dry

    # ── STFT ──
    import librosa

    D_dry = librosa.stft(dry_mono.astype(np.float32), n_fft=_N_FFT, hop_length=_HOP)
    D_wet = librosa.stft(wet_mono.astype(np.float32), n_fft=_N_FFT, hop_length=_HOP)

    mag_dry = np.abs(D_dry)  # (n_freq, n_frames)
    mag_delta = np.abs(D_wet - D_dry)  # Magnitude difference (perceptual blending metric)

    n_freq, n_frames = mag_dry.shape
    freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr)

    # ── Bark-Band-Mapping ──
    bark_per_freq = _hz_to_bark(freqs.astype(np.float32))
    bark_edges = _hz_to_bark(_BARK_EDGES_HZ)
    n_bark = len(bark_edges) - 1

    # ── Pro-Bark-Band Maskierungsschwelle ──
    # Spread-Funktion: Energie in Band i maskiert benachbarte Bänder
    spread_matrix = np.zeros((n_bark, n_bark), dtype=np.float32)
    for i in range(n_bark):
        for j in range(n_bark):
            dz = abs(i - j)
            if dz <= 1:
                spread_matrix[i, j] = 1.0
            elif dz <= 3:
                spread_matrix[i, j] = 10.0 ** (-(dz - 1) / 4.0)
            elif dz <= 8:
                spread_matrix[i, j] = 10.0 ** (-0.5 - (dz - 3) / 2.0)
            else:
                spread_matrix[i, j] = 0.001

    # ── Frame-weise Maskierungsberechnung ──
    gain_mask = np.ones(n_frames, dtype=np.float32) * scalar_wet

    # §v10.101 Temporale Maskierung (ISO 11172-3):
    # Laute Transienten maskieren leisere Ereignisse VOR (20ms) und NACH (100ms) sich.
    # Wir berechnen die Onset-Stärke via Spectral Flux und erweitern die Maskierung.
    _PRE_MASK_FRAMES = max(1, int(0.020 * sr / _HOP))  # 20ms pre-masking
    _POST_MASK_FRAMES = max(1, int(0.100 * sr / _HOP))  # 100ms post-masking
    _ONSET_THRESHOLD = 0.15  # Relative Schwelle für Spectral-Flux-Onsets

    # Spectral Flux: Summe der positiven Magnituden-Differenzen pro Frame
    if n_frames > 1:
        _flux = np.sum(np.maximum(0.0, np.diff(mag_dry, axis=1)), axis=0)  # (n_frames-1,)
        _flux = np.concatenate([[0.0], _flux])
        _flux_norm = _flux / (np.mean(_flux) + 1e-12)
        _onset_frames = np.where(_flux_norm > _ONSET_THRESHOLD)[0]

        # Temporal mask: raise gain around onsets (artifacts are masked by transients)
        _temporal_boost = np.ones(n_frames, dtype=np.float32)
        for _onset in _onset_frames:
            _pre_start = max(0, _onset - _PRE_MASK_FRAMES)
            _post_end = min(n_frames, _onset + _POST_MASK_FRAMES + 1)
            # Pre-masking: linear ramp up to onset (weaker masking before onset)
            for _t in range(_pre_start, _onset):
                _frac = (_t - _pre_start) / max(_onset - _pre_start, 1)
                _boost = 1.0 + 0.3 * _frac  # Up to +30% gain (more wet in masked zone)
                _temporal_boost[_t] = max(_temporal_boost[_t], _boost)
            # Post-masking: exponential decay after onset
            for _t in range(_onset, _post_end):
                _decay = np.exp(-(_t - _onset) / max(_POST_MASK_FRAMES * 0.3, 1))
                _boost = 1.0 + 0.4 * _decay  # Up to +40% gain, decaying
                _temporal_boost[_t] = max(_temporal_boost[_t], _boost)
    else:
        _temporal_boost = np.ones(n_frames, dtype=np.float32)

    # Smooth temporal boost to prevent abrupt changes
    from scipy.ndimage import uniform_filter1d

    _temporal_boost = uniform_filter1d(_temporal_boost.astype(np.float64), size=3).astype(np.float32)

    for frame in range(n_frames):
        # Energie pro Bark-Band im Dry-Signal
        bark_energy_dry = np.zeros(n_bark, dtype=np.float32)
        for fb in range(n_bark):
            mask = (bark_per_freq >= bark_edges[fb]) & (bark_per_freq < bark_edges[fb + 1])
            if mask.any():
                bark_energy_dry[fb] = np.mean(mag_dry[mask, frame] ** 2)

        # Spread-Funktion: Maskierung durch benachbarte Bänder
        masked_threshold = spread_matrix @ bark_energy_dry

        # Energie pro Bark-Band im Delta-Signal
        bark_energy_delta = np.zeros(n_bark, dtype=np.float32)
        for fb in range(n_bark):
            mask = (bark_per_freq >= bark_edges[fb]) & (bark_per_freq < bark_edges[fb + 1])
            if mask.any():
                bark_energy_delta[fb] = np.mean(mag_delta[mask, frame] ** 2)

        # Tonality-Adjustment: sinusoidale Komponenten maskieren weniger
        # → vereinfachter SFM (Spectral Flatness Measure)
        sfm = np.zeros(n_bark, dtype=np.float32)
        for fb in range(n_bark):
            mask = (bark_per_freq >= bark_edges[fb]) & (bark_per_freq < bark_edges[fb + 1])
            if mask.any():
                band_mag = mag_dry[mask, frame]
                geo = np.exp(np.mean(np.log(band_mag + 1e-12)))
                ari = np.mean(band_mag)
                sfm[fb] = min(geo / (ari + 1e-12), 1.0)

        # Tonalitäts-Offset: tonal → höhere Maskierungsschwelle
        tonality_offset = -2.0 * (1.0 - sfm)  # -2 dB für tonal, 0 für rauschig
        masked_threshold_db = 10.0 * np.log10(masked_threshold + 1e-12) + tonality_offset

        # Absolute Hörschwelle (ISO 226)
        bark_centers = (bark_edges[:-1] + bark_edges[1:]) / 2
        abs_thresh = np.zeros(n_bark, dtype=np.float32)
        for fb in range(n_bark):
            bark_z = bark_centers[fb]
            abs_thresh[fb] = 3.64 * (bark_z**-0.8) - 6.5 * np.exp(-0.6 * (bark_z - 3.3) ** 2) + 1e-3 * (bark_z**4)

        abs_thresh_db = 10.0 * np.log10(np.maximum(abs_thresh, 1e-12))
        effective_thresh_db = np.maximum(masked_threshold_db, abs_thresh_db)

        # Delta-Energie in dB
        delta_energy_db = 10.0 * np.log10(bark_energy_delta + 1e-12)

        # Vergleich: Ist die Änderung pro Band hörbar?
        audible_mask = delta_energy_db > effective_thresh_db
        n_audible = int(np.sum(audible_mask))

        if n_audible > 0:
            # Gewichtung: je mehr Bänder hörbar geändert, desto mehr Wet
            audibility_ratio = float(n_audible) / float(n_bark)
            # Skaliere den globalen Wet-Faktor mit der Hörbarkeit
            gain_mask[frame] = float(
                np.clip(
                    scalar_wet * (0.3 + 0.7 * audibility_ratio),
                    min_gain,
                    1.0,
                )
            )
        else:
            # Keine hörbare Änderung → fast nur Dry
            gain_mask[frame] = min_gain

        # §v10.101 Temporale Maskierung: Bei Transienten mehr Wet erlauben
        gain_mask[frame] = float(
            np.clip(
                gain_mask[frame] * _temporal_boost[frame],
                min_gain,
                1.0,
            )
        )

    # ── Smooth Gain über Frames ──
    from scipy.ndimage import uniform_filter1d

    gain_mask = uniform_filter1d(gain_mask.astype(np.float64), size=5).astype(np.float32)

    # ── Blend ──
    blended = dry + gain_mask[np.newaxis, :] * (wet - dry) if is_stereo else None
    if not is_stereo:
        # Upsample frame-gain to sample-level
        gain_samples = np.interp(
            np.arange(n_samples),
            np.arange(n_frames) * _HOP,
            gain_mask,
        ).astype(np.float32)
        blended = dry + gain_samples * (wet - dry)

    return np.clip(blended.astype(np.float32), -1.0, 1.0)  # type: ignore[union-attr, no-any-return]
