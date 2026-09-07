"""Gemeinsames psychoakustisches Front-End (Muster 1+4 der Qualitäts-/Effizienz-Roadmap).

Problem vorher: ~25 Module machten eigene STFTs; ``einladungs_gate`` berechnete
Roughness/Sharpness/Loudness in drei getrennten Fenster-Schleifen mit je eigener
Spektralarbeit — teuer UND inkonsistent (Gates urteilten auf verschiedenen
Repräsentationen, kurze Fenster → schlechte Frequenzauflösung).

Lösung: EIN Frame pro Signal, danach nur noch Ableitungen:

- feines STFT (n_fft=2048, hop=512) für Bark-Energie/Sharpness/Maskierung
- Fluss-Hüllkurve (n_fft=512, hop=128 → Envelope-Rate 375 Hz) für Roughness
  (40–200-Hz-Modulationsband, Nyquist 187.5 Hz deckt das Band voll ab)
- Rectangular-Bark-Masken aus BARK_EDGES_HZ (Zwicker 1961)
- Maskierungsschwelle je Bark-Band (ISO 11172-3 vereinfacht: Band-Energie − 12 dB)
- vektorisierte Ganzsignal-Metriken mit Fenster-Aggregation (Muster 4):
  roughness_zwicker (Zwicker & Fastl 1990), sharpness_bismarck (Bismarck 2001),
  loudness_erb (Sone-Skala-Proxy) — Fenster werden auf der Repräsentation
  geschnitten, nicht auf dem Audio.
- Audibility-Routinen für Early-Termination (Muster 2): ``band_energy_dbfs`` +
  ``is_below_masking`` → Phasen können stoppen, sobald der Defekt unter der
  Maskierungsschwelle liegt (Hörordnung §4).

Synergien: einladungs_gate (Konsistenz), phase_02/phase_05 (Early-Termination),
Wohlklang-Blend (billige Re-Evaluierung) lesen dieselbe Repräsentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from backend.core.audio_utils import safe_stft
from backend.core.dsp.bark_lufs_util import BARK_EDGES_HZ

# Bandgrenzen des Roughness-Modulationsbandes (Zwicker & Fastl 1990)
_ROUGHNESS_BAND_HZ = (40.0, 200.0)
# G-Gewichtung der Sharpness (Bismarck 2001): Energieanteil > 7 kHz
_SHARPNESS_EDGE_HZ = 7000.0
# Maskierungs-Spreizung: Signal − 12 dB (ISO 11172-3, vereinfacht)
_MASKING_SPREAD_DB = 12.0
# Standard-Sicherheitsabstand für Audibility-Entscheidungen
_DEFAULT_AUDIBILITY_MARGIN_DB = 6.0
# Hüllkurven-Glättung für Roughness: ~133 ms MA bei Envelope-Rate 375 Hz
_ENV_SMOOTH_SAMPLES = 50
# Absolute Hörschwelle als Maskierungs-Floor (≈ 0 dB SPL Referenz, Ruhe)
_ABSOLUTE_MASKING_FLOOR_DB = -95.0


@dataclass
class PsychoacousticFrame:
    """Einmal berechnete psychoakustische Repräsentation eines Signals."""

    sr: int
    # Feines STFT: |S| [n_freq, n_frames_fine]
    stft_mag: np.ndarray
    freqs: np.ndarray
    stft_times: np.ndarray
    stft_hop: int
    # Fluss-Hüllkurve für Modulations-Analyse (hop 128)
    onset_env: np.ndarray
    env_rate: float
    # Bark: Masken [n_bark, n_freq] + Energie je Band [n_bark, n_frames_fine]
    bark_masks: np.ndarray
    bark_energy_db: np.ndarray
    # Maskierungsschwelle je Bark-Band [n_bark]
    masking_threshold_db: np.ndarray
    # Mono-Zeitsignal für RMS-basierte Loudness (Referenzsemantik)
    mono: np.ndarray = field(repr=False)

    @property
    def n_bark(self) -> int:
        return int(self.bark_masks.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.stft_mag.shape[1])

    # ── Fenster-Aggregation (Muster 4: Fenster auf der Repräsentation) ──────

    def roughness_windows(self, window_s: float, hop_s: float) -> list[float]:
        """Roughness je Fenster: Modulationsband-Energie 40–200 Hz / Gesamtenergie.

        Fenster werden auf der Hüllkurve geschnitten — eine rfft pro Fenster,
        keine erneute STFT auf Audio-Chunks. Hüllkurven-Glättung (~133 ms MA)
        + Hann-Fenster entfernen Signal-Attack-Transienten aus dem Band
        (sauberer 440-Hz-Sinus: max 0.08 statt 0.75; 70-Hz-AM: 0.98).
        """
        values: list[float] = []
        env = uniform_filter1d(self.onset_env, size=_ENV_SMOOTH_SAMPLES)
        hop_env = max(1, int(round(hop_s * self.env_rate)))
        win_env = max(1, int(round(window_s * self.env_rate)))
        lo, hi = _ROUGHNESS_BAND_HZ
        hann = np.hanning(win_env)
        for i in range(0, len(env), hop_env):
            seg = env[i : i + win_env]
            if len(seg) < max(64, win_env // 2):  # gleiche Abbruch-Semantik wie alte Audio-Chunk-Schleife
                break
            spec = np.abs(np.fft.rfft((seg - np.mean(seg)) * hann[: len(seg)]))
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / self.env_rate)
            band = (freqs >= lo) & (freqs <= hi)
            band_energy = float(np.sum(spec[band] ** 2))
            total = float(np.sum(spec**2)) + 1e-12
            values.append(float(np.clip(np.sqrt(band_energy / total), 0.0, 1.0)))
        return values

    def sharpness_windows(self, window_s: float, hop_s: float) -> list[float]:
        """Sharpness je Fenster: G-gewichteter Energieanteil > 7 kHz × 5 (acum)."""
        values: list[float] = []
        hop_frames = max(1, int(round(hop_s * self.sr / self.stft_hop)))
        win_frames = max(1, int(round(window_s * self.sr / self.stft_hop)))
        g = (self.freqs > _SHARPNESS_EDGE_HZ).astype(np.float64)
        s2 = self.stft_mag**2
        for i in range(0, self.n_frames, hop_frames):
            block = s2[:, i : i + win_frames]
            if block.shape[1] < max(2, win_frames // 2):
                break
            weighted = float(np.sum(block * g[:, None]))
            total = float(np.sum(block)) + 1e-12
            values.append(float(np.clip(weighted / total, 0.0, 1.0)) * 5.0)
        return values

    def loudness_windows(self, window_s: float, hop_s: float) -> list[float]:
        """Loudness je Fenster (ERB/Sone-Proxy, identische Semantik wie bisher)."""
        values: list[float] = []
        hop_smp = max(1, int(round(hop_s * self.sr)))
        win_smp = max(1, int(round(window_s * self.sr)))
        for i in range(0, len(self.mono), hop_smp):
            chunk = self.mono[i : i + win_smp]
            if len(chunk) < max(2, win_smp // 2):
                break
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            loudness_db = 20.0 * np.log10(rms + 1e-12)
            sones = 2.0 ** ((loudness_db + 40.0) / 10.0)
            values.append(float(np.clip(sones / 100.0, 0.0, 1.0)))
        return values

    # ── Audibility-Routinen (Muster 2) ──────────────────────────────────────

    def band_energy_dbfs(self, lo_hz: float, hi_hz: float) -> float:
        """Mittlere Energie (dBFS) aller Bark-Bänder, deren Mitten in [lo, hi] liegen."""
        centers = _bark_centers_hz()
        idx = np.where((centers >= lo_hz) & (centers <= hi_hz))[0]
        if len(idx) == 0:
            idx = np.array([0])
        return float(np.mean(self.bark_energy_db[idx]))

    def line_energy_dbfs(self, freqs_hz: list[float], width_hz: float = 2.0) -> float:
        """Schmalband-Energie (dBFS) um diskrete Linien (z. B. Hum-Harmonische).

        Summiert die mittlere STFT-Power in ±width_hz-Fenstern um jede Linie —
        eine Hum-only-Schätzung aus derselben Repräsentation wie die Maskierung.
        """
        if not freqs_hz:
            return -120.0
        mask = np.zeros_like(self.freqs, dtype=bool)
        # Bin-breiten-bewusst: ±width_hz ist bei n_fft=2048 (23.4-Hz-Bins) zu
        # schmal — mindestens 60 % der Bin-Breite, sonst fängt das Fenster keine
        # Linien.
        bin_hz = float(self.freqs[1]) if len(self.freqs) > 1 else 1.0
        eff_width = max(width_hz, bin_hz * 0.6)
        for f0 in freqs_hz:
            mask |= np.abs(self.freqs - f0) <= eff_width
        if not np.any(mask):
            return -120.0
        power = float(np.mean(self.stft_mag[mask, :] ** 2))
        return float(10.0 * np.log10(power + 1e-12))

    def is_below_masking(
        self, energy_dbfs: float, lo_hz: float, hi_hz: float, margin_db: float = _DEFAULT_AUDIBILITY_MARGIN_DB
    ) -> bool:
        """True, wenn *energy_dbfs* unter der Maskierungsschwelle des Bandes + Marge liegt.

        Hörordnung §4: Reparatur gilt als abgeschlossen, wenn der Defekt unter der
        Maskierungsschwelle liegt — hier als Laufzeit-Entscheidung.
        """
        centers = _bark_centers_hz()
        idx = np.where((centers >= lo_hz) & (centers <= hi_hz))[0]
        if len(idx) == 0:
            idx = np.array([0])
        # Maskierung kann nicht unter die absolute Hörschwelle fallen — sonst
        # würde eine isolierte Linie sich selbst „maskieren“ (self-referenz).
        threshold = max(float(np.mean(self.masking_threshold_db[idx])), _ABSOLUTE_MASKING_FLOOR_DB)
        return bool(energy_dbfs <= threshold + margin_db)


def _bark_centers_hz() -> np.ndarray:
    edges = np.asarray(BARK_EDGES_HZ, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])  # type: ignore[no-any-return]


# ── Kanonische Referenz-Metriken (aus artifact_freedom_gate migriert) ────────
# Exakt dieselbe Mathematik wie die test-gepinnten Implementierungen — nun an
# EINER Stelle, von allen Gates gemeinsam genutzt (§Muster 1, kein Wert-Drift).

_BARK_CENTERS_REF_HZ = np.array(
    [
        50,
        150,
        250,
        350,
        450,
        570,
        700,
        840,
        1000,
        1170,
        1370,
        1600,
        1850,
        2150,
        2500,
        2900,
        3400,
        4000,
        4800,
        5800,
        7000,
        8500,
        10500,
        13500,
    ],
    dtype=np.float64,
)
_BARK_VALUES_REF = np.array(
    [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        17.5,
        18.5,
        19.5,
        20.5,
        21.5,
        22.5,
        23.5,
    ],
    dtype=np.float64,
)
_ROUGHNESS_CALIB_ASPER = 1.5e-3  # ~1 asper bei 70-Hz-AM, 60 dB SPL, 100 % AM


def roughness_asper(audio: np.ndarray, sr: int) -> float:
    """Kanonische Roughness in asper (Zwicker 1991, Hilbert-Hüllkurve).

    Modulationsband-Energie 15–300 Hz der Hilbert-Hüllkurve, kalibriert auf
    1.5e-3 (Referenz: 1 kHz-Ton, 60 dB SPL, 100 % AM bei 70 Hz), Cap [0, 10].
    """
    if len(audio) < int(0.1 * sr):
        return 0.0
    audio_arr = np.asarray(audio, dtype=np.float64)
    analytic = np.asarray(hilbert(audio_arr), dtype=np.complex128)
    envelope = np.asarray(np.abs(analytic), dtype=np.float64)
    envelope -= np.mean(envelope)
    env_fft = np.abs(np.fft.rfft(envelope))
    env_freqs = np.fft.rfftfreq(len(envelope), d=1.0 / sr)
    mask = (env_freqs >= 15.0) & (env_freqs <= 300.0)
    if not np.any(mask):
        return 0.0
    am_energy = float(np.sum(env_fft[mask] ** 2)) / max(len(audio), 1)
    roughness = float(am_energy / (_ROUGHNESS_CALIB_ASPER + 1e-12))
    return max(0.0, min(roughness, 10.0))


def sharpness_acum(audio: np.ndarray, sr: int) -> float:
    """Kanonische Sharpness in acum (Bismarck 1974 / DIN 45692, Bark-Zentroid).

    0.11 × ∫ N'(z)·g(z)·z dz / ∫ N'(z) dz über 24 kritische Bänder mit
    g(z) = 1 für z ≤ 16, sonst 0.066·exp(0.171·z); Cap [0, 10].
    """
    if len(audio) < int(0.05 * sr):
        return 0.0

    def _g(z: float) -> float:
        return 1.0 if z <= 16.0 else 0.066 * np.exp(0.171 * z)

    n_fft = min(4096, len(audio))
    win = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(audio[:n_fft] * win))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    band_edges = np.append(_BARK_CENTERS_REF_HZ, 16000.0)
    bandwidths = np.diff(band_edges)
    n_prime = np.zeros(len(_BARK_CENTERS_REF_HZ), dtype=np.float64)
    for i, (f_c, bw_hz) in enumerate(zip(_BARK_CENTERS_REF_HZ, bandwidths)):
        mask = (freqs >= f_c - bw_hz / 2) & (freqs < f_c + bw_hz / 2)
        n_prime[i] = float(np.sum(mag[mask] ** 2))

    total_n = float(np.sum(n_prime))
    if total_n < 1e-12:
        return 0.0
    g_weights = np.array([_g(z) for z in _BARK_VALUES_REF], dtype=np.float64)
    weighted_sum = float(np.sum(n_prime * g_weights * _BARK_VALUES_REF))
    sharpness = 0.11 * weighted_sum / total_n
    return max(0.0, min(sharpness, 10.0))


def build_psychoacoustic_frame(audio: np.ndarray, sr: int) -> PsychoacousticFrame:
    """Baut den gemeinsamen Frame für ein (mono oder stereo) Signal.

    NaN/Inf-frei; bei zu kurzem Signal werden konservativ leere Werte geliefert,
    alle Metriken bleiben float und [0, 1].
    """
    mono = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    mono = np.asarray(mono, dtype=np.float32)

    # Feines STFT (Bark/Sharpness/Maskierung)
    _, _, Zxx = safe_stft(mono, fs=float(sr), window="hann", nperseg=2048, noverlap=1536)
    stft_mag = np.abs(Zxx).astype(np.float32)
    freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
    stft_times = np.arange(stft_mag.shape[1]) * (512 / sr)

    # Bark-Masken aus BARK_EDGES_HZ (Zwicker 1961)
    edges = np.asarray(BARK_EDGES_HZ, dtype=np.float64)
    n_bark = max(len(edges) - 1, 1)
    bark_masks = np.zeros((n_bark, len(freqs)), dtype=bool)
    for b in range(n_bark):
        bark_masks[b] = (freqs >= edges[b]) & (freqs < edges[b + 1])

    bark_energy = np.zeros((n_bark, stft_mag.shape[1]), dtype=np.float64)
    for b in range(n_bark):
        if np.any(bark_masks[b]):
            bark_energy[b] = np.mean(stft_mag[bark_masks[b], :] ** 2, axis=0)
    bark_energy_db = 10.0 * np.log10(np.mean(bark_energy, axis=1) + 1e-12)
    masking_threshold_db = bark_energy_db - _MASKING_SPREAD_DB

    # Fluss-Hüllkurve (hop 128 → Envelope-Rate sr/128) aus einem schlanken
    # 512-Punkt-STFT — positive spektrale Flussdifferenzen + MA-Glättung in
    # roughness_windows. Kalibriert: sauberer 440-Hz-Sinus max 0.04, 70-Hz-AM
    # 0.99; ~8× schneller als die librosa-onset_strength-Route (n_fft=2048).
    _, _, Zf = safe_stft(mono, fs=float(sr), window="hann", nperseg=512, noverlap=384)
    flux_mag = np.abs(Zf).astype(np.float64)
    onset_env = np.sum(np.maximum(np.diff(flux_mag, axis=1), 0.0), axis=0)
    onset_env = np.concatenate([[0.0], onset_env])  # Länge = n_frames_flux
    env_rate = float(sr) / 128.0

    return PsychoacousticFrame(
        sr=sr,
        stft_mag=stft_mag,
        freqs=freqs,
        stft_times=stft_times,
        stft_hop=512,
        onset_env=onset_env,
        env_rate=env_rate,
        bark_masks=bark_masks,
        bark_energy_db=bark_energy_db,
        masking_threshold_db=masking_threshold_db,
        mono=mono,
    )
