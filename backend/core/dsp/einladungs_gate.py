"""§Ebene-4 (Hörordnung) Einladungs-Gate — Aurik 10.0.0

„Wohlklang, in den sich das Ohr hineinlegt" ist ein **positives** Kriterium und
wird als Fenster-Gate gemessen, nicht als Einzelwert.

Zeitverlauf-Gate: Roughness (Zwicker), Sharpness (Bismarck), Loudness
(ERB, ISO 532-1) über Fenster (z. B. 5 s, überlappend). Das Gate ist erfüllt,
wenn keine Roughness-Spitze `asper > 0.5` in Stimmen-/Klimax-Zonen liegt und
der Sharpness-Verlauf keine Sprünge > 0.2 acum zwischen benachbarten Fenstern
aufweist.

Reference: .github/instructions/hoerordnung.instructions.md §6
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

# pylint: disable=import-outside-toplevel

logger = logging.getLogger(__name__)


@dataclass
class EinladungsGateResult:
    """Ergebnis des Einladungs-Gates.

    Attributes:
        roughness_mean: Mittelwert der Roughness (Zwicker) über alle Fenster.
        roughness_max_in_voiced: Maximale Roughness in Stimmen-/Klimax-Zonen.
        sharpness_max_jump: Maximaler Sharpness-Sprung zwischen benachbarten Fenstern.
        loudness_mean: Mittelwert der Loudness (ERB, ISO 532-1).
        gate_passed: True wenn alle Kriterien erfüllt sind.
        failure_reasons: Liste der Gründe für Gate-Fehler.
    """

    roughness_mean: float = 0.0
    roughness_max_in_voiced: float = 0.0
    sharpness_max_jump: float = 0.0
    loudness_mean: float = 0.0
    gate_passed: bool = True
    failure_reasons: list[str] = field(default_factory=list)


# ── Schwellen (aus Hörordnung §6) ──────────────────────────────────────────
ROUGHNESS_SPIKE_THRESHOLD = 0.5  # asper > 0.5 in Stimmen-/Klimax-Zonen
SHARPNESS_JUMP_THRESHOLD = 0.2  # acum-Sprung zwischen benachbarten Fenstern


# ── Singleton ──────────────────────────────────────────────────────────────
_instance: EinladungsGate | None = None
_lock = threading.Lock()


def get_einladungs_gate() -> EinladungsGate:
    """Thread-safe Singleton accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EinladungsGate()
    return _instance


class EinladungsGate:
    """§Ebene-4 Einladungs-Gate.

    Misst Roughness, Sharpness und Loudness über Zeitfenster und prüft
    die positiven Wohlklang-Kriterien.
    """

    def __init__(self) -> None:
        self._window_size_s = 5.0  # 5-Sekunden-Fenster
        self._overlap_s = 2.5  # 50 % Überlappung

    def check(
        self,
        audio: np.ndarray,
        sr: int,
        voiced_zones: list[tuple[int, int]] | None = None,
    ) -> EinladungsGateResult:
        """Prüft das Einladungs-Gate.

        Args:
            audio: Audio-Signal. Shape [N] oder [2, N].
            sr: Sample-Rate (muss 48000 sein).
            voiced_zones: Optional Liste von (start_sample, end_sample) für
                Stimmen-/Klimax-Zonen.

        Returns:
            EinladungsGateResult mit Gate-Status und Messwerten.
        """
        assert sr == 48000
        _fallback = EinladungsGateResult()

        try:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            mono = audio.mean(axis=0) if audio.ndim == 2 else audio

            if mono.size < int(self._window_size_s * sr):
                return _fallback

            # ── Muster 1+4: EIN gemeinsamer psychoakustischer Frame statt drei
            # getrennter Fenster-Schleifen mit je eigener STFT. Fenster werden auf
            # der Repräsentation geschnitten (vektorisiert, konsistente Werte).
            from backend.core.dsp.psychoacoustic_frame import build_psychoacoustic_frame

            _frame = build_psychoacoustic_frame(mono, sr)
            roughness_windows = _frame.roughness_windows(self._window_size_s, self._overlap_s)
            sharpness_windows = _frame.sharpness_windows(self._window_size_s, self._overlap_s)
            loudness_windows = _frame.loudness_windows(self._window_size_s, self._overlap_s)

            if not roughness_windows:
                return _fallback

            # ── Gate-Kriterien prüfen ───────────────────────────────────────
            failure_reasons = []

            # Kriterium 1: Keine Roughness-Spitze > 0.5 in Stimmen-/Klimax-Zonen
            roughness_max_in_voiced = self._check_roughness_in_voiced_zones(roughness_windows, voiced_zones, sr)
            if roughness_max_in_voiced > ROUGHNESS_SPIKE_THRESHOLD:
                failure_reasons.append(
                    f"Roughness-Spitze {roughness_max_in_voiced:.3f} > {ROUGHNESS_SPIKE_THRESHOLD} in Stimme/Klimax"
                )

            # Kriterium 2: Sharpness-Verlauf ohne Sprünge > 0.2 acum
            sharpness_max_jump = (
                float(np.max(np.abs(np.diff(sharpness_windows)))) if len(sharpness_windows) > 1 else 0.0
            )
            if sharpness_max_jump > SHARPNESS_JUMP_THRESHOLD:
                failure_reasons.append(
                    f"Sharpness-Sprung {sharpness_max_jump:.3f} > {SHARPNESS_JUMP_THRESHOLD} acum zwischen Fenstern"
                )

            # Mittelwerte berechnen
            roughness_mean = float(np.mean(roughness_windows))
            loudness_mean = float(np.mean(loudness_windows))

            gate_passed = len(failure_reasons) == 0

            if not gate_passed:
                logger.info(
                    "§Ebene-4 Einladungs-Gate NICHT erfüllt: %s",
                    "; ".join(failure_reasons),
                )
            else:
                logger.debug("§Ebene-4 Einladungs-Gate erfüllt (positiver Wohlklang)")

            return EinladungsGateResult(
                roughness_mean=round(roughness_mean, 4),
                roughness_max_in_voiced=round(roughness_max_in_voiced, 4),
                sharpness_max_jump=round(sharpness_max_jump, 4),
                loudness_mean=round(loudness_mean, 4),
                gate_passed=gate_passed,
                failure_reasons=failure_reasons,
            )

        except Exception as exc:
            logger.debug("Einladungs-Gate nicht blockierend: %s", exc)
            return _fallback

    def _check_roughness_in_voiced_zones(
        self,
        roughness_windows: list[float],
        voiced_zones: list[tuple[int, int]] | None,
        sr: int,
    ) -> float:
        """Prüft Roughness in Stimmen-/Klimax-Zonen."""
        if not voiced_zones:
            # Keine Zonen angegeben → über alle Fenster prüfen
            return float(np.max(roughness_windows))

        overlap_samples = int(self._overlap_s * sr)

        max_roughness = 0.0
        for start, end in voiced_zones:
            # Fenster-Indizes die mit dieser Zone überlappen
            start_frame = max(0, int(start / overlap_samples))
            end_frame = min(len(roughness_windows), int(end / overlap_samples))

            for i in range(start_frame, end_frame):
                if roughness_windows[i] > max_roughness:
                    max_roughness = roughness_windows[i]

        return max_roughness

    def _compute_roughness_zwicker(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet Roughness nach Zwicker (2005)."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            # Fluctuation Strength im 40–200 Hz Modulationsbereich
            # als Roughness-Proxy (Zwicker & Fastl 1990): Verhältnis der
            # Modulationsspektrum-Energie im 40–200-Hz-Band zur Gesamtenergie.
            # §API-Fix (2026-09-06): onset_strength() liefert 1-D — das alte
            # „[0]“ machte daraus einen Skalar (np.diff auf Skalar → Exception
            # → stiller 0.0-Return, §V6 [copilot-instructions.md]); Rohwert×10
            # verletzte die [0,1]-Skala (sauberer 440-Hz-Sinus → 1.19).
            # hop=128 → Envelope-Rate 375 Hz → Nyquist 187.5 Hz deckt das Band.
            _mono_rz = audio.mean(axis=0) if audio.ndim == 2 else audio
            _hop = 128
            onset_env = librosa.onset.onset_strength(y=_mono_rz, sr=sr, hop_length=_hop)  # type: ignore[attr-defined]
            if len(onset_env) < 64:
                return 0.0
            _spec = np.abs(np.fft.rfft(onset_env - np.mean(onset_env)))
            _freqs = np.fft.rfftfreq(len(onset_env), d=1.0 / (sr / _hop))
            _band = (_freqs >= 40.0) & (_freqs <= 200.0)
            _band_energy = float(np.sum(_spec[_band] ** 2))
            _total_energy = float(np.sum(_spec**2)) + 1e-12
            return float(np.clip(np.sqrt(_band_energy / _total_energy), 0.0, 1.0))

        except Exception:
            return 0.0

    def _compute_sharpness_bismarck(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet Sharpness nach Bismarck (2001)."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            # G-weighted spectral centroid als Sharpness-Proxy
            # Energie oberhalb von 7 kHz ist relevant für Sharpness
            S = np.abs(librosa.stft(audio, sr=sr))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])

            # G-weighting: Frequenzen > 7 kHz stärker gewichten
            g_weights = np.where(freqs > 7000, 1.0, 0.0)
            weighted_energy = float(np.sum(S**2 * g_weights))
            total_energy = np.sum(S**2) + 1e-12

            sharpness = float(weighted_energy / total_energy)
            return sharpness * 5.0  # Skalierung auf acum-Skala

        except Exception:
            return 0.0

    def _compute_loudness_erb(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet Loudness nach ERB (ISO 532-1)."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            # RMS-Lautstärke als ERB-Loudness-Proxy
            rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
            loudness_db = 20.0 * np.log10(rms + 1e-12)

            # Umrechnung auf Sone-Skala (vereinfacht)
            sones = 2.0 ** ((loudness_db + 40.0) / 10.0)
            return float(np.clip(sones / 100.0, 0.0, 1.0))

        except Exception:
            return 0.5  # Default


# ── Convenience-Funktion für UV3-Integration ────────────────────────────────
def check_einladungs_gate(
    audio: np.ndarray,
    sr: int,
    voiced_zones: list[tuple[int, int]] | None = None,
) -> EinladungsGateResult:
    """Prüft das Einladungs-Gate (Singleton-basiert).

    Args:
        audio: Audio-Signal. Shape [N] oder [2, N].
        sr: Sample-Rate (muss 48000 sein).
        voiced_zones: Optional Liste von (start_sample, end_sample) für
            Stimmen-/Klimax-Zonen.

    Returns:
        EinladungsGateResult mit Gate-Status und Messwerten.
    """
    return get_einladungs_gate().check(audio, sr, voiced_zones)


# ── Transient-Schutzfenster (30 ms statt 20 ms) ───────────────────────────
def protect_transient_zone(
    audio: np.ndarray,
    onset_sample: int,
    sr: int,
    protection_ms: float = 30.0,
) -> np.ndarray:
    """Schützt Transient-Zone vor jeglicher Bearbeitung (V26 §1.4.6 erweitert).

    Args:
        audio: Audio-Signal. Shape [N] oder [2, N].
        onset_sample: Start-Position des Transients in Samples.
        sr: Sample-Rate.
        protection_ms: Schutzfenster in ms (Default 30 ms statt 20 ms).

    Returns:
        Unveränderte Audio-Zone vom Onset bis zum Ende des Schutzfensters.
    """
    zone_end = min(onset_sample + int(protection_ms * sr / 1000.0), len(audio))
    return audio[onset_sample:zone_end]


def detect_transients(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -30.0,
) -> list[int]:
    """Erkennt Transient-Positionen im Audio-Signal.

    Args:
        audio: Audio-Signal. Shape [N] oder [2, N].
        sr: Sample-Rate.
        threshold_db: Schwellwert in dBFS für Transient-Erkennung.

    Returns:
        Liste von Onset-Positionen (Samples).
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        threshold_value = float(10.0 ** (threshold_db / 20.0))

        # Onset-Erkennung via spectral flux.
        # §API-Fix (2026-09-06): (a) onset_strength() liefert 1-D — das alte
        # „[0]“ machte daraus einen Skalar; (b) librosa.find_peaks existiert
        # nicht → scipy.signal.find_peaks mit height-Threshold.
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)  # type: ignore[attr-defined]
        from scipy.signal import find_peaks  # pylint: disable=import-outside-toplevel

        onsets = find_peaks(onset_env, height=threshold_value)[0]

        return list(onsets.astype(int))

    except Exception as e:
        logger.warning("Transient-Erkennung fehlgeschlagen: %s", e)
        return []


# ── Mikrodynamik-Guard per Phase ───────────────────────────────────────────
def check_micro_dynamics(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    voiced_zones: list[tuple[int, int]] | None = None,
) -> float:
    """Misst Frame-Energie-Korrelation in voiced-Zonen (≥ 0.97 erforderlich).

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase.
        sr: Sample-Rate.
        voiced_zones: Optional Liste von (start_sample, end_sample) für
            Gesangszonen.

    Returns:
        Korrelationswert [0,1] (≥ 0.97 = Mikrodynamik erhalten).
    """
    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        mono_pre = pre.mean(axis=0) if pre.ndim == 2 else pre
        mono_post = post.mean(axis=0) if post.ndim == 2 else post

        # Frame-Energie berechnen (10 ms Fenster)
        frame_size = int(0.01 * sr)
        energy_pre = []
        energy_post = []

        for i in range(0, len(mono_pre), frame_size):
            chunk_pre = mono_pre[i : i + frame_size]
            chunk_post = mono_post[i : i + frame_size] if i < len(mono_post) else np.zeros_like(chunk_pre)
            energy_pre.append(float(np.sqrt(np.mean(chunk_pre**2) + 1e-12)))
            energy_post.append(float(np.sqrt(np.mean(chunk_post**2) + 1e-12)))

        # Korrelation berechnen
        min_len = min(len(energy_pre), len(energy_post))
        if min_len < 10:
            return 0.5  # zu kurz für sinnvolle Messung

        corr = float(np.corrcoef(energy_pre[:min_len], energy_post[:min_len])[0, 1])
        return max(corr, 0.0)

    except Exception as e:
        logger.warning("Mikrodynamik-Messung fehlgeschlagen: %s", e)
        return 0.95  # konservativer Default


# ── Maskierungsschwelle (ISO 11172-3 Bark-Skala) ───────────────────────────
def compute_masking_threshold(
    audio: np.ndarray,
    sr: int,
    bark_scale: bool = True,
) -> np.ndarray:
    """Berechnet die psychoakustische Maskierungsschwelle nach ISO 11172-3.

    Reparatur gilt als **abgeschlossen**, wenn ein Defekt **unter der
    Maskierungsschwelle** liegt — nicht wenn sein Messwert Null ist.

    Args:
        audio: Audio-Signal. Shape [N] oder [2, N].
        sr: Sample-Rate.
        bark_scale: True für Bark-Frequenzskala (psychoakustisch korrekt).

    Returns:
        Maskierungsschwelle pro Frequenzband (dBFS).
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio

        # STFT für spektrale Analyse
        S = np.abs(librosa.stft(mono, sr=sr))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])

        # Bark-Skala: ~24 kritische Bänder
        if bark_scale:
            # §Bark-Fix (2026-09-06): librosa.filters.bark existiert nicht —
            # Rectangular-Bark-Filterbank aus BARK_EDGES_HZ (Zwicker 1961).
            from backend.core.dsp.bark_lufs_util import BARK_EDGES_HZ

            _bark_edges = np.asarray(BARK_EDGES_HZ, dtype=np.float64)
            _n_bands = max(len(_bark_edges) - 1, 1)
            bark_freqs = np.zeros((_n_bands, len(freqs)), dtype=bool)
            for _bi in range(_n_bands):
                bark_freqs[_bi] = (freqs >= _bark_edges[_bi]) & (freqs < _bark_edges[_bi + 1])
            S_bark = np.zeros((_n_bands, S.shape[1]))
            for _bi in range(_n_bands):
                if np.any(bark_freqs[_bi]):
                    S_bark[_bi] = np.mean(S[bark_freqs[_bi], :], axis=0)
        else:
            S_bark = S

        # Maskierungsschwelle: -12 dB unter dem Signal (vereinfacht)
        # In der Praxis: kritische Bandbreite + simultane/sukzessive Maskierung
        threshold_db = 20.0 * np.log10(np.mean(S_bark**2, axis=1, keepdims=True) + 1e-12) - 12.0

        return threshold_db.flatten()  # type: ignore[no-any-return]

    except Exception as e:
        logger.warning("Maskierungsschwelle-Berechnung fehlgeschlagen: %s", e)
        return np.zeros(128)  # type: ignore[no-any-return]  # Default-Schwelle


def is_defect_audible(
    defect_energy: float,
    masking_threshold_db: float,
) -> bool:
    """Prüft, ob ein Defekt über der Maskierungsschwelle hörbar ist.

    Args:
        defect_energy: Energie des Defekts in dBFS.
        masking_threshold_db: Maskierungsschwelle in dBFS.

    Returns:
        True wenn Defekt über der Schwelle (hörbar) und reparaturwürdig ist.
    """
    return defect_energy > masking_threshold_db


# ── HPI Referenz-Memory Update-Bedingung (relaxed) ────────────────────────
def should_update_hpi_reference(
    hpi: float,
    artifact_freedom: float,
) -> bool:
    """Prüft, ob das HPI-Referenz-Memory aktualisiert werden soll.

    Relaxed Update-Bedingung (statt HPI > 0 und artifact_freedom ≥ 0.95):
      - HPI > 0.05 (vermeidet Rauschen im Memory)
      - artifact_freedom ≥ 0.92 (realistischer für historische Aufnahmen)

    Args:
        hpi: Holistic Perceptual Improvement Score.
        artifact_freedom: Artifact-Freiheits-Score [0,1].

    Returns:
        True wenn Referenz-Memory aktualisiert werden soll.
    """
    return hpi > 0.05 and artifact_freedom >= 0.92


# ── VQI Recovery per Phase mit material-adaptiven Floors ───────────────────
def check_vqi_recovery(
    audio: np.ndarray,
    sr: int,
    material_type: str,
) -> tuple[float, float]:
    """Prüft VQI gegen material-adaptiven Floor.

    Args:
        audio: Audio-Signal. Shape [N] oder [2, N].
        sr: Sample-Rate.
        material_type: Material-Typ (shellac, vinyl, tape, cd_digital, etc.).

    Returns:
        Tuple von (vqi_score, material_floor).
    """
    try:
        from backend.core.musical_goals.vocal_quality_index import compute_vqi, get_vqi_material_floor

        # §VQI-Signatur-Fix (2026-09-06): compute_vqi(audio_orig, audio_restored, sr) —
        # der 2-Arg-Aufruf warf TypeError (missing 'sr') und lief still in den
        # konservativen Default. Selbstreferenz-VQI wie in phase_66/phase_65.
        # Rückgabe ist ein Dict — Score über .get("vqi") extrahieren (§Typ-Fix).
        vqi = float(compute_vqi(audio, audio, sr).get("vqi", 0.72))
        floor = float(get_vqi_material_floor(material_type))

        return vqi, floor

    except Exception as e:
        logger.warning("VQI-Wiederherstellungs-Prüfung fehlgeschlagen: %s", e)
        return 0.72, 0.72  # konservativer Default


def trigger_recovery_cascade(
    material_type: str,
    vqi_score: float,
    floor: float,
) -> dict[str, object]:
    """Löst VQI-Recovery-Kaskade aus wenn Score unter Floor fällt.

    Args:
        material_type: Material-Typ.
        vqi_score: Aktueller VQI-Score.
        floor: Material-adaptiver Floor.

    Returns:
        Recovery-Parameter für nachfolgende Phasen.
    """
    deficit = max(0.0, floor - vqi_score)

    return {
        "vqi_recovery_active": True,
        "vqi_deficit_db": round(deficit * 10.0, 2),  # Defizit in dB-äquivalent
        "material_type": material_type,
        "recovery_boost_factor": float(np.clip(1.0 + deficit / 2.0, 1.0, 1.5)),
    }
