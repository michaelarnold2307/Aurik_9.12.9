"""§CODEC-DISKRIMINATOR — Unterscheidet MP3/AAC-Artefakte von echten analogen Defekten.

Wird vom DefectScanner in jedem Detektor konsultiert, um False Positives
durch verlustbehaftete Codecs (mp3_low, mp3_high, aac, streaming) zu verhindern.

Diskriminations-Logik pro Defekttyp:
  - Clicks:     MP3-Block-Boundary-Gitter (26ms @ 44100) vs zufällige Vinyl-Clicks
  - Crackle:    Onset-Korrelation (MP3 pre-echo folgt Transienten, Vinyl nicht)
  - Wow/Flutter: Spectral Flatness (MP3 SBR-Lücken sind bandbegrenzt, Wow ist breitbandig)
  - Tape-Hiss:  Subband-Noise-Varianz (MP3 hat hohe Varianz pro Subband, Tape nicht)
  - Dropouts:   Frame-Loss-Signatur (26ms-Multiple, abrupt) vs Tape-Aussetzer (weich)
  - Surface-Noise: Quantisierungs-Stufen (MP3 = 8-bit-ähnliche Stufen, Vinyl = kontinuierlich)

Alle Methoden sind stateless — sie bekommen Audio + Codec-Kontext und geben
einen Diskriminations-Faktor [0.0, 1.0] zurück (1.0 = definitiv analog, 0.0 = definitiv Codec).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── MP3 Frame Constants ────────────────────────────────────────
MP3_FRAME_SAMPLES_44100 = 1152  # MPEG Layer 3 frame length in samples
MP3_FRAME_SAMPLES_48000 = 1152  # Same in samples (frame is constant in samples, not ms)

# ── Public API ─────────────────────────────────────────────────


class CodecDiscriminator:
    """Stateless-Diskriminator für analoge vs. Codec-Defekte."""

    def __init__(self, terminal_codec: str | None = None, discount: float = 1.0):
        self._terminal_codec = terminal_codec
        self._discount = discount
        self._enabled = terminal_codec is not None and discount < 1.0

    @property
    def has_terminal_codec(self) -> bool:
        return self._enabled

    @property
    def codec_discount(self) -> float:
        return self._discount

    # Convenience: onset_correlation wrapper für DefectScanner-Kompatibilität
    def onset_correlation(self, audio, locations_s: list) -> float:
        """Wrapper für crackle_onset_correlation: Zeit-basierte Locations → Sample-Indizes."""
        if not self._enabled or not locations_s:
            return 0.0
        try:
            onsets = self._detect_transients(audio)
            regions = [(int(s * 44100), int(e * 44100)) for s, e in locations_s if e > s]
            return self.crackle_onset_correlation(regions, onsets)
        except Exception as e:
            logger.warning("codec_discriminator.py::onset_correlation Ersatzpfad: %s", e)
            return 0.0

    def mp3_boundary_fraction(self, locations_s: list[tuple[float, float]]) -> float:
        """Wrapper für click_boundary_density: Zeit → Sample-Konvertierung."""
        if not self._enabled or not locations_s:
            return 0.0
        indices = [int((s + e) / 2 * 44100) for s, e in locations_s if e > s]
        return self.click_boundary_density(indices, 44100)

    def _detect_transients(self, audio) -> list:
        """Einfache Transienten-Detektion via Energie-Anstieg."""
        try:
            import numpy as np

            a = np.asarray(audio).ravel()
            if len(a) < 1024:
                return []
            energy = np.abs(a[::256])
            onsets = []
            for i in range(1, len(energy) - 1):
                if energy[i] > 3.0 * max(energy[i - 1], 1e-12) and energy[i] > energy[i + 1] * 1.5:
                    onsets.append(i * 256)
            return onsets[:500]  # Max 500 onsets
        except Exception as e:
            logger.warning("codec_discriminator.py::_erkennen_transients Ersatzpfad: %s", e)
            return []

    # ── Clicks: MP3-Block-Boundary vs. Vinyl-Click ─────────

    def is_codec_click(self, click_sample_idx: int, sample_rate: int) -> bool:
        """Prüft ob ein Click exakt auf einer MP3-Blockgrenze liegt.

        MP3-Frames sind 1152 Samples lang. Codec-Block-Artefakte treten
        periodisch an diesen Grenzen auf. Echte Vinyl-Clicks sind zufällig verteilt.

        Returns True wenn Codec-Ursprung wahrscheinlich."""
        if not self._enabled:
            return False
        frame_samples = (
            MP3_FRAME_SAMPLES_44100 if sample_rate <= 44100 else int(MP3_FRAME_SAMPLES_48000 * sample_rate / 48000)
        )
        offset = click_sample_idx % frame_samples
        tolerance = max(2, frame_samples // 64)  # ~1.5% tolerance
        return bool(offset <= tolerance or offset >= frame_samples - tolerance)

    def click_boundary_density(self, click_indices: list[int], sample_rate: int) -> float:
        """Anteil der Clicks die auf MP3-Grenzen liegen. >0.6 = Codec."""
        if not click_indices or not self._enabled:
            return 0.0
        on_boundary = sum(1 for i in click_indices if self.is_codec_click(i, sample_rate))
        return on_boundary / max(len(click_indices), 1)

    # ── Crackle: Onset-Korrelation ─────────────────────────

    def crackle_onset_correlation(self, crackle_regions: list[tuple[int, int]], onsets: list[int]) -> float:
        """Misst wie stark Crackle-Regionen mit Transienten-Onsets korrelieren.

        MP3-Pre-Echo: tritt 5-35ms VOR Onsets auf → hohe Korrelation.
        Vinyl-Crackle: zufällig verteilt → niedrige Korrelation.
        Returns 0.0-1.0 (hoch = Codec-verdächtig).
        """
        if not crackle_regions or not onsets or not self._enabled:
            return 0.0
        correlated = 0
        for start, end in crackle_regions:
            for onset in onsets:
                # Ist die Crackle-Region innerhalb von 35ms VOR einem Onset?
                if 0 < onset - end < 35 * 44.1 and onset - start < 40 * 44.1:
                    correlated += 1
                    break
        return correlated / max(len(crackle_regions), 1)

    # ── Wow/Flutter: Spectral Flatness ─────────────────────

    def wow_spectral_flatness_is_codec(self, pitch_curve: np.ndarray, sample_rate: int) -> float:
        """Prüft ob eine Tonhöhenmodulation von MP3-SBR stammt.

        MP3 SBR (Spectral Band Replication) erzeugt bandbegrenzte
        HF-Lücken, die Wow/Flutter ähneln. Echter Wow ist breitbandig
        und hat eine klarere spektrale Signatur.

        Returns Wahrscheinlichkeit 0.0-1.0 dass es Codec ist.
        """
        if not self._enabled or len(pitch_curve) < 8:
            return 0.0
        # FFT der Pitch-Kurve: Wow hat klare <0.5Hz Peaks, MP3 SBR ist breiter
        fft = np.abs(np.fft.rfft(pitch_curve - np.mean(pitch_curve)))
        fft = fft[: max(1, len(fft) // 2)]
        if len(fft) < 3:
            return 0.0
        # Spectral Flatness Measure: geometric / arithmetic mean
        eps = 1e-12
        geo_mean = np.exp(np.mean(np.log(fft + eps)))
        arith_mean = np.mean(fft)
        sfm = geo_mean / (arith_mean + eps)
        # SFM > 0.4 → breitbandig (echtes Wow)
        # SFM < 0.15 → schmalbandig (Codec-SBR-Artefakt)
        if sfm < 0.15:
            return 0.9  # Sehr wahrscheinlich Codec
        elif sfm < 0.4:
            return 0.4  # Unklar
        else:
            return 0.05  # Eher echtes Wow

    # ── Tape-Hiss: Subband-Noise-Varianz ───────────────────

    def hiss_subband_variance_is_codec(self, audio: np.ndarray, sample_rate: int) -> float:
        """Prüft ob HF-Rauschen von MP3-Subband-Quantisierung stammt.

        MP3: Rauschen variiert pro Subband (grobkörnig, signalabhängig).
        Tape: Rauschen ist gleichmäßig über das Spektrum (feinkörnig, konstant).

        Returns Wahrscheinlichkeit 0.0-1.0 dass es Codec ist.
        """
        if not self._enabled or len(audio) < 2048:
            return 0.0
        # 32 Subbänder (wie MP3 Layer 3)
        n_subbands = 32
        fft_n = min(4096, len(audio))
        spec = np.abs(np.fft.rfft(audio[:fft_n] * np.hanning(fft_n)))
        band_size = max(1, len(spec) // n_subbands)
        band_energies = np.array([np.sum(spec[i * band_size : (i + 1) * band_size] ** 2) for i in range(n_subbands)])
        band_energies = band_energies[band_energies > 1e-12]
        if len(band_energies) < 4:
            return 0.0
        # CV der Subband-Energien: MP3 hat hohe Varianz, Tape nicht
        cv = float(np.std(band_energies) / (np.mean(band_energies) + 1e-12))
        # CV > 0.8 → starke Subband-Varianz (Codec)
        # CV < 0.3 → gleichmäßig (Tape)
        return float(np.clip((cv - 0.3) / 0.7, 0.0, 1.0))

    # ── Dropouts: Frame-Loss-Signatur ──────────────────────

    def dropout_is_frame_loss(self, gap_start_ms: float, gap_duration_ms: float, sample_rate: int) -> bool:
        """Prüft ob eine Dropout-Lücke exakt einem MP3-Frame-Multiple entspricht.

        MP3-Frame-Loss: Lücken sind exakte Vielfache von ~26.1ms.
        Tape-Dropout: weiche, variable Lückenlängen.

        Returns True wenn Codec-Frame-Loss wahrscheinlich.
        """
        if not self._enabled:
            return False
        frame_ms = (MP3_FRAME_SAMPLES_44100 / 44100.0) * 1000.0  # ~26.1ms
        duration_frames = gap_duration_ms / frame_ms
        # Prüfe ob die Dauer ein (annäherndes) Vielfaches der Frame-Länge ist
        remainder = duration_frames - round(duration_frames)
        return bool(abs(remainder) < 0.15)  # <15% Abweichung

    # ── Surface-Noise: Quantisierungs-Stufen ───────────────

    def surface_noise_is_quantization(self, audio_segment: np.ndarray) -> float:
        """Prüft ob Grundrauschen von MP3-Quantisierung stammt.

        MP3: 8-bit-ähnliche Quantisierungsstufen im Zeitbereich erkennbar.
        Vinyl: Kontinuierliches Rauschen ohne diskrete Stufen.

        Returns Wahrscheinlichkeit 0.0-1.0 dass es Codec ist.
        """
        if not self._enabled or len(audio_segment) < 256:
            return 0.0
        # Histogramm-Methode: Zähle wie viele Samples auf diskreten Pegelstufen liegen
        hist, _ = np.histogram(audio_segment, bins=64)
        hist_norm = hist / (np.sum(hist) + 1e-12)
        # Spitzen im Histogramm deuten auf Quantisierungsstufen
        peaks = int(np.sum(hist_norm > 3.0 / 64))  # >3x Mittelwert
        # >8 Peaks = starke Quantisierung (Codec)
        return float(np.clip((peaks - 3) / 10.0, 0.0, 1.0))


def make_discriminator(chain: list[str] | None) -> CodecDiscriminator:
    """Erstellt einen CodecDiscriminator basierend auf der Transferkette.

    Args:
        chain: Transferkette, z.B. ['vinyl', 'cassette', 'mp3_low']

    Returns:
        CodecDiscriminator mit terminal_codec und discount gesetzt.
    """
    if chain is None or len(chain) < 2:
        return CodecDiscriminator(None, 1.0)

    _DIGITAL_LOSSY = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
    terminal = str(chain[-1]).lower()
    if terminal not in _DIGITAL_LOSSY:
        return CodecDiscriminator(None, 1.0)

    _DISCOUNT_MAP = {
        "mp3_low": 0.45,
        "mp3_high": 0.60,
        "aac": 0.55,
        "streaming": 0.50,
        "minidisc": 0.65,
    }
    discount = _DISCOUNT_MAP.get(terminal, 0.60)
    return CodecDiscriminator(terminal, discount)
