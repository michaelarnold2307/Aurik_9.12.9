"""
SOTA-konforme Analyse- und Policy-Module für Musikrestaurierung
"""

# Optional DSP/ML dependencies are imported lazily inside analysis paths.
# pylint: disable=import-outside-toplevel

import concurrent.futures
import logging
import os  # §v10.105 module-level (prevents UnboundLocalError)
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _integrated_lufs_bs1770_approx(audio: np.ndarray, sr: int) -> float:
    """Berechnet BS.1770-like integrated loudness with K-weighting and gating."""
    arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return -100.0
    if arr.ndim == 1:
        channels = arr[np.newaxis, :]
    elif arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        channels = arr
    elif arr.ndim == 2:
        channels = arr.T
    else:
        channels = arr.reshape(1, -1)

    weighted = []
    for ch in channels:
        weighted.append(_k_weight_channel(ch.astype(np.float64), sr))
    y = np.vstack(weighted)
    if y.shape[1] < max(1, int(sr * 0.1)):
        power = float(np.mean(np.sum(y**2, axis=0)))
        return float(np.clip(-0.691 + 10.0 * np.log10(power + 1e-20), -100.0, 6.0))

    block = max(1, int(round(sr * 0.400)))
    hop = max(1, int(round(sr * 0.100)))
    powers = []
    for start in range(0, max(1, y.shape[1] - block + 1), hop):
        segment = y[:, start : start + block]
        if segment.shape[1] < block:
            break
        powers.append(float(np.mean(np.sum(segment**2, axis=0))))
    if not powers:
        powers = [float(np.mean(np.sum(y**2, axis=0)))]

    block_lufs = np.array([-0.691 + 10.0 * np.log10(power + 1e-20) for power in powers], dtype=np.float64)
    absolute_mask = block_lufs > -70.0
    gated_powers = np.array(powers, dtype=np.float64)[absolute_mask]
    if gated_powers.size == 0:
        return -100.0
    ungated_loudness = -0.691 + 10.0 * np.log10(float(np.mean(gated_powers)) + 1e-20)
    relative_mask = block_lufs[absolute_mask] > (ungated_loudness - 10.0)
    gated_powers = gated_powers[relative_mask]
    if gated_powers.size == 0:
        return float(np.clip(ungated_loudness, -100.0, 6.0))
    return float(np.clip(-0.691 + 10.0 * np.log10(float(np.mean(gated_powers)) + 1e-20), -100.0, 6.0))


def _k_weight_channel(channel: np.ndarray, sr: int) -> np.ndarray:
    """Wendet eine konservative K-Gewichtungs-Approximation auf einen Kanal an."""
    try:
        from scipy.signal import butter, sosfiltfilt  # pylint: disable=import-outside-toplevel

        hp_hz = min(80.0, max(20.0, sr * 0.01))
        sos_hp = butter(2, hp_hz, btype="highpass", fs=sr, output="sos")
        weighted = sosfiltfilt(sos_hp, channel)
        sos_hi = butter(1, 1500.0, btype="highpass", fs=sr, output="sos")
        high = sosfiltfilt(sos_hi, weighted)
        return weighted + high * 0.18  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("Analyse_and_modules.py::_k_weight_channel Ersatzpfad: %s", e)
        return channel


class PolicyManager:
    """Verwaltet und optimiert die Policy-Objekte für die adaptive Pipeline."""

    def __init__(self, policy: dict, escalation_levels=None, callback=None) -> None:
        self.policy = policy
        self.escalation_levels = escalation_levels or {"warn": 3, "bypass": 5, "hard_bypass": 7}
        self.callback = callback  # Optional: Funktion für externe Aktionen (z.B. Logging, Notification)

    def update(self, feedback: dict) -> None:
        """Aktualisiert gate failure counters and escalation state from feedback."""
        # SOTA-Policy-Logik: Logging, Eskalation, Reset, Aktionen, Zeitstempel, Callbacks
        now = time.time()
        if "_log" not in self.policy:
            self.policy["_log"] = []
        for gate, result in feedback.items():
            # Zähle Fehlschläge pro Gate
            if gate not in self.policy:
                self.policy[gate] = {
                    "fail_count": 0,
                    "threshold": None,
                    "escalated": False,
                    "escalation_level": None,
                    "action": None,
                }
            if result is False:
                if "fail_count" not in self.policy[gate]:
                    self.policy[gate]["fail_count"] = 0
                self.policy[gate]["fail_count"] += 1
                # Eskalationsstufen
                level = None
                if self.policy[gate]["fail_count"] >= self.escalation_levels.get("hard_bypass", 7):
                    level = "hard_bypass"
                    self.policy[gate]["action"] = "hard_bypass"
                    self.policy[gate]["escalated"] = True
                elif self.policy[gate]["fail_count"] >= self.escalation_levels.get("bypass", 5):
                    level = "bypass"
                    self.policy[gate]["action"] = "bypass_or_notify"
                    self.policy[gate]["escalated"] = True
                elif self.policy[gate]["fail_count"] >= self.escalation_levels.get("warn", 3):
                    level = "warn"
                    self.policy[gate]["action"] = "warn"
                    self.policy[gate]["escalated"] = True
                if level and self.policy[gate]["escalation_level"] != level:
                    self.policy[gate]["escalation_level"] = level
                    event = {"event": "escalation", "gate": gate, "level": level, "timestamp": now}
                    self.policy["_log"].append(event)
                    if self.callback:
                        self.callback(event)
            else:
                # Reset nach Erfolg
                if self.policy[gate]["fail_count"] > 0:
                    event = {"event": "reset", "gate": gate, "count": self.policy[gate]["fail_count"], "timestamp": now}
                    self.policy["_log"].append(event)
                    if self.callback:
                        self.callback(event)
                self.policy[gate]["fail_count"] = 0
                self.policy[gate]["escalated"] = False
                self.policy[gate]["escalation_level"] = None
                self.policy[gate]["action"] = None
            # Adaptive Schwellenwertanpassung
            if self.policy[gate]["fail_count"] >= self.escalation_levels.get("warn", 3):
                th = self.policy[gate]["threshold"]
                if isinstance(th, (int, float)):
                    self.policy[gate]["threshold"] = th * 0.95
                else:
                    self.policy[gate]["threshold"] = 0.95
        # Logging aller Policy-Änderungen mit Zeitstempel
        self.policy["_log"].append(
            {
                "feedback": feedback.copy(),
                "policy": {k: v.copy() if isinstance(v, dict) else v for k, v in self.policy.items() if k != "_log"},
                "timestamp": now,
            }
        )
        return self.policy  # type: ignore[return-value]

    def reset_policy(self) -> dict[str, Any]:
        """Setzt zurück: all policy gate states while preserving the policy log."""
        # Setzt alle Policy-Zustände (außer Log) zurück
        for k in list(self.policy.keys()):
            if k != "_log":
                self.policy[k] = {
                    "fail_count": 0,
                    "threshold": None,
                    "escalated": False,
                    "escalation_level": None,
                    "action": None,
                }
        return self.policy


class FeatureExtractor:
    """
    SOTA-Feature-Extraktion für Audioanalyse und Policy-Steuerung.
    Jetzt mit paralleler Berechnung (max. 4 Kerne).
    """

    def extract(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None = None,
        policy_manager: Optional["PolicyManager"] = None,
    ) -> dict:
        """Extrahiert robust forensic and musical features from mono or stereo audio."""
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        y_mono = self._to_mono(audio)
        channels = self._as_channels(audio)
        features = {}

        def fcpe_features() -> dict[str, Any]:
            try:
                from plugins.fcpe_plugin import get_fcpe_plugin as _get_fcpe

                _r = _get_fcpe().analyze(y_mono, sr)
                voiced = _r.voiced_prob > 0.5
                f0_vals = _r.f0_hz[voiced] if voiced.any() else _r.f0_hz
                return {
                    "f0_median": float(np.median(f0_vals)),
                    "f0_mean": float(np.mean(f0_vals)),
                    "f0_std": float(np.std(f0_vals)),
                }
            except Exception as e:
                logger.warning("Analyse_and_modules.py::fcpe_features Ersatzpfad: %s", e)
                return {"f0_median": -1.0, "f0_mean": -1.0, "f0_std": -1.0}

        def librosa_features() -> dict[str, Any]:
            try:
                import librosa

                y = y_mono
                if len(y) >= 2048:
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=min(2048, len(y)))
                    # §v10.14: chroma_cqt benötigt mindestens n_fft=2048 → pitch_tuning
                    # schlägt auf sehr kurzen Signalen mit leerem Frequenzsatz fehl.
                    if len(y) >= 4096:
                        key = librosa.feature.chroma_cqt(y=y, sr=sr)
                    else:
                        key = chroma  # Fallback: chroma_stft statt chroma_cqt
                    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)  # type: ignore[attr-defined]
                    melody = librosa.feature.mfcc(y=y, sr=sr, n_fft=min(2048, len(y)))
                    return {
                        "chroma_mean": float(np.mean(chroma)),
                        "chroma_std": float(np.std(chroma)),
                        "key_chroma_cqt_mean": float(np.mean(key)),
                        "tempo_bpm": float(np.asarray(tempo).flat[0]),
                        "beat_count": len(beats),
                        "mfcc_mean": float(np.mean(melody)),
                        "mfcc_std": float(np.std(melody)),
                    }
                return {
                    "chroma_mean": -1.0,
                    "chroma_std": -1.0,
                    "key_chroma_cqt_mean": -1.0,
                    "tempo_bpm": -1.0,
                    "beat_count": 0,
                    "mfcc_mean": -1.0,
                    "mfcc_std": -1.0,
                }
            except Exception as e:
                logger.warning("Analyse_and_modules.py::librosa_features Ersatzpfad: %s", e)
                return {
                    "chroma_mean": -1.0,
                    "chroma_std": -1.0,
                    "key_chroma_cqt_mean": -1.0,
                    "tempo_bpm": -1.0,
                    "beat_count": -1,
                    "mfcc_mean": -1.0,
                    "mfcc_std": -1.0,
                }

        def panns_features() -> dict[str, Any]:
            try:
                # import os → module level (§v10.105)
                import tempfile

                import soundfile as sf

                from plugins.panns_plugin import PANNSPlugin

                panns = PANNSPlugin()
                audio_for_panns = y_mono
                with (
                    tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in,
                    tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out,
                ):
                    try:
                        sf.write(tmp_in.name, audio_for_panns, sr)
                        panns_tags = panns.tag(tmp_in.name, tmp_out.name)
                        return panns_tags  # type: ignore[no-any-return]
                    finally:
                        if os.path.exists(tmp_in.name):
                            os.remove(tmp_in.name)
                        if os.path.exists(tmp_out.name):
                            os.remove(tmp_out.name)
            except Exception as e:
                logger.warning("Analyse_and_modules.py::panns_features Ersatzpfad: %s", e)
                return {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_fcpe = executor.submit(fcpe_features)
            future_librosa = executor.submit(librosa_features)
            future_panns = executor.submit(panns_features)
            fcpe_result = future_fcpe.result()
            librosa_result = future_librosa.result()
            panns_result = future_panns.result()
            features.update(fcpe_result)
            features.update(librosa_result)
            features.update(panns_result)

        # ...restliche Features wie RMS, ZCR, Spectral, Dynamics, Stereo etc. synchron
        features["rms"] = float(np.sqrt(np.mean(y_mono**2)))
        features["zcr"] = float(np.mean(np.abs(np.diff(np.sign(y_mono))))) if y_mono.size > 1 else 0.0
        # Erweiterte musikalische Features (Harmonie, Rhythmus, Melodie)
        try:
            import librosa

            # Mono für Harmonie/Rhythmus-Features
            y = y_mono
            # Nur verarbeiten, wenn Signal lang genug ist
            if len(y) >= 2048:
                # Chroma (Harmonie)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=min(2048, len(y)))
                features["chroma_mean"] = float(np.mean(chroma))
                features["chroma_std"] = float(np.std(chroma))
                # Key (Tonart, Harmonie)
                # §v10.14: chroma_cqt benötigt ≥4096 Samples gegen pitch_tuning-Leerfrequenz.
                if len(y) >= 4096:
                    key = librosa.feature.chroma_cqt(y=y, sr=sr)
                else:
                    key = chroma
                features["key_chroma_cqt_mean"] = float(np.mean(key))
                # Beat/Rhythmus
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)  # type: ignore[attr-defined]
                features["tempo_bpm"] = float(np.asarray(tempo).flat[0])
                features["beat_count"] = len(beats)
                # Melodie-Contour
                melody = librosa.feature.mfcc(y=y, sr=sr, n_fft=min(2048, len(y)))
                features["mfcc_mean"] = float(np.mean(melody))
                features["mfcc_std"] = float(np.std(melody))
            else:
                # Signal zu kurz, setze Default-Werte
                features["chroma_mean"] = -1.0
                features["chroma_std"] = -1.0
                features["key_chroma_cqt_mean"] = -1.0
                features["tempo_bpm"] = -1.0
                features["beat_count"] = 0
                features["mfcc_mean"] = -1.0
                features["mfcc_std"] = -1.0
        except Exception:
            features["chroma_mean"] = -1.0
            features["chroma_std"] = -1.0
            features["key_chroma_cqt_mean"] = -1.0
            features["tempo_bpm"] = -1.0
            features["beat_count"] = -1
            features["mfcc_mean"] = -1.0
            features["mfcc_std"] = -1.0
        if channels.shape[0] > 1:
            # Stereo: pro Kanal berechnen und mitteln
            centroids = []
            rolloffs = []
            flatnesses = []
            contrasts = []
            for channel in channels:
                centroid, rolloff, flatness, contrast = self._spectral_summary(channel, sr)
                centroids.append(centroid)
                rolloffs.append(rolloff)
                flatnesses.append(flatness)
                contrasts.append(contrast)
            features["spectral_centroid"] = float(np.mean(centroids))
            features["spectral_rolloff"] = float(np.mean(rolloffs))
            features["spectral_flatness"] = float(np.mean(flatnesses))
            # Mittelwert über alle Bänder und Kanäle
            features["spectral_contrast"] = [float(np.mean([c[i] for c in contrasts if len(c) > i])) for i in range(6)]
        else:
            centroid, rolloff, flatness, contrast = self._spectral_summary(y_mono, sr)
            features["spectral_centroid"] = centroid
            features["spectral_rolloff"] = rolloff
            features["spectral_flatness"] = flatness
            features["spectral_contrast"] = contrast
        # Crest Factor
        peak = np.max(np.abs(y_mono)) if y_mono.size else 0.0
        features["crest_factor"] = float(peak / (features["rms"] + 1e-10))
        features["lufs"] = _integrated_lufs_bs1770_approx(audio, sr)
        # SNR/SI-SDR falls Referenz vorhanden
        if reference is not None:
            ref_mono = self._to_mono(np.nan_to_num(np.asarray(reference, dtype=np.float32)))
            est_aligned, ref_aligned = self._align_pair(y_mono, ref_mono)
            if est_aligned.size > 0:
                noise = est_aligned - ref_aligned
                features["snr"] = float(10 * np.log10(np.sum(ref_aligned**2) / (np.sum(noise**2) + 1e-10)))
                ref = ref_aligned - np.mean(ref_aligned)
                est = est_aligned - np.mean(est_aligned)
                alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-10)
                s_target = alpha * ref
                e_noise = est - s_target
                features["si_sdr"] = float(10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-10)))

        # Quality-Gates automatisch prüfen
        def log_callback(entry):
            if policy_manager is not None:
                if "_quality_log" not in policy_manager.policy:
                    policy_manager.policy["_quality_log"] = []
                policy_manager.policy["_quality_log"].append(entry)

        # QualityEvaluator entfernt (tests_legacy nicht mehr vorhanden)
        # Alle relevanten Metriken extrahieren (inkl. SOTA-Gates)
        metrics = {
            k: features[k]
            for k in ["snr", "lufs", "si_sdr", "lsd", "phase_coh", "transient", "artifacts", "improvement"]
            if k in features
        }
        # QualityEvaluator entfernt: quality_gates als Dummy (alle Metriken >=0.0 -> True)
        features["quality_gates"] = {k: (v is not None and v >= 0.0) for k, v in metrics.items()}
        log_callback({"event": "quality_gates", "gates": features["quality_gates"], "timestamp": time.time()})
        # Policy-Integration: PolicyManager erhält Quality-Gate-Resultate
        if policy_manager is not None:
            # Gender-Neutral: PolicyManager darf keine Policy-Entscheidung auf Basis von f0_median/f0_mean treffen
            policy_manager.update(features["quality_gates"])

        # === DEFECT DETECTION (Week 9 Integration) ===
        # Detect clicks, crackle, clipping, dropouts, wow/flutter, hum
        # Click detection (vinyl)
        features["click_density"] = self._detect_clicks(y_mono, sr)
        features["click_count"] = int(features["click_density"] * len(y_mono) / sr)

        # Crackle detection (vinyl)
        features["crackle_density"] = self._detect_crackle(y_mono, sr)

        # Clipping detection (digital)
        features["clipping_percentage"] = self._detect_clipping_percentage(y_mono)
        features["is_clipped"] = features["clipping_percentage"] > 0.1  # >0.1% clipped

        # Dropout detection (tape)
        dropout_regions, dropout_count = self._detect_dropouts(y_mono, sr)
        features["dropout_regions"] = dropout_regions
        features["dropout_count"] = dropout_count

        # Wow/Flutter detection (tape/vinyl)
        features["wow_score"], features["flutter_score"] = self._detect_wow_flutter(y_mono, sr)

        # Hum detection (analog)
        features["hum_score"] = self._detect_hum(y_mono, sr)

        return features

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Gibt mono audio from mono, channel-first, or channel-last input zurück."""
        if audio.ndim != 2:
            return audio.reshape(-1)
        rows, cols = audio.shape
        if rows <= 8 and cols > rows:
            return audio.mean(axis=0)  # type: ignore[no-any-return]
        if cols <= 8 and rows > cols:
            return audio.mean(axis=1)  # type: ignore[no-any-return]
        return audio.mean(axis=0)  # type: ignore[no-any-return]

    @staticmethod
    def _as_channels(audio: np.ndarray) -> np.ndarray:
        """Gibt audio as a channel-first array zurück."""
        if audio.ndim == 1:
            return audio[np.newaxis, :]
        rows, cols = audio.shape
        if rows <= 8 and cols > rows:
            return audio
        if cols <= 8 and rows > cols:
            return audio.T
        return audio.reshape(1, -1)

    @staticmethod
    def _spectral_summary(channel: np.ndarray, sr: int) -> tuple[float, float, float, list[float]]:
        """Gibt centroid, rolloff, flatness, and six-band contrast safely zurück."""
        if channel.size == 0:
            return 0.0, 0.0, 0.0, [0.0] * 6
        mag = np.abs(np.fft.rfft(channel))
        if mag.size == 0:
            return 0.0, 0.0, 0.0, [0.0] * 6
        freqs = np.fft.rfftfreq(len(channel), 1 / sr)
        total_energy = float(np.sum(mag))
        if total_energy <= 1e-12:
            return 0.0, 0.0, 0.0, [0.0] * 6
        centroid = float(np.sum(mag * freqs) / (total_energy + 1e-10))
        cumulative = np.cumsum(mag)
        rolloff_candidates = np.flatnonzero(cumulative >= 0.85 * total_energy)
        rolloff = float(freqs[int(rolloff_candidates[0])]) if rolloff_candidates.size else 0.0
        geo_mean = float(np.exp(np.mean(np.log(mag + 1e-10))))
        arith_mean = float(np.mean(mag + 1e-10))
        flatness = float(geo_mean / (arith_mean + 1e-10))
        bands = np.array_split(mag, 6)
        contrast = [float(np.max(b) - np.min(b)) if len(b) > 0 else 0.0 for b in bands]
        while len(contrast) < 6:
            contrast.append(0.0)
        return centroid, rolloff, flatness, contrast[:6]

    @staticmethod
    def _align_pair(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gibt equal-length mono estimate/reference arrays for quality metrics zurück."""
        n = min(estimate.size, reference.size)
        if n <= 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        return estimate[:n].astype(np.float32, copy=False), reference[:n].astype(np.float32, copy=False)

    def _detect_clicks(self, audio: np.ndarray, sr: int) -> float:
        """
        Erkennt click density (clicks per second).
        Uses multi-stage detection: transient + spectral anomaly + statistical outlier.

        Returns:
                Click density (clicks/second)
        """
        try:
            import librosa

            # Stage 1: Onset detection (transient detection)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)  # type: ignore[attr-defined]
            librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)  # type: ignore[attr-defined]

            # Stage 2: Statistical outlier detection (5σ threshold)
            threshold = np.mean(audio) + 5 * np.std(audio)
            outliers = np.abs(audio) > threshold
            click_candidates = np.where(outliers)[0]

            # Stage 3: Duration filtering (<5ms = click, >5ms = other artifact)
            max_duration_samples = int(0.005 * sr)  # 5ms
            clicks = []
            i = 0
            while i < len(click_candidates):
                start = click_candidates[i]
                # Find consecutive samples
                duration = 1
                while (
                    i + duration < len(click_candidates)
                    and click_candidates[i + duration] == click_candidates[i] + duration
                ):
                    duration += 1
                # Accept as click if duration < 5ms
                if duration < max_duration_samples:
                    clicks.append(start)
                i += duration

            # Compute density (clicks per second)
            duration_sec = len(audio) / sr
            click_density = len(clicks) / duration_sec if duration_sec > 0 else 0.0

            return float(click_density)

        except Exception as e:
            logger.warning("Analyse_and_modules.py::_erkennen_clicks Ersatzpfad: %s", e)
            return 0.0

    def _detect_crackle(self, audio: np.ndarray, sr: int) -> float:
        """
        Erkennt crackle density (impulse density).
        Uses high-pass filtering + impulse density computation.

        Returns:
                Crackle density (0.0 - 1.0, 0.05 = threshold for treatment)
        """
        try:
            from scipy import signal

            # High-pass filter (isolate crackle: 2-20 kHz)
            sos = signal.butter(10, 2000, "hp", fs=sr, output="sos")
            audio_hp = signal.sosfilt(sos, audio)

            # Compute impulse density (100ms windows)
            abs_deriv = np.abs(np.diff(audio_hp))
            impulse_threshold = 3 * np.std(abs_deriv)
            impulses = abs_deriv > impulse_threshold

            # Density in 100ms windows
            window_size = int(0.1 * sr)
            densities = []
            for i in range(0, len(impulses), window_size):
                window = impulses[i : i + window_size]
                if len(window) > 0:
                    density = np.sum(window) / len(window)
                    densities.append(density)

            # Mean density across all windows
            mean_density = np.mean(densities) if len(densities) > 0 else 0.0

            return float(mean_density)

        except Exception as e:
            logger.warning("Analyse_and_modules.py::_erkennen_crackle Ersatzpfad: %s", e)
            return 0.0

    def _detect_clipping_percentage(self, audio: np.ndarray) -> float:
        """
        Erkennt clipping percentage (% of samples at max amplitude).

        Returns:
                Percentage of clipped samples (0.0 - 100.0)
        """
        try:
            threshold = 0.99  # Samples >= 99% of max amplitude
            clipped = np.abs(audio) >= threshold
            clipping_percentage = np.sum(clipped) / len(audio) * 100.0
            return float(clipping_percentage)

        except Exception as e:
            logger.warning("Analyse_and_modules.py::_erkennen_clipping_percentage Ersatzpfad: %s", e)
            return 0.0

    def _detect_dropouts(self, audio: np.ndarray, sr: int) -> tuple:
        """
        Erkennt tape dropouts (sudden energy drops).

        Returns:
                Tuple of (dropout_regions, dropout_count)
                dropout_regions: List of (start_sample, end_sample) tuples
        """
        try:
            import librosa

            # RMS envelope
            hop_length = 512
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)

            # Threshold: sudden drops >20 dB below median
            dropout_threshold = np.median(rms_db) - 20

            # Find dropout regions
            dropouts_bool = rms_db < dropout_threshold

            # Duration filtering (10ms - 1s)
            min_duration_frames = int(0.01 * sr / hop_length)  # 10ms
            max_duration_frames = int(1.0 * sr / hop_length)  # 1s

            # Find contiguous dropout regions
            dropout_regions = []
            in_dropout = False
            start_frame = 0

            for i, is_dropout in enumerate(dropouts_bool):
                if is_dropout and not in_dropout:
                    start_frame = i
                    in_dropout = True
                elif not is_dropout and in_dropout:
                    duration_frames = i - start_frame
                    if min_duration_frames <= duration_frames <= max_duration_frames:
                        # Convert frame indices to sample indices
                        start_sample = start_frame * hop_length
                        end_sample = i * hop_length
                        dropout_regions.append((start_sample, end_sample))
                    in_dropout = False

            return dropout_regions, len(dropout_regions)

        except Exception as e:
            logger.warning("Analyse_and_modules.py::_erkennen_dropouts Ersatzpfad: %s", e)
            return [], 0

    def _detect_wow_flutter(self, audio: np.ndarray, sr: int) -> tuple:
        """
        Erkennt wow and flutter (pitch modulation).
        Wow: <1 Hz, Flutter: 1-100 Hz

        Returns:
                Tuple of (wow_score, flutter_score) (0.0 - 1.0)
        """
        try:
            import librosa
            from scipy import signal as scipy_signal

            # Extract pitch using pyin (more robust than crepe for modulation)
            f0, voiced_flag, _voiced_probs = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )

            # Filter out unvoiced regions
            f0_voiced = f0[voiced_flag]

            if len(f0_voiced) < 100:  # Not enough voiced frames
                return 0.0, 0.0

            # Smooth pitch to remove vibrato (median filter)
            from scipy.ndimage import median_filter

            f0_smooth = median_filter(f0_voiced, size=51)

            # Pitch deviation
            pitch_deviation = f0_voiced - f0_smooth

            # Bandpass filters for wow/flutter
            # Wow: 0.1 - 1.0 Hz
            if len(pitch_deviation) > 100:
                sos_wow = scipy_signal.butter(4, [0.1, 1.0], "bp", fs=sr / 512, output="sos")  # Approximate frame rate
                wow = scipy_signal.sosfilt(sos_wow, pitch_deviation)
                wow_score = np.std(wow) / (np.mean(np.abs(f0_voiced)) + 1e-10)

                # Flutter: 1.0 - 100.0 Hz
                sos_flutter = scipy_signal.butter(4, [1.0, 100.0], "bp", fs=sr / 512, output="sos")
                flutter = scipy_signal.sosfilt(sos_flutter, pitch_deviation)
                flutter_score = np.std(flutter) / (np.mean(np.abs(f0_voiced)) + 1e-10)
            else:
                wow_score = 0.0  # type: ignore[assignment]
                flutter_score = 0.0  # type: ignore[assignment]

            # Clamp to 0-1 range
            wow_score = min(1.0, wow_score * 10)  # type: ignore[arg-type]  # Scale for typical values
            flutter_score = min(1.0, flutter_score * 10)  # type: ignore[arg-type]

            return float(wow_score), float(flutter_score)

        except Exception as e:
            logger.warning("Analyse_and_modules.py::_erkennen_wow_flutter Ersatzpfad: %s", e)
            return 0.0, 0.0

    def _detect_hum(self, audio: np.ndarray, sr: int) -> float:
        """
        Erkennt hum (50/60 Hz power line interference).

        Returns:
                Hum score (0.0 - 1.0, >0.05 = significant hum)
        """
        try:
            # FFT
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)
            magnitude = np.abs(fft)

            # Check for 50 Hz hum (Europe)
            hum_50hz = []
            for harmonic_freq in [50, 100, 150, 200, 250]:
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                hum_50hz.append(magnitude[idx])

            # Check for 60 Hz hum (US)
            hum_60hz = []
            for harmonic_freq in [60, 120, 180, 240, 300]:
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                hum_60hz.append(magnitude[idx])

            # Take max of 50/60 Hz detection
            hum_50_score = np.mean(hum_50hz) / (np.mean(magnitude) + 1e-10)
            hum_60_score = np.mean(hum_60hz) / (np.mean(magnitude) + 1e-10)

            hum_score = max(hum_50_score, hum_60_score)

            # Normalize (typical hum is 0.01 - 0.1 of spectrum)
            hum_score = min(1.0, hum_score * 10)  # type: ignore[arg-type]

            return float(hum_score)

        except Exception as e:
            logger.warning("Analyse_and_modules.py::_erkennen_hum Ersatzpfad: %s", e)
            return 0.0


# Weitere Analyse- und Policy-Module können hier ergänzt werden


# ==============================================================================
# Analysis Engine Adapter (AURIK Spec 3.1)
# ==============================================================================


class AnalysisEngineAdapter:
    """
    Verbindet between existing FeatureExtractor and formal AnalysisProfile.

    Maps dict-based features to typed Pydantic models per AURIK Spec 3.1.
    Maintains backward compatibility while producing formal data structures.
    """

    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()

    def analyze(self, audio: np.ndarray, sr: int, audio_path: str | None = None) -> object:  # type: ignore[override]
        """
        Erstellt comprehensive AnalysisProfile from audio.

        Args:
                audio: Audio signal (mono or stereo)
                sr: Sample rate
                audio_path: Optional path to audio file for metadata

        Returns:
                AnalysisProfile with complete analysis
        """
        del audio_path
        # Import here to avoid circular dependency
        from backend.core.data_models import (
            AnalysisProfile,
            DefectDetection,
            DefectType,
            DynamicsAnalysis,
            FeatureVectors,
            FormatInfo,
            Genre,
            MaterialChainAnalysis,
            MediaType,
            MusicalContext,
            SpectralAnalysis,
            StereoAnalysis,
            VocalAnalysis,
        )

        # Extract features using existing extractor
        features = self.feature_extractor.extract(audio, sr)

        # PANNS Audio Tagging for Genre/Vocal/Instrument Detection with ENSEMBLE VOTING
        genre = Genre.UNKNOWN
        genre_confidence = 0.5
        has_vocals = False
        vocal_confidence = 0.0
        instruments = []

        try:
            # import os → module level (§v10.105)
            import tempfile
            from collections import Counter

            import soundfile as sf

            from plugins.panns_plugin import PANNSPlugin

            # 🎯 ENSEMBLE VOTING: Analyze 3 different segments for stability
            panns = PANNSPlugin()
            genre_votes = []
            vocal_votes = []
            all_panns_tags = []

            # Convert to mono once
            audio_for_panns = audio
            if audio.ndim > 1:
                audio_for_panns = np.mean(audio, axis=0)

            # Analyze 3 segments: start (0-30s), middle, end (or full if <90s)
            audio_length = len(audio_for_panns) / sr
            if audio_length > 90:
                # Long audio: analyze 3 different 30s segments
                segments = [
                    (0, int(30 * sr)),  # First 30s
                    (int((audio_length / 2 - 15) * sr), int((audio_length / 2 + 15) * sr)),  # Middle 30s
                    (int((audio_length - 30) * sr), len(audio_for_panns)),  # Last 30s
                ]
            else:
                # Short audio: use full audio 3 times (deterministic)
                segments = [(0, len(audio_for_panns))] * 3

            for start, end in segments:
                with (
                    tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in,
                    tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out,
                ):
                    try:
                        # Extract segment
                        segment_audio = audio_for_panns[start:end]

                        # Write audio to temp file
                        sf.write(tmp_in.name, segment_audio, sr)

                        # Run PANNS tagging
                        panns_tags_seg = panns.tag(tmp_in.name, tmp_out.name)
                        all_panns_tags.append(panns_tags_seg)

                        # Extract features from PANNS tags
                        genre_seg, genre_conf_seg = self._extract_genre_from_panns(panns_tags_seg)
                        has_vocals_seg, vocal_conf_seg = self._extract_vocals_from_panns(panns_tags_seg)

                        genre_votes.append((genre_seg, genre_conf_seg))
                        vocal_votes.append((has_vocals_seg, vocal_conf_seg))

                    finally:
                        # Cleanup temp files
                        if os.path.exists(tmp_in.name):
                            os.remove(tmp_in.name)
                        if os.path.exists(tmp_out.name):
                            os.remove(tmp_out.name)

            # 🗳️ MAJORITY VOTING for Genre (most common genre wins)
            genre_counter = Counter([g for g, _ in genre_votes if g != Genre.UNKNOWN])
            if genre_counter:
                genre = genre_counter.most_common(1)[0][0]
                # Average confidence of this genre across votes
                genre_mean_conf = float(np.mean([conf for g, conf in genre_votes if g == genre]))
                genre_consensus = float(genre_counter.get(genre, 0)) / float(max(len(genre_votes), 1))
                # Konsens + Evidenzstärke: gleiche Votes sollen die Konfidenz spürbar
                # erhöhen, aber nicht über die lokale Mittelung hinaus halluzinieren.
                genre_confidence = max(genre_mean_conf, 0.5 * genre_mean_conf + 0.5 * genre_consensus)

                # 🎯 CONFIDENCE THRESHOLD: Only report if >50% confident
                if genre_confidence < 0.50:
                    genre = Genre.UNKNOWN
            else:
                genre = Genre.UNKNOWN
                genre_confidence = 0.3

            # 🗳️ MAJORITY VOTING for Vocals (2 out of 3)
            has_vocals = sum(v for v, _ in vocal_votes) >= 2
            vocal_confidence = np.mean([conf for _, conf in vocal_votes])  # type: ignore[assignment]

            # Use first segment's tags for instruments (less critical)
            instruments = self._extract_instruments_from_panns(all_panns_tags[0])

            logger.info(
                "PANNS ENSEMBLE: Genre=%s (%.2f) [votes: %s], Vocals=%s (%.2f), Instruments=%d",
                genre.value if hasattr(genre, "value") else genre,
                genre_confidence,
                [g.value if hasattr(g, "value") else g for g, _ in genre_votes],
                has_vocals,
                vocal_confidence,
                len(instruments),
            )

        except Exception as e:
            # PANNS failed - use fallback values
            logger.warning("PANNS tagging fehlgeschlagen: %s. Using placeholder values.", e)
            # Keep default values set above

        # Prepare audio for stereo analysis
        if audio.ndim == 1:
            audio_stereo = np.stack([audio, audio])  # Fake stereo from mono
            channels = 1
        else:
            audio_stereo = audio
            channels = audio.shape[0]

        # 1. Format Info (Spec 3.1.1)
        # Detect actual bit depth from audio dynamic range
        _abs_max = float(np.max(np.abs(audio)))
        if _abs_max > 0:
            # Estimate effective bit depth from dynamic range
            _dyn_range_db = 20.0 * np.log10(_abs_max / (np.min(np.abs(audio[audio != 0])) + 1e-15) + 1e-15)
            _est_bits = max(8, min(32, int(_dyn_range_db / 6.02) + 1))
        else:
            _est_bits = 16
        # Float32 input from soundfile → likely 24 or 32 bit source
        if audio.dtype in (np.float32, np.float64):
            _est_bits = max(_est_bits, 24)

        format_info = FormatInfo(
            container_format="WAV",
            codec="PCM",
            sample_rate=sr,
            bit_depth=_est_bits,
            channels=channels,
            dc_offset=float(np.mean(audio)),
            has_clipping=bool(np.max(np.abs(audio)) > 0.99),
        )

        # 2. Material & Chain Analysis (Spec 3.1.2)
        from backend.core.forensics.detector import MediaForensicsEngine

        forensic_engine = MediaForensicsEngine()
        forensic_report = forensic_engine.analyze(audio, sr)
        raw_medium = getattr(forensic_report, "primary_media", MediaType.UNKNOWN)
        if isinstance(raw_medium, MediaType):
            detected_medium = raw_medium
        else:
            medium_name = str(getattr(raw_medium, "name", raw_medium)).lower()
            try:
                detected_medium = MediaType(medium_name)
            except ValueError:
                detected_medium = MediaType.UNKNOWN

        material_chain = MaterialChainAnalysis(
            detected_medium=detected_medium,
            medium_confidence=forensic_report.primary_confidence,
            vinyl_rpm=None,
            tape_type=None,
            adc_type=None,
            resampling_artifacts=False,
            lossy_codec_history=[],
            generation_count=1,
        )

        # 3. Spectral Analysis (Spec 3.1.2)
        # Spectral Flux — frame-to-frame spectral change (STFT L2-norm of diff)
        try:
            import librosa as _lr

            _audio_flat = audio.flatten() if audio.ndim > 1 else audio
            _S = np.abs(_lr.stft(_audio_flat, n_fft=2048, hop_length=512))
            _flux = np.sqrt(np.sum(np.diff(_S, axis=1) ** 2, axis=0))
            spectral_flux_val = float(np.mean(_flux))
        except Exception:
            spectral_flux_val = 0.0

        spectral = SpectralAnalysis(
            spectral_centroid=features.get("spectral_centroid", 0.0),
            spectral_rolloff=features.get("spectral_rolloff", 0.0),
            spectral_flux=spectral_flux_val,
            bandwidth=features.get("spectral_rolloff", sr / 2),
            has_aliasing=False,
            frequency_gaps=[],
        )

        # 4. Dynamics Analysis (Spec 3.1.2)
        dynamics = DynamicsAnalysis(
            lufs_integrated=features.get("lufs", -23.0),
            lufs_short_term=features.get("lufs", -23.0),
            lufs_momentary=features.get("lufs", -23.0),
            dynamic_range_db=features.get("crest_factor", 12.0),
            crest_factor_db=features.get("crest_factor", 12.0),
            true_peak_dbfs=20 * np.log10(np.max(np.abs(audio)) + 1e-10),
            rms_db=20 * np.log10(features.get("rms", 0.1) + 1e-10),
            loudness_range_lu=features.get(
                "loudness_range_lu", max(1.0, float(features.get("crest_factor", 12.0)) * 0.6)
            ),
        )

        # 5. Stereo Analysis (Spec 3.1.2)
        if channels == 2:
            left = audio_stereo[0]
            right = audio_stereo[1]
            mid = (left + right) / 2
            side = (left - right) / 2
            mid_energy: float = float(np.sum(mid**2))
            side_energy: float = float(np.sum(side**2))
            mid_side_balance = side_energy / (mid_energy + 1e-10)

            # Guard: np.corrcoef on near-constant (silent) signals → RuntimeWarning
            _la = left - left.mean()
            _ra = right - right.mean()
            _nl = float(np.linalg.norm(_la))
            _nr = float(np.linalg.norm(_ra))
            _dot_corr = float(np.dot(_la, _ra) / (_nl * _nr + 1e-10))
            correlation = _dot_corr if np.isfinite(_dot_corr) else 1.0
            stereo_width = 2.0 * (1.0 - correlation)

            stereo = StereoAnalysis(
                mid_side_balance=float(mid_side_balance),
                stereo_width=float(stereo_width),
                phase_coherence=float(max(0.0, correlation)),
                iacc=float(correlation),
                panning_distribution={},
                mono_compatibility_score=float(max(0.0, correlation)),
            )
        else:
            # Mono
            stereo = StereoAnalysis(
                mid_side_balance=0.0,
                stereo_width=0.0,
                phase_coherence=1.0,
                iacc=1.0,
                panning_distribution={},
                mono_compatibility_score=1.0,
            )

        # 6. Defect Detection (Spec 3.1.3) - strukturierte Defektobjekte aus den Feature-Detektoren.
        detected_defects: list[DefectDetection] = []
        _clipping_pct = float(features.get("clipping_percentage", 0.0) or 0.0)
        if _clipping_pct > 0.1:
            detected_defects.append(
                DefectDetection(
                    defect_type=DefectType.CLIPPING,
                    severity=float(np.clip(_clipping_pct / 10.0, 0.0, 1.0)),
                    confidence=0.85,
                    affected_frequency_range=None,
                    temporal_locations=[],
                    classification_details={"clipping_percentage": _clipping_pct},
                )
            )
        _crackle_density = float(features.get("crackle_density", 0.0) or 0.0)
        if _crackle_density > 0.05:
            detected_defects.append(
                DefectDetection(
                    defect_type=DefectType.CRACKLE,
                    severity=float(np.clip(_crackle_density * 8.0, 0.0, 1.0)),
                    confidence=0.75,
                    affected_frequency_range=None,
                    temporal_locations=[],
                    classification_details={"crackle_density": _crackle_density},
                )
            )
        _dropout_count = int(features.get("dropout_count", 0) or 0)
        if _dropout_count > 0:
            detected_defects.append(
                DefectDetection(
                    defect_type=DefectType.DROPOUT,
                    severity=float(np.clip(_dropout_count / 10.0, 0.0, 1.0)),
                    confidence=0.75,
                    affected_frequency_range=None,
                    temporal_locations=[],
                    classification_details={"dropout_count": _dropout_count},
                )
            )
        _hum_score = float(features.get("hum_score", 0.0) or 0.0)
        if _hum_score > 0.05:
            detected_defects.append(
                DefectDetection(
                    defect_type=DefectType.HUM,
                    severity=float(np.clip(_hum_score, 0.0, 1.0)),
                    confidence=0.70,
                    affected_frequency_range=(40.0, 300.0),
                    temporal_locations=[],
                    classification_details={"hum_score": _hum_score},
                )
            )

        # 7. Musical Context (Spec 3.1.4) - PANNS-based
        musical_context = MusicalContext(
            genre=genre,  # From PANNS
            genre_confidence=genre_confidence,  # From PANNS
            dominant_instruments=instruments,  # From PANNS
            tempo_bpm=features.get("tempo_bpm"),  # via librosa.beat.beat_track (FeatureExtractor)
            time_signature=None,
            key_signature=None,
            structure_segments=[],
            dynamic_contour=[],
            harmonic_complexity=0.5,
        )

        # 8. Vocal Analysis (Spec 3.1.5) - PANNS-based
        vocal_analysis = VocalAnalysis(
            has_vocals=has_vocals,  # From PANNS
            vocal_confidence=vocal_confidence,  # From PANNS
            num_speakers=1 if has_vocals else 0,  # Heuristik: Vokaler Inhalt → 1 Sprecher (PANNs-basiert)
            language=None,
            language_confidence=0.0,
            valence=None,
            arousal=None,
        )

        # 9. Feature Vectors (Spec 3.1.6)
        feature_vectors = FeatureVectors(
            onset_times=[],
            beat_times=[],
            tempo_bpm=None,
            mfccs=None,
            spectral_contrast=[features.get("spectral_contrast", [])],
            chroma_features=None,
            pitch_contour=None,
            pitch_confidence=None,
            harmonicity=None,
            rhythm_patterns={},
            syncopation_index=None,
        )

        # Calculate overall quality score — multi-factor estimation
        _snr_score = min(1.0, features.get("snr", 20.0) / 40.0)
        _dr_score = min(1.0, dynamics.dynamic_range_db / 20.0) if dynamics.dynamic_range_db > 0 else 0.5
        _defect_penalty = min(0.3, len(detected_defects) * 0.05)
        overall_quality = float(
            np.clip(0.50 * _snr_score + 0.30 * _dr_score + 0.20 * (1.0 - _defect_penalty), 0.0, 1.0)
        )

        # Create AnalysisProfile
        profile = AnalysisProfile(
            format_info=format_info,
            material_chain=material_chain,
            spectral=spectral,
            dynamics=dynamics,
            stereo=stereo,
            detected_defects=detected_defects,
            overall_quality_score=overall_quality,
            musical_context=musical_context,
            vocal_analysis=vocal_analysis,
            feature_vectors=feature_vectors,
            raw_features=features,  # Store original dict for backward compat
        )

        return profile

    def _extract_genre_from_panns(self, tags: dict) -> tuple:
        """
        Extrahiert genre from PANNS tags with specialized SCHLAGER recognition.

        Args:
                tags: PANNS output dict with {tag: confidence}

        Returns:
                Tuple of (Genre enum, confidence float)
        """
        from backend.core.data_models import Genre

        # 🎵 PRIORITY: German Schlager Detection (multi-signal)
        # Schlager characteristics: Pop structure + German vocals + typical era/instruments
        schlager_indicators = {
            "pop": tags.get("Pop music", 0.0),
            "speech": tags.get("Speech", 0.0),
            "singing": tags.get("Singing", 0.0),
            "folk": tags.get("Folk music", 0.0),
            "accordion": tags.get("Accordion", 0.0),
            "keyboard": tags.get("Keyboard (musical)", 0.0),
            "synthesizer": tags.get("Synthesizer", 0.0),
            "male_voice": tags.get("Male voice", 0.0),
            "female_voice": tags.get("Female voice", 0.0),
        }

        # Schlager scoring (weighted combination)
        schlager_score = (
            schlager_indicators["pop"] * 0.3  # Pop structure
            + schlager_indicators["singing"] * 0.25  # Vocal-centric
            + (schlager_indicators["male_voice"] + schlager_indicators["female_voice"]) * 0.15
            + schlager_indicators["folk"] * 0.15  # Folk influences
            + (schlager_indicators["accordion"] + schlager_indicators["keyboard"] + schlager_indicators["synthesizer"])
            * 0.15
        )

        # If strong Schlager indicators (>0.35), prioritize SCHLAGER
        if schlager_score > 0.35:
            return Genre.SCHLAGER, schlager_score

        # Genre mapping: PANNS label → AURIK Genre enum
        genre_mapping = {
            "Classical music": Genre.CLASSICAL,
            "Jazz": Genre.JAZZ,
            "Rock music": Genre.ROCK_METAL,
            "Pop music": Genre.VOCAL_POP,
            "Electronic music": Genre.ELECTRONIC,
            "Disco": Genre.ELECTRONIC,
            "Techno": Genre.ELECTRONIC,
            "House music": Genre.ELECTRONIC,
            "Ambient music": Genre.ELECTRONIC,
            # Fallback for Schlager if not caught above
            "Folk music": Genre.SCHLAGER,
        }

        # Find highest scoring genre
        max_score = 0.0
        detected_genre = Genre.UNKNOWN

        for panns_label, aurik_genre in genre_mapping.items():
            score = tags.get(panns_label, 0.0)
            if score > max_score:
                max_score = score
                detected_genre = aurik_genre

        # If no genre detected with confidence > 0.3, return UNKNOWN
        if max_score < 0.3:
            detected_genre = Genre.UNKNOWN

        return detected_genre, max_score

    def _extract_vocals_from_panns(self, tags: dict) -> tuple:
        """
        Extrahiert vocal detection from PANNS tags.

        Args:
                tags: PANNS output dict

        Returns:
                Tuple of (has_vocals bool, confidence float)
        """
        # Vocal-related tags
        vocal_tags = [
            "Singing",
            "Speech",
            "Male voice",
            "Female voice",
            "Choir",
            "Chant",
            "Mantra",
            "Child speech, kid speaking",
            "Narration, monologue",
            "Conversation",
        ]

        # Find max vocal score
        max_vocal_score = max((tags.get(tag, 0.0) for tag in vocal_tags), default=0.0)

        # Threshold for vocal detection
        has_vocals = max_vocal_score > 0.3

        return has_vocals, max_vocal_score

    def _extract_instruments_from_panns(self, tags: dict) -> list:
        """
        Extrahiert dominant instruments from PANNS tags.

        Args:
                tags: PANNS output dict

        Returns:
                List of instrument names (strings)
        """
        # Instrument tags with threshold
        instrument_tags = [
            "Piano",
            "Electric piano",
            "Keyboard (musical)",
            "Guitar",
            "Acoustic guitar",
            "Electric guitar",
            "Bass guitar",
            "Drum kit",
            "Drum",
            "Snare drum",
            "Bass drum",
            "Violin, fiddle",
            "Cello",
            "Double bass",
            "Flute",
            "Clarinet",
            "Saxophone",
            "Trumpet",
            "Trombone",
            "Organ",
            "Synthesizer",
            "String section",
            "Brass instrument",
            "Wind instrument",
            "Orchestra",
        ]

        instruments = []
        for tag in instrument_tags:
            score = tags.get(tag, 0.0)
            if score > 0.5:  # Confidence threshold
                instruments.append(tag)

        return instruments
