"""
Aurik 10.0.0 — IntroducedArtifactDetector (IAD) §2.23
==================================================
Erkennt durch Restaurierung neu eingebrachte Artefakte im restaurierten Audio.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArtifactRegion:
    """Zeitbereich mit erkanntem Artefakt."""

    artifact_type: str
    start_sample: int
    end_sample: int
    severity: float
    confidence: float
    description: str = ""


@dataclass
class IADResult:
    """Ergebnis des IntroducedArtifactDetectors."""

    has_artifacts: bool
    artifacts: list[ArtifactRegion] = field(default_factory=list)
    n_ml_hallucinations: int = 0
    n_nmf_clicks: int = 0
    n_pvoc_smearing: int = 0
    n_musical_noise: int = 0
    artifact_mask: np.ndarray | None = None
    total_contaminated_fraction: float = 0.0
    confidence: float = 1.0

    @property
    def artifact_types(self) -> list[str]:
        return sorted({a.artifact_type for a in self.artifacts}) if self.artifacts else []


# Rückwärtskompatibel
IADRegion = ArtifactRegion


class IntroducedArtifactDetector:
    """Erkennt durch Restaurierung neu eingebrachte Artefakte (§2.23)."""

    IAD_ARTIFACT_TYPES: list[str] = [
        "ml_hallucination",
        "nmf_residual_click",
        "phase_vocoder_smearing",
        "musical_noise",
    ]

    CLICK_THRESHOLD_DB: float = 12.0
    CLICK_MAX_DURATION_MS: float = 5.0
    PVOC_SMEAR_THRESHOLD_MS: float = (
        50.0  # Fallback floor; _detect_pvoc_smearing uses signal-adaptive threshold (see below)
    )
    MUSICAL_NOISE_THRESHOLD_DB: float = 3.0
    SILENCE_THRESHOLD_DBFS: float = -40.0
    HARMONICITY_THRESHOLD: float = 0.70
    # §2.46e Relative-Harmonicity-Guard (v10.0.0):
    # Residuum-Harmonizität muss HALLUCINATION_RELATIVE_MARGIN über Original-Harmonizität liegen.
    # Verhindert False-Positives: Restaurierungs-Delta auf vokalen/harmonischen Content erbt
    # die Harmonizität des Originals → kein Hallucination-Flag.
    HALLUCINATION_RELATIVE_MARGIN: float = 0.20
    # Min. Residuum-RMS für echte ML-Halluzination (≈ −30 dBFS):
    # Unterhalb dieser Schwelle ist das Residuum zu leise für eine wahrnehmbare Halluzination.
    HALLUCINATION_MIN_RMS: float = 0.032
    # Runtime guards for long real-audio tracks (UAT): keep detector bounded.
    MAX_HALLUCINATION_WINDOWS: int = 48
    HALLUCINATION_DETECT_BUDGET_S: float = 8.0

    @staticmethod
    def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
        """Konvertiert input to finite float32 safely without overflow warnings."""
        arr64 = np.asarray(audio, dtype=np.float64)
        arr64 = np.nan_to_num(arr64, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(arr64, -1.0, 1.0).astype(np.float32, copy=False)  # type: ignore[no-any-return]

    def scan(self, audio: np.ndarray, sr: int) -> IADResult:
        """Single-audio Artefakt-Erkennung (ohne Original-Referenz).

        Wird vom Defect Consensus Pipeline aufgerufen, wenn nur das restaurierte
        Signal verfügbar ist. Nutzt heuristische Mustererkennung für typische
        Verarbeitungsartefakte im restaurierten Audio.

        Args:
            audio: Restauriertes Audiosignal (mono oder stereo).
            sr: Abtastrate in Hz.

        Returns:
            IADResult mit erkannten Artefakten und Kontaminations-Fraktion.
        """
        audio = self._sanitize_audio(audio)
        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        n_samples = len(mono)
        if n_samples == 0:
            return IADResult(has_artifacts=False, confidence=1.0)

        artifact_mask = np.zeros(n_samples, dtype=bool)
        artifacts: list[ArtifactRegion] = []

        # 1. Click-Erkennung im restaurierten Signal (energetische Spitzen)
        click_len = max(1, int(self.CLICK_MAX_DURATION_MS / 1000.0 * sr))
        kernel = np.ones(click_len) / click_len
        energy = np.sqrt(np.convolve(mono**2, kernel, mode="same") + 1e-12)
        global_rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
        if global_rms > 1e-10:
            click_threshold = global_rms * 10.0 ** (self.CLICK_THRESHOLD_DB / 20.0)
            click_mask = energy > click_threshold
            in_click = False
            start = 0
            for i, v in enumerate(click_mask):
                if v and not in_click:
                    in_click = True
                    start = i
                elif not v and in_click:
                    in_click = False
                    if (i - start) <= click_len * 2:
                        sev = float(np.clip(float(np.max(energy[start:i])) / click_threshold, 0.0, 1.0))
                        artifacts.append(
                            ArtifactRegion(
                                "nmf_residual_click",
                                max(0, start - click_len),
                                min(n_samples, i + click_len),
                                sev,
                                0.75,
                            )
                        )
                        artifact_mask[max(0, start - click_len) : min(n_samples, i + click_len)] = True

        # 2. Musical Noise: tonaler Content in Stille-Bereichen
        win = max(1, int(0.10 * sr))
        hop = max(1, int(0.05 * sr))
        for s in range(0, n_samples - win, hop):
            e = s + win
            db = 20.0 * np.log10(max(float(np.sqrt(np.mean(mono[s:e] ** 2))), 1e-10))
            if db < self.SILENCE_THRESHOLD_DBFS:
                # Prüfe auf tonalen Content (Harmonizität) in Stille
                h = self._harmonicity(mono[s:e], sr)
                if h > self.HARMONICITY_THRESHOLD:
                    sev = float(np.clip(h / 1.0, 0.0, 1.0))
                    artifacts.append(ArtifactRegion("musical_noise", s, e, sev, 0.70))
                    artifact_mask[s:e] = True

        # 3. Phase-Vocoder Smearing: temporale Inkonsistenzen im restaurierten Signal
        # Nutzt RMS-Gradient-Analyse auf dem Mono-Signal allein
        pvoc_hop = 512
        rms_frames = np.array(
            [float(np.sqrt(np.mean(mono[i : i + pvoc_hop] ** 2))) for i in range(0, n_samples - pvoc_hop, pvoc_hop)]
        )
        if len(rms_frames) >= 3:
            diff = np.diff(rms_frames)
            max_diff = float(np.max(np.abs(diff)) + 1e-12)
            onset_threshold = max(0.05 * max_diff, 1e-6)
            in_smear = False
            smear_start = 0
            for i, d in enumerate(diff):
                if d > onset_threshold and not in_smear:
                    in_smear = True
                    smear_start = i
                elif d <= onset_threshold and in_smear:
                    duration_samples = (i - smear_start) * pvoc_hop
                    if duration_samples > max(1, int(self.PVOC_SMEAR_THRESHOLD_MS / 1000.0 * sr)):
                        sev = float(np.clip(duration_samples / max(int(self.PVOC_SMEAR_THRESHOLD_MS / 50.0 * sr), 1), 0.0, 1.0))
                        artifacts.append(
                            ArtifactRegion(
                                "phase_vocoder_smearing",
                                smear_start * pvoc_hop,
                                min(n_samples, i * pvoc_hop + pvoc_hop),
                                sev,
                                0.65,
                            )
                        )
                        artifact_mask[smear_start * pvoc_hop : min(n_samples, i * pvoc_hop + pvoc_hop)] = True
                    in_smear = False

        frac = float(np.sum(artifact_mask)) / n_samples if n_samples > 0 else 0.0
        return IADResult(
            has_artifacts=len(artifacts) > 0,
            artifacts=artifacts,
            n_ml_hallucinations=sum(1 for a in artifacts if a.artifact_type == "ml_hallucination"),
            n_nmf_clicks=sum(1 for a in artifacts if a.artifact_type == "nmf_residual_click"),
            n_pvoc_smearing=sum(1 for a in artifacts if a.artifact_type == "phase_vocoder_smearing"),
            n_musical_noise=sum(1 for a in artifacts if a.artifact_type == "musical_noise"),
            artifact_mask=artifact_mask,
            total_contaminated_fraction=frac,
            confidence=float(np.clip(1.0 - frac, 0.0, 1.0)),
        )

    def detect(self, original: np.ndarray, restored: np.ndarray, sr: int) -> IADResult:
        """Erkennt durch Restaurierung eingebrachte Artefakte."""
        original = self._sanitize_audio(original)
        restored = self._sanitize_audio(restored)
        orig_mono = original if original.ndim == 1 else original.mean(axis=0)
        rest_mono = restored if restored.ndim == 1 else restored.mean(axis=0)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len == 0:
            return IADResult(has_artifacts=False, confidence=1.0)
        orig_mono = orig_mono[:min_len]
        rest_mono = rest_mono[:min_len]
        n_samples = min_len
        artifact_mask = np.zeros(n_samples, dtype=bool)
        artifacts: list[ArtifactRegion] = []
        residuum = np.nan_to_num(rest_mono - orig_mono)

        for a in self._detect_nmf_clicks(orig_mono, residuum, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True
        for a in self._detect_musical_noise(orig_mono, residuum, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True
        for a in self._detect_ml_hallucinations(orig_mono, residuum, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True
        for a in self._detect_pvoc_smearing(orig_mono, rest_mono, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True

        frac = float(np.sum(artifact_mask)) / n_samples
        return IADResult(
            has_artifacts=len(artifacts) > 0,
            artifacts=artifacts,
            n_ml_hallucinations=sum(1 for a in artifacts if a.artifact_type == "ml_hallucination"),
            n_nmf_clicks=sum(1 for a in artifacts if a.artifact_type == "nmf_residual_click"),
            n_pvoc_smearing=sum(1 for a in artifacts if a.artifact_type == "phase_vocoder_smearing"),
            n_musical_noise=sum(1 for a in artifacts if a.artifact_type == "musical_noise"),
            artifact_mask=artifact_mask,
            total_contaminated_fraction=frac,
            confidence=float(np.clip(1.0 - frac, 0.0, 1.0)),
        )

    def get_artifact_mask(self, iad_result: IADResult, n_samples: int) -> np.ndarray:
        if iad_result.artifact_mask is None:
            return np.zeros(n_samples, dtype=bool)  # type: ignore[no-any-return]
        mask = iad_result.artifact_mask
        if len(mask) < n_samples:
            return np.pad(mask, (0, n_samples - len(mask)))  # type: ignore[no-any-return]
        return mask[:n_samples]

    def _detect_nmf_clicks(self, orig: np.ndarray, residuum: np.ndarray, sr: int) -> list[ArtifactRegion]:
        click_len = max(1, int(self.CLICK_MAX_DURATION_MS / 1000.0 * sr))
        kernel = np.ones(click_len) / click_len
        energy_orig = np.sqrt(np.convolve(orig**2, kernel, mode="same") + 1e-12)
        energy_res = np.sqrt(np.convolve(residuum**2, kernel, mode="same") + 1e-12)
        threshold_ratio = 10.0 ** (self.CLICK_THRESHOLD_DB / 20.0)
        click_mask = (energy_res / energy_orig) > threshold_ratio
        artifacts: list[ArtifactRegion] = []
        in_click = False
        start = 0
        for i, v in enumerate(click_mask):
            if v and not in_click:
                in_click = True
                start = i
            elif not v and in_click:
                in_click = False
                if (i - start) <= click_len * 2:
                    ratio = energy_res[start:i] / energy_orig[start:i]
                    sev = float(np.clip(float(np.max(ratio)) / threshold_ratio, 0.0, 1.0))
                    artifacts.append(
                        ArtifactRegion(
                            "nmf_residual_click", max(0, start - click_len), min(len(orig), i + click_len), sev, 0.75
                        )
                    )
        return artifacts

    def _detect_musical_noise(self, orig: np.ndarray, residuum: np.ndarray, sr: int) -> list[ArtifactRegion]:
        win = max(1, int(0.10 * sr))
        hop = max(1, int(0.05 * sr))
        artifacts: list[ArtifactRegion] = []
        for s in range(0, len(orig) - win, hop):
            e = s + win
            db_o = 20.0 * np.log10(max(float(np.sqrt(np.mean(orig[s:e] ** 2))), 1e-10))
            if db_o < self.SILENCE_THRESHOLD_DBFS:
                db_r = 20.0 * np.log10(max(float(np.sqrt(np.mean(residuum[s:e] ** 2))), 1e-10))
                if db_r > db_o + self.MUSICAL_NOISE_THRESHOLD_DB:
                    sev = float(np.clip((db_r - db_o) / 20.0, 0.0, 1.0))
                    artifacts.append(ArtifactRegion("musical_noise", s, e, sev, 0.70))
        return artifacts

    def _detect_ml_hallucinations(self, orig: np.ndarray, residuum: np.ndarray, sr: int) -> list[ArtifactRegion]:
        """§2.46e Relative-Harmonicity-Guard (v10.0.0).

        Detects truly hallucinated harmonic content introduced by ML restoration.
        The residuum (restored − original) of any vocal/musical restoration inherits
        harmonic structure from the underlying music signal — this is NOT a hallucination.
        A true hallucination requires the residuum to be SIGNIFICANTLY MORE harmonic
        than the original at the same location (new harmonic content was added).

        Gate: h_res > HARMONICITY_THRESHOLD AND h_res > h_orig + HALLUCINATION_RELATIVE_MARGIN.
        Effective threshold: max(0.70, h_orig + 0.20) — scales with original harmonicity.
        - Vocal (h_orig ≈ 0.80): effective threshold ≈ 1.00 → never fires on vocal restoration ✓
        - Noise section (h_orig ≈ 0.10): effective threshold ≈ 0.30 → fires at 0.70 (original gate) ✓
        """
        win = max(1, int(2.0 * sr))
        hop = max(1, int(1.0 * sr))
        artifacts: list[ArtifactRegion] = []
        starts = list(range(0, len(residuum) - win, hop))
        if len(starts) > self.MAX_HALLUCINATION_WINDOWS:
            stride = max(1, int(np.ceil(len(starts) / float(self.MAX_HALLUCINATION_WINDOWS))))
            starts = starts[::stride]

        _t0 = time.perf_counter()
        for s in starts:
            if (time.perf_counter() - _t0) > self.HALLUCINATION_DETECT_BUDGET_S:
                logger.warning(
                    "IAD hallucination guard: time Grenze %.1fs exceeded after %d windows — early stop",
                    self.HALLUCINATION_DETECT_BUDGET_S,
                    len(artifacts),
                )
                break
            frame = residuum[s : s + win]
            # Energie-Gate: Residuum muss wahrnehmbar sein (>= HALLUCINATION_MIN_RMS)
            # Verhindert False-Positives durch numerisches Rauschen bei leichten Phasen-änderungen
            frame_rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
            if frame_rms < self.HALLUCINATION_MIN_RMS:
                continue
            # Lightweight decimation keeps harmonicity robust while reducing FFT cost.
            if len(frame) >= 8192 and sr >= 32000:
                frame_eval = frame[::2]
                sr_eval = sr // 2
            else:
                frame_eval = frame
                sr_eval = sr

            h_res = self._harmonicity(frame_eval, sr_eval)
            if h_res > self.HARMONICITY_THRESHOLD:
                # §2.46e Relative-Harmonicity-Guard:
                # Only flag if residuum is significantly MORE harmonic than the original.
                # Musical restoration modifies existing harmonics → residuum inherits their
                # harmonicity → NOT a hallucination (artifact_freedom = 0.95 adhesion fix).
                orig_frame = orig[s : s + win]
                orig_eval = orig_frame[::2] if (len(orig_frame) >= 8192 and sr >= 32000) else orig_frame
                h_orig = self._harmonicity(orig_eval, sr_eval)
                if h_res <= h_orig + self.HALLUCINATION_RELATIVE_MARGIN:
                    continue  # Restoration delta on existing harmonic content — not a hallucination
                artifacts.append(
                    ArtifactRegion("ml_hallucination", s, s + win, float(np.clip(h_res, 0.0, 1.0)), float(h_res))
                )
        return artifacts

    def _detect_pvoc_smearing(self, orig: np.ndarray, rest: np.ndarray, sr: int) -> list[ArtifactRegion]:
        hop = 512

        # Signal-adaptive smearing threshold (§2.49 adaptive intelligence):
        # Estimate transient density from original and set threshold accordingly.
        # High transient density (percussive) → tighter threshold (smearing more audible).
        # Low transient density (sustained) → relaxed threshold (avoids false positives).
        # Floor: PVOC_SMEAR_THRESHOLD_MS (50 ms) for very low-event / noisy material.
        try:
            _frame_len = max(2, int(0.05 * sr))  # 50 ms energy frames
            _hop_td = max(1, _frame_len // 2)
            _energies = np.array(
                [
                    float(np.sqrt(np.mean(orig[i : i + _frame_len] ** 2) + 1e-12))
                    for i in range(0, max(1, len(orig) - _frame_len), _hop_td)
                ],
                dtype=np.float32,
            )
            if len(_energies) >= 2:
                _diff = np.diff(_energies)
                _transient_frames: int = int(np.sum(_diff > 0.20 * float(np.max(_energies) + 1e-12)))
                _duration_s = max(len(orig) / sr, 1e-3)
                _transient_density = float(_transient_frames) / _duration_s  # events/s
            else:
                _transient_density = 0.0
        except Exception:
            _transient_density = 0.0

        # Threshold mapping: percussive → 15 ms, rhythmic → 25 ms, sustained → 35 ms,
        # low-event/noisy → 50 ms floor.
        if _transient_density > 3.0:
            _smear_ms = 15.0  # percussion / drums
        elif _transient_density > 1.5:
            _smear_ms = 25.0  # rhythmic / mixed
        elif _transient_density > 0.5:
            _smear_ms = 35.0  # sustained melodic
        else:
            _smear_ms = self.PVOC_SMEAR_THRESHOLD_MS  # 50 ms floor for near-static/noise

        # Dense transient material produces weaker per-frame RMS slopes because
        # event energy is distributed across many short events. Use adaptive onset
        # gating to keep smear detection sensitive in percussive contexts while
        # remaining conservative for sparse/sustained material.
        if _transient_density > 3.0:
            _onset_threshold = 0.08
        elif _transient_density > 1.5:
            _onset_threshold = 0.12
        else:
            _onset_threshold = 0.20
        smear = max(1, int(_smear_ms / 1000.0 * sr))

        def frames(sig: np.ndarray) -> np.ndarray:
            return np.array([float(np.sqrt(np.mean(sig[i : i + hop] ** 2))) for i in range(0, len(sig) - hop, hop)])  # type: ignore[no-any-return]

        eo = frames(orig)
        er = frames(rest)
        n = min(len(eo), len(er))
        if n < 2:
            return []
        do = np.diff(eo[:n])
        dr = np.diff(er[:n])
        artifacts: list[ArtifactRegion] = []
        for i in range(1, min(n - 1, len(do))):
            orig_idx = i - 1
            if do[orig_idx] > _onset_threshold:
                bj, bv = i, -1.0
                for j in range(
                    max(0, i - 6), min(n - 1, i + 6)
                ):  # was ±20 frames; narrowed to prevent false cross-matching with unrelated transients
                    if j < len(dr) and dr[j] > bv:
                        bv = dr[j]
                        bj = j
                delay_samples = abs(bj - orig_idx) * hop
                if delay_samples > smear:
                    sev = float(np.clip(delay_samples / max(smear * 5, 1), 0.0, 1.0))
                    artifacts.append(
                        ArtifactRegion(
                            "phase_vocoder_smearing",
                            orig_idx * hop,
                            min(len(orig), (orig_idx + 5) * hop),
                            sev,
                            0.65,
                        )
                    )
        return artifacts

    def _harmonicity(self, frame: np.ndarray, sr: int) -> float:
        if len(frame) < sr // 10:
            return 0.0
        n = len(frame)
        nf = 1
        while nf < 2 * n:
            nf <<= 1
        F = np.fft.rfft(frame, n=nf)
        ac = np.fft.irfft(F * np.conj(F), n=nf)[:n]
        if ac[0] < 1e-10:
            return 0.0
        acn = ac / ac[0]
        lo = max(1, int(sr / 255))
        hi = min(n - 1, int(sr / 85))
        if lo >= hi:
            return 0.0
        return float(np.clip(np.nan_to_num(float(np.max(acn[lo:hi]))), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: IntroducedArtifactDetector | None = None
_lock = threading.Lock()


def get_iad() -> IntroducedArtifactDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = IntroducedArtifactDetector()
    return _instance


get_introduced_artifact_detector = get_iad


def detect_introduced_artifacts(original: np.ndarray, restored: np.ndarray, sr: int) -> IADResult:
    """Convenience-Wrapper für Artefakt-Erkennung."""
    if sr != 48000:
        raise ValueError(f"SR muss 48000 Hz sein, erhalten: {sr}")
    return get_iad().detect(original, restored, sr)


def scan_introduced_artifacts(audio: np.ndarray, sr: int) -> IADResult:
    """Convenience-Wrapper für Single-Audio Artefakt-Erkennung (ohne Original)."""
    if sr != 48000:
        raise ValueError(f"SR muss 48000 Hz sein, erhalten: {sr}")
    return get_iad().scan(audio, sr)


__all__ = [
    "ArtifactRegion",
    "IADRegion",
    "IADResult",
    "IntroducedArtifactDetector",
    "detect_introduced_artifacts",
    "scan_introduced_artifacts",
    "get_iad",
    "get_introduced_artifact_detector",
]
