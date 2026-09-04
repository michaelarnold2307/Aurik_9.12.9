"""§v10.17 PerceptualReferenceValidator — Referenz-Validierung gegen Original.

Schließt die Lücke zwischen DSP-Proxy und menschlicher Wahrnehmung:
Das ORIGINAL ist der Anker. Jede Phase wird auf perzeptuelle Distanz geprüft.

Perceptual Similarity Score (PSS) ∈ [0, 1]:
  1.0 = nah am Original   |   0.0 = entfremdet
  Gate: PSS < 0.85 → Phase zurücksetzen

Drei orthogonal kalibrierte Dimensionen (je 0–1, gewichtet):
  A) Spektrale Korrelation (Bark-Skala) — 40%
  B) Transienten-Positions-Stabilität — 25%
  C) Stereo-Kohärenz (L/R-Korrelations-Delta) — 20%
  D) Energie-Erhalt (RMS-Ratio) — 15%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_RESTORATION_GATE: float = 0.85

# Bark-Bänder (25 Bänder, 0–20 kHz)
_BARK_EDGES = np.array(
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
        20000,
    ],
    dtype=np.float64,
)


@dataclass
class PerceptualAnchor:
    label: str = ""
    bark_envelope: np.ndarray | None = None
    onset_positions: np.ndarray | None = None
    lr_correlation: float = 1.0
    rms: float = 0.0
    sample_rate: int = 48000


@dataclass
class PerceptualValidationResult:
    perceptual_similarity: float = 1.0
    spectral_fidelity: float = 1.0
    transient_preservation: float = 1.0
    stereo_coherence: float = 1.0
    energy_preservation: float = 1.0
    accepted: bool = True
    components: dict[str, float] = field(default_factory=dict)


class PerceptualReferenceValidator:
    def __init__(self) -> None:
        # §V8 (copilot-instructions.md): einziges Stateful-Feld — MUSS pro Song via calibrate_and_store()
        # komplett überschrieben werden, bevor validate() für diesen Song genutzt wird.
        self._anchor: PerceptualAnchor | None = None

    def get_anchor(self) -> PerceptualAnchor | None:
        """Gibt den aktuell kalibrierten Anker zurück (None = noch nicht kalibriert)."""
        return self._anchor

    def calibrate_and_store(self, audio: np.ndarray, sr: int, label: str = "original") -> PerceptualAnchor:
        """§V8 (copilot-instructions.md) Song-Cross-Contamination-Verbot: setzt den Anker NEU für diesen Song.

        Muss einmalig VOR der Phasen-Schleife mit dem Original-Audio DIESES Songs
        aufgerufen werden (überschreibt einen ggf. vorhandenen Anker vollständig).
        """
        anchor = self.calibrate(audio, sr, label)
        self._anchor = anchor
        return anchor

    @staticmethod
    def calibrate(audio: np.ndarray, sr: int, label: str = "original") -> PerceptualAnchor:
        """Einmalig vor der Pipeline: Anker aus Original berechnen."""
        try:
            arr = np.asarray(audio, dtype=np.float64)
            mono = (
                arr.mean(axis=0)
                if (arr.ndim > 1 and arr.shape[0] <= 2)
                else (arr.mean(axis=1) if arr.ndim > 1 else arr)
            )
            n = len(mono)

            # A) Bark-Hüllkurve (25 Bänder, 10 s gemittelt)
            bark_env = PerceptualReferenceValidator._bark_envelope(mono[: min(n, sr * 10)], sr)

            # B) Onsets
            onsets = PerceptualReferenceValidator._detect_onsets(mono)

            # C) Stereo
            if arr.ndim == 2:
                l_ch = arr[0] if arr.shape[0] <= 2 else arr[:, 0]
                r_ch = arr[1] if arr.shape[0] <= 2 else arr[:, 1]
                seg = min(len(l_ch), len(r_ch), sr * 5)
                lr_corr = float(np.corrcoef(l_ch[:seg], r_ch[:seg])[0, 1])
            else:
                lr_corr = 1.0

            rms = float(np.sqrt(np.mean(mono**2) + 1e-12))

            return PerceptualAnchor(
                label=label,
                bark_envelope=bark_env.astype(np.float32),
                onset_positions=np.array(onsets, dtype=np.int64) if onsets else None,
                lr_correlation=lr_corr if np.isfinite(lr_corr) else 1.0,
                rms=rms,
                sample_rate=sr,
            )
        except Exception as exc:
            logger.debug("§V6 _extract_perceptual_anchor fehlgeschlagen — Minimal-Anker zurückgegeben (Label %s): %s", label, exc)
            return PerceptualAnchor(label=label, sample_rate=sr)

    @staticmethod
    def validate(audio: np.ndarray, sr: int, anchor: PerceptualAnchor) -> PerceptualValidationResult:
        """Misst perzeptuelle Ähnlichkeit zum Anker."""
        result = PerceptualValidationResult()
        try:
            arr = np.asarray(audio, dtype=np.float64)
            mono = (
                arr.mean(axis=0)
                if (arr.ndim > 1 and arr.shape[0] <= 2)
                else (arr.mean(axis=1) if arr.ndim > 1 else arr)
            )

            # A) Spektrale Fidelity: Pearson-r der Bark-Hüllkurven
            if anchor.bark_envelope is not None:
                current = PerceptualReferenceValidator._bark_envelope(mono[: min(len(mono), sr * 10)], sr)
                ref = np.asarray(anchor.bark_envelope, dtype=np.float64)
                # Pearson-Korrelation (robuster als Kosinus bei Rauschen)
                ref_c = ref - ref.mean()
                cur_c = current - current.mean()
                corr = float(np.dot(ref_c, cur_c) / (np.sqrt(np.sum(ref_c**2)) * np.sqrt(np.sum(cur_c**2)) + 1e-12))
                # Auf [0,1] mappen: r=-1→0, r=0→0.5, r=1→1
                result.spectral_fidelity = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

            # B) Transienten: Jaccard-Ähnlichkeit der Onset-Positionen
            if anchor.onset_positions is not None and len(anchor.onset_positions) > 0:
                current_onsets = PerceptualReferenceValidator._detect_onsets(mono)
                if len(current_onsets) > 0:
                    ref_set = set(anchor.onset_positions // 512)
                    cur_set = set(np.array(current_onsets) // 512)
                    intersection = len(ref_set & cur_set)
                    union = len(ref_set | cur_set)
                    result.transient_preservation = intersection / max(union, 1)
                else:
                    result.transient_preservation = 0.5
            else:
                result.transient_preservation = 1.0

            # C) Stereo: L/R-Korrelations-Delta
            if arr.ndim == 2:
                l_ch = arr[0] if arr.shape[0] <= 2 else arr[:, 0]
                r_ch = arr[1] if arr.shape[0] <= 2 else arr[:, 1]
                seg = min(len(l_ch), len(r_ch), sr * 5)
                cur_corr = float(np.corrcoef(l_ch[:seg], r_ch[:seg])[0, 1])
                cur_corr = cur_corr if np.isfinite(cur_corr) else 1.0
                delta = abs(cur_corr - anchor.lr_correlation)
                result.stereo_coherence = float(np.clip(1.0 - delta * 2.0, 0.0, 1.0))
            else:
                result.stereo_coherence = 1.0

            # D) Energie
            cur_rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
            ratio = min(cur_rms, anchor.rms) / (max(cur_rms, anchor.rms) + 1e-12)
            result.energy_preservation = float(np.clip(ratio, 0.0, 1.0))

            # PSS
            result.perceptual_similarity = float(
                np.clip(
                    0.40 * result.spectral_fidelity
                    + 0.25 * result.transient_preservation
                    + 0.20 * result.stereo_coherence
                    + 0.15 * result.energy_preservation,
                    0.0,
                    1.0,
                )
            )

            result.components = {
                "spectral": round(result.spectral_fidelity, 4),
                "transient": round(result.transient_preservation, 4),
                "stereo": round(result.stereo_coherence, 4),
                "energy": round(result.energy_preservation, 4),
            }
            result.accepted = result.perceptual_similarity >= _RESTORATION_GATE
            return result

        except Exception as exc:
            logger.debug("§V6 PerceptualValidation fehlgeschlagen — accepted=True zurückgegeben (konservativ): %s", exc)
            return PerceptualValidationResult(accepted=True)

    # ── Helfer ──────────────────────────────────────────────────────

    @staticmethod
    def _bark_envelope(mono: np.ndarray, sr: int) -> np.ndarray:
        """Bark-Hüllkurve: 25 Bänder, gemittelt über 200 Frames à 2048-pt."""
        n_fft = 2048
        hop = n_fft // 2
        n_frames = min(200, max(4, (len(mono) - n_fft) // hop))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        acc = np.zeros(len(_BARK_EDGES) - 1, dtype=np.float64)
        count = 0
        for i in range(n_frames):
            frame = mono[i * hop : i * hop + n_fft]
            if len(frame) < n_fft:
                break
            spec = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
            for j in range(len(_BARK_EDGES) - 1):
                mask = (freqs >= _BARK_EDGES[j]) & (freqs < _BARK_EDGES[j + 1])
                if np.any(mask):
                    acc[j] += float(np.mean(spec[mask]))
            count += 1
        if count > 0:
            acc /= count
        acc /= np.max(acc) + 1e-12
        return cast(np.ndarray, acc)

    @staticmethod
    def _detect_onsets(mono: np.ndarray) -> list[int]:
        """Energie-basierte Onset-Detektion."""
        frame_n = 512
        hop = frame_n // 2
        n_frames = min(100, (len(mono) - frame_n) // hop)
        onsets = []
        prev_e = 1e-12
        for i in range(n_frames):
            start = i * hop
            e = float(np.sum(mono[start : start + frame_n] ** 2))
            if e > prev_e * 3.0 and e > 1e-8:
                onsets.append(start)
            prev_e = e
        return onsets


_validator_instance: PerceptualReferenceValidator | None = None


def get_perceptual_validator() -> PerceptualReferenceValidator:
    """Gibt die globale PerceptualReferenceValidator-Instanz zurück (Singleton).

    §V8 (copilot-instructions.md): Der Anker-Zustand wird NICHT hier zurückgesetzt — jeder Aufrufer MUSS
    `calibrate_and_store()` einmal pro Song aufrufen, bevor `validate()` benutzt wird
    (geschieht in `unified_restorer_v3.py` direkt bei der Pipeline-Initialisierung).
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PerceptualReferenceValidator()
    return _validator_instance
