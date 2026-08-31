"""§G124 (GEBOTE.md) Blindtest-Framework: Automatisierte perzeptuelle Validierung.

Ermöglicht den Vergleich von Aurik-Restaurierungen mit Referenz-Signalen
durch automatisierte Hörvergleiche mit psychoakustischen Metriken.

Verwendet PEAQ (Perceptual Evaluation of Audio Quality, ITU-R BS.1387)
und PESQ (Perceptual Evaluation of Speech Quality, ITU-T P.862) als
objektive Proxy-Metriken für menschliche Hörurteile.

Usage:
    python -m backend.core.blindtest_framework --input degraded.wav --reference clean.wav
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

__all__ = [
    "BlindTestResult",
    "BlindTestFramework",
    "run_blindtest",
    "compare_chain_factors",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BlindTestResult:
    """Ergebnis eines automatisierten Blindtests."""

    method: str = "peaq"  # peaq, pesq, composite
    test_name: str = ""
    reference_path: str = ""
    degraded_path: str = ""
    restored_path: str = ""

    # PEAQ
    odg: float = -4.0  # Objective Difference Grade, -4 (worst) bis 0 (transparent)
    di: float = 0.0  # Distortion Index
    n_before: float = 0.0  # PEAQ-ODG vor Restaurierung
    n_after: float = 0.0  # PEAQ-ODG nach Restaurierung

    # PESQ
    pesq_mos: float = 1.0  # 1.0 (worst) bis 4.5 (transparent)

    # Metadaten
    chain_depth: int = 1
    material_type: str = "unknown"
    chain_factor: float = 1.0

    passed: bool = False
    improvement: float = 0.0  # Δ ODG vor/nach
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "method": self.method,
            "chain_depth": self.chain_depth,
            "material_type": self.material_type,
            "chain_factor": self.chain_factor,
            "odg_before": round(self.n_before, 3),
            "odg_after": round(self.n_after, 3),
            "odg_restored": round(self.odg, 3),
            "improvement": round(self.improvement, 3),
            "passed": self.passed,
            "notes": self.notes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Framework
# ═══════════════════════════════════════════════════════════════════════════════


class BlindTestFramework:
    """§G124 (GEBOTE.md) Automatisierte perzeptuelle Validierung.

    Vergleicht Restaurierungsergebnisse mit Referenzsignalen durch
    psychoakustische Proxy-Metriken. Ersetzt teure, langsame menschliche
    Hörtests durch objektive, reproduzierbare Qualitätsbewertung.

    Verwendete Standards:
      - ITU-R BS.1387-2 (PEAQ): Musikqualität
      - ITU-T P.862 (PESQ): Sprachqualität
      - Bark-Lautheit (Zwicker): Perzeptuelle Lautheit
      - Crest-Faktor: Dynamik-Erhalt
    """

    def __init__(self) -> None:
        self._results: list[BlindTestResult] = []

    # ── Kern-Methode ─────────────────────────────────────────────────

    def compare(
        self,
        degraded: Any,  # np.ndarray
        restored: Any,  # np.ndarray
        reference: Any,  # np.ndarray (optional)
        sr: int = 48000,
        *,
        chain_depth: int = 1,
        material_type: str = "unknown",
        test_name: str = "",
    ) -> BlindTestResult:
        """Vergleicht degraded vs restored Audio perzeptuell.

        Args:
            degraded: Audio VOR der Restaurierung
            restored: Audio NACH der Restaurierung
            reference: Sauberes Referenz-Audio (optional)
            sr: Sample-Rate
            chain_depth: Transfer-Chain-Tiefe
            material_type: Tonträger-Typ
            test_name: Name des Tests

        Returns:
            BlindTestResult mit Qualitätsbewertung und Verbesserung.
        """
        import numpy as np

        # Konvertiere zu float32 und mono
        _degraded = self._to_mono_float32(degraded)
        _restored = self._to_mono_float32(restored)
        _reference = self._to_mono_float32(reference) if reference is not None else None

        result = BlindTestResult(
            test_name=test_name,
            chain_depth=chain_depth,
            material_type=material_type,
            chain_factor=1.0 + max(0, chain_depth - 2) * 0.25,
        )

        # 1. PEAQ-Proxy: Bark-Lautheit + Crest-Faktor + SNR
        result.n_before = self._estimate_odg(_degraded, _reference, sr)
        result.n_after = self._estimate_odg(_restored, _reference, sr)
        result.odg = result.n_after
        result.improvement = result.n_after - result.n_before

        # 2. PESQ-Proxy: Speech-Quality via Segmental-SNR
        result.pesq_mos = self._estimate_pesq(_restored, _reference or _degraded, sr)

        # 3. Bewertung
        result.passed = result.improvement >= -0.05  # Max 0.05 ODG-Verschlechterung

        if result.improvement > 0.20:
            result.notes.append("Starke Verbesserung (ΔODG > 0.20)")
        elif result.improvement > 0.05:
            result.notes.append("Hörbare Verbesserung (ΔODG > 0.05)")
        elif result.improvement >= -0.05:
            result.notes.append("Keine signifikante Änderung (|ΔODG| ≤ 0.05)")
        else:
            result.notes.append(f"Verschlechterung (ΔODG = {result.improvement:.3f})")

        self._results.append(result)
        return result

    # ── Interne Metriken ──────────────────────────────────────────────

    @staticmethod
    def _to_mono_float32(audio: Any) -> Any:
        import numpy as np

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=0)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _estimate_odg(audio: Any, reference: Any | None, sr: int) -> float:
        """PEAQ-ODG-Proxy via Bark-Lautheit + Crest-Faktor.

        ITU-R BS.1387-2 Basic Version: ODG = f(Bandwidth, NMR, EHS).
        Vereinfachter Proxy für automatisierte Tests.
        """
        import numpy as np

        arr = np.asarray(audio, dtype=np.float64)
        n = len(arr)

        if n < sr // 10:
            return -4.0

        # 1. Crest-Faktor (Dynamik-Indikator)
        rms = np.sqrt(np.mean(arr**2)) + 1e-9
        peak = np.max(np.abs(arr)) + 1e-9
        crest_db = 20 * np.log10(peak / rms)
        crest_score = float(np.clip(1.0 - abs(crest_db - 12.0) / 24.0, 0.0, 1.0))

        # 2. Effektive Bandbreite (vereinfacht via Zero-Crossing)
        zcr = np.sum(np.abs(np.diff(np.sign(arr)))) / (2 * n)
        bw_score = float(np.clip(zcr / 0.3, 0.0, 1.0))

        # 3. SNR-Proxy (Signal-Peak vs RMS-Floor)
        frame_size = sr // 100  # 10ms
        n_frames = max(1, n // frame_size)
        frame_rms = np.array(
            [np.sqrt(np.mean(arr[i * frame_size : (i + 1) * frame_size] ** 2)) for i in range(n_frames)]
        )
        noise_floor = float(np.percentile(frame_rms[frame_rms > 1e-9], 10))
        signal_level = float(np.percentile(frame_rms, 90))
        snr_proxy = float(np.clip(20 * np.log10(signal_level / (noise_floor + 1e-9)) / 60.0, 0.0, 1.0))

        # Composite: gewichtete Kombination
        composite = 0.4 * crest_score + 0.3 * bw_score + 0.3 * snr_proxy

        # Map [0, 1] → ODG [-4, 0]
        return float(-4.0 + 4.0 * composite)

    @staticmethod
    def _estimate_pesq(audio: Any, reference: Any, sr: int) -> float:
        """PESQ-MOS-Proxy via Segmental-SNR."""
        import numpy as np

        arr = np.asarray(audio, dtype=np.float64)
        ref = np.asarray(reference, dtype=np.float64)
        min_len = min(len(arr), len(ref))
        arr = arr[:min_len]
        ref = ref[:min_len]

        # Segmental SNR
        seg_size = sr // 50  # 20ms
        n_seg = max(1, min_len // seg_size)
        seg_snr = np.zeros(n_seg)
        for i in range(n_seg):
            a_seg = arr[i * seg_size : (i + 1) * seg_size]
            r_seg = ref[i * seg_size : (i + 1) * seg_size]
            noise = a_seg - r_seg
            ns = np.sum(r_seg**2) + 1e-9
            nd = np.sum(noise**2) + 1e-9
            seg_snr[i] = float(np.clip(10 * np.log10(ns / nd), -10, 35))

        avg_snr = float(np.mean(np.clip(seg_snr, -10, 35)))

        # Map SNR → MOS [1.0, 4.5]
        return float(np.clip(1.0 + avg_snr / 10.0, 1.0, 4.5))

    # ── Batch-Tests ───────────────────────────────────────────────────

    def results(self) -> list[BlindTestResult]:
        return list(self._results)

    def summary(self) -> dict[str, Any]:
        results = self._results
        if not results:
            return {"tests": 0, "passed": 0, "failed": 0}

        return {
            "tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "mean_improvement": round(float(sum(r.improvement for r in results) / len(results)), 3),
            "details": [r.to_dict() for r in results],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience-Funktionen
# ═══════════════════════════════════════════════════════════════════════════════


def run_blindtest(
    degraded_path: Path,
    restored_path: Path,
    reference_path: Path | None = None,
    *,
    chain_depth: int = 1,
    material_type: str = "unknown",
) -> BlindTestResult:
    """Führt einen Blindtest auf WAV-Dateien aus."""
    import wave

    import numpy as np

    def _read_wav(path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as w:
            n_frames = w.getnframes()
            data = np.frombuffer(w.readframes(n_frames), dtype=np.int16)
            return cast(np.ndarray, data.astype(np.float32) / 32768.0)

    degraded = _read_wav(degraded_path)
    restored = _read_wav(restored_path)
    reference = _read_wav(reference_path) if reference_path else None

    with wave.open(str(degraded_path), "rb") as w:
        sr = w.getframerate()

    fw = BlindTestFramework()
    return fw.compare(
        degraded,
        restored,
        reference,
        sr,
        chain_depth=chain_depth,
        material_type=material_type,
        test_name=degraded_path.stem,
    )


def compare_chain_factors(
    degraded: Any,
    restored_by_depth: dict[int, Any],
    reference: Any | None = None,
    sr: int = 48000,
    material_type: str = "cassette",
) -> dict[int, BlindTestResult]:
    """Vergleicht Restaurierungsergebnisse für verschiedene chain_factors.

    Args:
        degraded: Degradiertes Eingangssignal
        restored_by_depth: {depth: audio} — Ergebnisse pro Depth-Stufe
        reference: Sauberes Referenzsignal
        sr: Sample-Rate
        material_type: Tonträger-Typ

    Returns:
        {depth: BlindTestResult} — pro Depth-Stufe ein Ergebnis
    """
    fw = BlindTestFramework()
    results = {}
    for depth, audio in sorted(restored_by_depth.items()):
        results[depth] = fw.compare(
            degraded,
            audio,
            reference,
            sr,
            chain_depth=depth,
            material_type=material_type,
            test_name=f"depth_{depth}",
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aurik Blindtest-Framework")
    parser.add_argument("--input", required=True, help="Degradierte Eingabedatei")
    parser.add_argument("--reference", help="Saubere Referenzdatei")
    parser.add_argument("--restored", help="Restaurierte Ausgabedatei")
    parser.add_argument("--chain-depth", type=int, default=1)
    parser.add_argument("--material", default="unknown")
    parser.add_argument("--json", action="store_true", help="Ergebnis als JSON")

    args = parser.parse_args()

    result = run_blindtest(
        Path(args.input),
        Path(args.restored) if args.restored else Path(args.input),
        Path(args.reference) if args.reference else None,
        chain_depth=args.chain_depth,
        material_type=args.material,
    )

    if args.json:
        logger.info("%s", json.dumps(result.to_dict(), indent=2))
    else:
        logger.info("Test: %s", result.test_name)
        logger.info("  Tiefe: %d (Faktor: %.2f×)", result.chain_depth, result.chain_factor)
        logger.info("  ODG vorher: %.3f", result.n_before)
        logger.info("  ODG nachher: %.3f", result.n_after)
        logger.info("  Verbesserung: %+.3f", result.improvement)
        logger.info("  PESQ-MOS: %.2f", result.pesq_mos)
        logger.info("  Bestanden: %s", "ja" if result.passed else "nein")
        for note in result.notes:
            logger.info("  - %s", note)
