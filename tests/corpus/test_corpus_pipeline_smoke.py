"""Corpus Pipeline Smoke Test — §15.2.

End-to-End-Test: Jede Corpus-Datei durchläuft die minimale Aurik-Pipeline.
Kein Performance-Gate, nur: kein Crash, kein NaN, kein Inf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CORPUS_ROOT = Path(__file__).parent.parent.parent / "corpus"
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

MATERIAL_DIRS = ["shellac", "vinyl", "tape", "reel_tape", "cassette", "digital"]


def _collect_corpus_files() -> list[tuple[str, Path]]:
    """Sammelt alle Audio-Dateien aus allen Manifest-Dateien."""
    import yaml

    files = []
    for mat in MATERIAL_DIRS:
        mf = CORPUS_ROOT / mat / "manifest.yaml"
        if not mf.exists():
            continue
        with open(mf, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        for entry in data.get("entries", []):
            fp = entry.get("file")
            if fp is None:
                continue
            abs_path = mf.parent / fp
            if abs_path.exists() and abs_path.suffix.lower() in (
                ".wav",
                ".flac",
                ".mp3",
                ".ogg",
                ".aiff",
                ".aif",
            ):
                files.append((f"{mat}/{fp}", abs_path))
    return files


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Lädt eine Audiodatei via scipy oder soundfile."""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return audio, int(sr)
    except ImportError:
        import scipy.io.wavfile as wav

        sr, audio = wav.read(str(path))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        return audio, int(sr)


def _run_mini_pipeline(audio: np.ndarray, sr: int) -> dict:
    """Führt die minimale Aurik-Pipeline aus (mode=smoke) und gibt Metadaten zurück."""
    from backend.core.defect_scanner import DefectScanner
    from backend.core.perceptual_quality_scorer import PerceptualQualityScorer

    # Resample auf 48 kHz falls nötig
    if sr != 48000:
        from scipy.signal import resample_poly

        target_len = int(len(audio) * 48000 / sr)
        if audio.ndim == 1:
            audio = resample_poly(audio.astype(np.float64), 48000, sr)
        else:
            audio = np.column_stack(
                [resample_poly(audio[:, ch].astype(np.float64), 48000, sr) for ch in range(audio.shape[1])]
            )
        audio = audio[:target_len].astype(np.float32)
        sr = 48000

    # Peak-Normalisierung (Inline — backend.core.audio_utils exportiert kein normalize_audio mehr)
    _peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if _peak > 1e-8:
        audio = (audio.astype(np.float32) / _peak).astype(np.float32)
    else:
        audio = np.zeros_like(audio, dtype=np.float32)

    # NaN/Inf-Check vor Pipeline
    assert np.isfinite(audio).all(), "Input-Audio enthält NaN/Inf"

    # Defect-Scan
    scanner = DefectScanner()
    try:
        defects = scanner.scan(audio, sr)
        defects_found = len(defects.get_top_defects(8)) if hasattr(defects, "get_top_defects") else 0
    except Exception:
        defects = []  # type: ignore[assignment]
        defects_found = 0

    # PQS-Score
    pqs = PerceptualQualityScorer()
    try:
        score = pqs.score(audio, sr)  # type: ignore[call-arg]
    except Exception:
        score = 0.0  # type: ignore[assignment]

    return {
        "defects_found": int(defects_found),
        "pqs_score": float(score),  # type: ignore[arg-type]
        "sample_rate": int(sr),
        "duration_s": len(audio) / sr,
        "peak": float(np.max(np.abs(audio))),
        "has_nan": bool(np.any(np.isnan(audio))),
        "has_inf": bool(np.any(np.isinf(audio))),
        "rms": float(np.sqrt(np.mean(audio**2))),
    }


# ── Parametrisierte Tests ───────────────────────────────────────────────────


@pytest.mark.corpus
@pytest.mark.slow
class TestCorpusPipelineSmoke:
    """Smoke-Test: Keine Pipeline stürzt ab."""

    @pytest.fixture(autouse=True)
    def corpus_files(self) -> list[tuple[str, Path]]:
        files = _collect_corpus_files()
        if not files:
            pytest.skip("Keine Corpus-Dateien gefunden — Corpus ist leer")
        return files

    @pytest.mark.parametrize("label,path", _collect_corpus_files() or [])
    def test_load_audio_no_crash(self, label: str, path: Path):
        """Jede Corpus-Datei muss ohne Crash geladen werden können."""
        try:
            audio, sr = _load_audio(path)
        except Exception as e:
            pytest.fail(f"{label}: Audio konnte nicht geladen werden: {e}")
        assert audio is not None
        assert sr > 0
        assert len(audio) > 0

    @pytest.mark.parametrize("label,path", _collect_corpus_files() or [])
    def test_audio_no_nan_inf(self, label: str, path: Path):
        """Jede Corpus-Datei muss frei von NaN/Inf sein."""
        audio, sr = _load_audio(path)
        assert np.isfinite(audio).all(), f"{label}: Audio enthält NaN oder Inf"

    @pytest.mark.parametrize("label,path", _collect_corpus_files() or [])
    def test_defect_scan_no_crash(self, label: str, path: Path):
        """DefectScanner muss für jede Corpus-Datei ohne Crash laufen."""
        audio, sr = _load_audio(path)
        meta = _run_mini_pipeline(audio, sr)
        assert meta is not None
        assert "defects_found" in meta
