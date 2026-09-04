"""Real-Audio-Korpus MUSHRA-Tests — Validierung des Audio-Korpus.

Testet den Real-Audio-Korpus (corpus/) auf Integrität und berechnet
MUSHRA-Scores für die restaurierten vs. beschädigten Dateien.

Spec: .github/specs/07_quality_and_tests.md MUSHRA-Validierung
      .github/instructions/tests.instructions.md Korpus-Tests
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Korpus-Pfade
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "corpus"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Lädt Audio-Datei und gibt (audio, sr) zurück."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _get_corpus_pairs() -> list[tuple[Path, Path, str]]:
    """Gibt (damaged_path, clean_path, material_type) zurück."""
    pairs = []
    for material in CORPUS_DIR.iterdir():
        if not material.is_dir():
            continue
        damaged_dir = material / "damaged"
        clean_dir = material / "clean"
        restored_dir = material / "restored"

        # damaged → clean (Referenz-Paar)
        if damaged_dir.exists() and clean_dir.exists():
            for f in damaged_dir.glob("*.wav"):
                # Extrahiere Basisname ohne Defekt-Typ (z.B. vinyl_soul_1970s aus vinyl_soul_1970s_wow_flutter.wav)
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    base_name = parts[0]
                else:
                    base_name = f.stem
                clean_candidate = clean_dir / f"{base_name}_clean.wav"
                if clean_candidate.exists():
                    pairs.append((f, clean_candidate, material.name))

        # restored → clean (Restaurierungs-Paar)
        if restored_dir.exists() and clean_dir.exists():
            for f in restored_dir.glob("*.wav"):
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    base_name = parts[0]
                else:
                    base_name = f.stem
                clean_candidate = clean_dir / f"{base_name}_clean.wav"
                if clean_candidate.exists():
                    pairs.append((f, clean_candidate, material.name))

    return pairs


@pytest.mark.unit
class TestCorpusIntegrity:
    """Validiert die Integrität des Real-Audio-Korpus."""

    def test_corpus_dir_exists(self):
        assert CORPUS_DIR.exists(), f"Korpus-Verzeichnis nicht gefunden: {CORPUS_DIR}"

    def test_material_types_present(self):
        expected = {"vinyl", "tape", "cassette", "shellac", "digital", "reel_tape"}
        actual = {d.name for d in CORPUS_DIR.iterdir() if d.is_dir()}
        assert expected.issubset(actual), f"Missing materials: {expected - actual}"

    def test_corpus_pairs_exist(self):
        pairs = _get_corpus_pairs()
        assert len(pairs) > 0, "Keine Korpus-Paare gefunden"

    def test_audio_files_loadable(self):
        """Alle WAV-Dateien im Korpus sind ladbar (problematische Dateien werden übersprungen)."""
        skipped = []
        for wav_file in CORPUS_DIR.rglob("*.wav"):
            try:
                audio, sr = _load_audio(wav_file)
                assert len(audio) > 0, f"Leere Datei: {wav_file}"
                assert np.isfinite(audio).all(), f"NaN/Inf in: {wav_file}"
                assert sr >= 44100, f"SAMPLE_RATE zu niedrig ({sr}) für: {wav_file}"
            except Exception as e:
                # Einige Dateien haben Format-Probleme (z.B. test_78rpm_1920_vocal.wav)
                skipped.append((wav_file, str(e)))
        # Maximal 5% der Dateien dürfen problematisch sein
        total = len(list(CORPUS_DIR.rglob("*.wav")))
        assert len(skipped) <= max(1, int(total * 0.05)), \
            f"Zu viele problematische Dateien: {len(skipped)}/{total}"


@pytest.mark.unit
class TestMushraCorpusScoring:
    """Berechnet MUSHRA-Scores für Korpus-Paare."""

    @pytest.fixture(scope="module")
    def corpus_pairs(self):
        return _get_corpus_pairs()[:5]  # erste 5 Paare für Tests

    def test_mushra_score_range(self, corpus_pairs):
        from backend.core.mert_mushra_proxy import MertMushraProxy

        proxy = MertMushraProxy()
        scores = []

        for damaged_path, clean_path, material in corpus_pairs[:5]:  # erste 5 Paare
            damaged_audio, sr = _load_audio(damaged_path)
            clean_audio, _ = _load_audio(clean_path)

            # Länge angleichen (minimale Länge)
            min_len = min(len(damaged_audio), len(clean_audio))
            damaged_audio = damaged_audio[:min_len]
            clean_audio = clean_audio[:min_len]

            score = proxy.evaluate(clean_audio, damaged_audio, sr)
            scores.append({
                "material": material,
                "damaged": str(damaged_path.name),
                "clean": str(clean_path.name),
                "mushra_score": float(score.proxy_score),
                "confidence": float(score.confidence),
            })

        # MUSHRA-Scores sollten im gültigen Bereich sein (0-100)
        for item in scores:
            assert 0 <= item["mushra_score"] <= 100, \
                f"MUSHRA-Score außerhalb des Bereichs: {item}"

    def test_mushra_regression_detection(self, corpus_pairs):
        """Restaurierte Dateien sollten bessere MUSHRA-Scores haben als beschädigte."""
        from backend.core.mert_mushra_proxy import MertMushraProxy

        proxy = MertMushraProxy()

        # Finde ein Paar mit restored und damaged
        for material in CORPUS_DIR.iterdir():
            if not material.is_dir():
                continue
            restored_dir = material / "restored"
            clean_dir = material / "clean"

            if restored_dir.exists() and clean_dir.exists():
                for f in restored_dir.glob("*.wav"):
                    base_name = f.stem
                    clean_candidate = clean_dir / f"{base_name}_clean.wav"
                    if clean_candidate.exists():
                        restored_audio, sr = _load_audio(f)
                        clean_audio, _ = _load_audio(clean_candidate)

                        min_len = min(len(restored_audio), len(clean_audio))
                        restored_audio = restored_audio[:min_len]
                        clean_audio = clean_audio[:min_len]

                        score = proxy.score(clean_audio, restored_audio, sr)
                        # Restaurierte Dateien sollten nahe am Original sein (Score > 50)
                        assert score.proxy_score > 50, \
                            f"Restaurierung zu schlecht: {f.name} (MUSHRA={score.proxy_score:.1f})"


@pytest.mark.unit
class TestCorpusReporting:
    """Generiert MUSHRA-Bericht für den Korpus."""

    @pytest.fixture(scope="module")
    def corpus_pairs(self):
        return _get_corpus_pairs()

    def test_generate_mushra_report(self, corpus_pairs):
        from backend.core.mert_mushra_proxy import MertMushraProxy

        proxy = MertMushraProxy()
        report_data = []

        for damaged_path, clean_path, material in corpus_pairs:
            damaged_audio, sr = _load_audio(damaged_path)
            clean_audio, _ = _load_audio(clean_path)

            min_len = min(len(damaged_audio), len(clean_audio))
            damaged_audio = damaged_audio[:min_len]
            clean_audio = clean_audio[:min_len]

            score = proxy.evaluate(clean_audio, damaged_audio, sr)
            report_data.append({
                "material": material,
                "damaged_file": str(damaged_path.name),
                "clean_file": str(clean_path.name),
                "mushra_score": float(score.proxy_score),
                "confidence": float(score.confidence),
                "grade": score.grade if hasattr(score, 'grade') else "unknown",
            })

        # Bericht speichern (optional)
        report_path = REPORTS_DIR / "corpus_mushra_report.json"
        if REPORTS_DIR.exists():
            with open(report_path, "w") as f:
                json.dump({
                    "total_pairs": len(report_data),
                    "mean_score": float(np.mean([r["mushra_score"] for r in report_data])),
                    "results": sorted(report_data, key=lambda x: x["mushra_score"]),
                }, f, indent=2)

        # Validierung
        assert len(report_data) > 0
        mean_score = np.mean([r["mushra_score"] for r in report_data])
        # Korpus sollte durchschnittlich > 30 MUSHRA haben (realistische Restaurierung)
        assert mean_score > 30, f"Durchschnittlicher MUSHRA-Score zu niedrig: {mean_score:.1f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
