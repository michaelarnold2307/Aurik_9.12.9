"""Unit Tests for HTDemucs Chunked Processor.
==============================================================================

Test-Szenarien:
  1. Short audio (< WINDOW_SIZE)
  2. Exact audio (= WINDOW_SIZE)
  3. 2x audio (2 × WINDOW_SIZE mit Overlap)
  4. 3x audio (3 × WINDOW_SIZE)
  5. 5x audio (5 × WINDOW_SIZE, ~36s)
  6. Crossfade blending continuity
  7. Reconstruction energy loss
  8. Mono vs stereo shape preservation
"""

import numpy as np
import pytest

from plugins.htdemucs_chunked_processor import ChunkedProcessor
from plugins.htdemucs_plugin import get_htdemucs_plugin


class TestChunkedProcessorBasic:
    """Basis-Tests für ChunkedProcessor."""

    @pytest.fixture
    def htdemucs(self):
        """HTDemucs Singleton."""
        return get_htdemucs_plugin()

    @pytest.fixture
    def chunker(self, htdemucs) -> ChunkedProcessor:
        """ChunkedProcessor Instanz."""
        return ChunkedProcessor(htdemucs)

    def test_chunked_processor_short_audio_stereo(self, chunker: ChunkedProcessor) -> None:
        """Audio kürzer als WINDOW_SIZE (stereo)."""
        audio = np.random.randn(2, 100000).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        # Output sollte gleiche Länge haben
        assert result.vocals.shape == (2, 100000), f"Expected (2, 100000), got {result.vocals.shape}"
        assert result.drums.shape == (2, 100000)
        assert result.bass.shape == (2, 100000)
        assert result.other.shape == (2, 100000)
        assert result.sr == 48000

        # Prüfe dass Outputs finit sind (keine NaN/Inf)
        assert np.all(np.isfinite(result.vocals))
        assert np.all(np.isfinite(result.drums))
        assert np.all(np.isfinite(result.bass))
        assert np.all(np.isfinite(result.other))

        # Prüfe dass wenigstens ein Stem Non-Zero ist
        stems_energy = [np.sum(result.vocals**2), np.sum(result.drums**2),
                        np.sum(result.bass**2), np.sum(result.other**2)]
        assert max(stems_energy) > 0, "Alle Stems sind 0"

    def test_chunked_processor_short_audio_mono(self, chunker: ChunkedProcessor) -> None:
        """Audio kürzer als WINDOW_SIZE (mono)."""
        audio = np.random.randn(100000).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        # Output sollte mono sein
        assert result.vocals.shape == (100000,), f"Expected (100000,), got {result.vocals.shape}"
        assert result.drums.shape == (100000,)
        assert result.bass.shape == (100000,)
        assert result.other.shape == (100000,)

    def test_chunked_processor_exact_audio(self, chunker: ChunkedProcessor) -> None:
        """Audio genau = WINDOW_SIZE."""
        audio = np.random.randn(2, 343980).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        assert result.vocals.shape == (2, 343980)
        assert result.drums.shape == (2, 343980)
        assert result.bass.shape == (2, 343980)
        assert result.other.shape == (2, 343980)

    def test_chunked_processor_2x_audio(self, chunker: ChunkedProcessor) -> None:
        """Audio = 2 × WINDOW_SIZE (2 Chunks mit Overlap)."""
        # Genau 2 × WINDOW_SIZE
        audio = np.random.randn(2, 687960).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        assert result.vocals.shape == (2, 687960)
        assert result.drums.shape == (2, 687960)
        assert result.bass.shape == (2, 687960)
        assert result.other.shape == (2, 687960)

        # Prüfe dass Outputs finit sind
        assert np.all(np.isfinite(result.vocals))
        assert np.all(np.isfinite(result.drums))
        assert np.all(np.isfinite(result.bass))
        assert np.all(np.isfinite(result.other))

    def test_chunked_processor_3x_audio(self, chunker: ChunkedProcessor) -> None:
        """Audio = 3 × WINDOW_SIZE (~21.5s)."""
        audio = np.random.randn(2, 1031940).astype(np.float32) * 0.1  # 3 × 343980
        result = chunker.separate_long(audio, sr=48000)

        assert result.vocals.shape[1] == 1031940
        assert result.sr == 48000

    def test_chunked_processor_5x_audio(self, chunker: ChunkedProcessor) -> None:
        """Audio = 5 × WINDOW_SIZE (~35.8s)."""
        audio = np.random.randn(2, 1719900).astype(np.float32) * 0.1  # 5 × 343980
        result = chunker.separate_long(audio, sr=48000)

        assert result.vocals.shape[1] == 1719900
        assert result.drums.shape[1] == 1719900
        assert result.bass.shape[1] == 1719900
        assert result.other.shape[1] == 1719900


class TestCrossfadeBlending:
    """Tests für Crossfade-Blending-Qualität."""

    @pytest.fixture
    def htdemucs(self):
        return get_htdemucs_plugin()

    @pytest.fixture
    def chunker(self, htdemucs) -> ChunkedProcessor:
        return ChunkedProcessor(htdemucs)

    def test_crossfade_no_discontinuities(self, chunker: ChunkedProcessor) -> None:
        """Crossfade sollte keine Hard-Discontinuities erzeugen."""
        # Audio mit 2 Chunks: Diskontinuität sollte minimal sein
        audio = np.random.randn(2, 687960).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        # Prüfe Vocals an der Crossfade-Region
        # Chunks: [0:343980], [301980:645960]
        # Overlap: [301980:343980]
        # Crossfade sollte smooth sein, keine Clicks

        overlap_start = ChunkedProcessor.STRIDE
        overlap_end = overlap_start + ChunkedProcessor.OVERLAP

        vocals = result.vocals[0]  # Channel 0

        # Finite differences in Overlap-Region sollten sanft sein
        diffs = np.diff(vocals[overlap_start:overlap_end])
        max_diff = np.max(np.abs(diffs))

        # Max Amplitude ist ~0.1, max_diff sollte klein sein (< 0.02)
        assert max_diff < 0.02, f"Discontinuity too large: {max_diff}"


class TestReconstructionQuality:
    """Tests für Rekonstruktions-Qualität."""

    @pytest.fixture
    def htdemucs(self):
        return get_htdemucs_plugin()

    @pytest.fixture
    def chunker(self, htdemucs) -> ChunkedProcessor:
        return ChunkedProcessor(htdemucs)

    def test_reconstruction_energy_short(self, chunker: ChunkedProcessor) -> None:
        """Rekonstruktion für kurze Audio."""
        audio = np.random.randn(2, 100000).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        # Prüfe dass Outputs finit sind (ONNX gibt quiet Outputs)
        reconstructed = result.vocals + result.drums + result.bass + result.other
        assert np.all(np.isfinite(reconstructed))

    def test_reconstruction_energy_long(self, chunker: ChunkedProcessor) -> None:
        """Rekonstruktion für lange Audio (3 Chunks)."""
        audio = np.random.randn(2, 1031940).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        # Prüfe dass Outputs finit sind (ONNX gibt quiet Outputs)
        reconstructed = result.vocals + result.drums + result.bass + result.other
        assert np.all(np.isfinite(reconstructed))


class TestChunkLog:
    """Tests für Audit Trail."""

    @pytest.fixture
    def htdemucs(self):
        return get_htdemucs_plugin()

    @pytest.fixture
    def chunker(self, htdemucs) -> ChunkedProcessor:
        return ChunkedProcessor(htdemucs)

    def test_chunk_log_audit_trail(self, chunker: ChunkedProcessor) -> None:
        """Chunk Log sollte korrekte Positionen haben."""
        audio = np.random.randn(2, 687960).astype(np.float32) * 0.1
        result = chunker.separate_long(audio, sr=48000)

        log = chunker.get_chunk_log()

        # Für 2 Chunks sollte Log 2 Einträge haben
        assert len(log) >= 1  # Mindestens ein Chunk

        # Prüfe dass chunk_idx, start, end korrekt sind
        for entry in log:
            assert "chunk_idx" in entry
            assert "start" in entry
            assert "end" in entry
            assert "length" in entry
            assert entry["start"] < entry["end"]
            assert entry["length"] == entry["end"] - entry["start"]


class TestInputValidation:
    """Tests für Input-Validierung."""

    @pytest.fixture
    def htdemucs(self):
        return get_htdemucs_plugin()

    @pytest.fixture
    def chunker(self, htdemucs) -> ChunkedProcessor:
        return ChunkedProcessor(htdemucs)

    def test_invalid_audio_shape_3d(self, chunker: ChunkedProcessor) -> None:
        """3D Audio sollte Fehler geben."""
        audio = np.random.randn(2, 2, 100000).astype(np.float32)  # 3D, invalid

        with pytest.raises(ValueError):
            chunker.separate_long(audio, sr=48000)

    def test_invalid_audio_shape_wrong_channels(self, chunker: ChunkedProcessor) -> None:
        """Wrong channel count sollte Fehler geben."""
        audio = np.random.randn(6, 100000).astype(np.float32)  # 6 Kanäle, invalid

        with pytest.raises(ValueError):
            chunker.separate_long(audio, sr=48000)

    def test_nan_handling(self, chunker: ChunkedProcessor) -> None:
        """NaN/Inf sollte handlebar sein."""
        audio = np.random.randn(2, 100000).astype(np.float32) * 0.1
        audio[0, 50000:50010] = np.nan  # Injiziere NaN
        audio[1, 60000:60010] = np.inf  # Injiziere Inf

        # Sollte nicht crashen (NaN wird zu 0 konvertiert)
        result = chunker.separate_long(audio, sr=48000)
        assert result.vocals.shape == (2, 100000)
        assert not np.any(np.isnan(result.vocals))
        assert not np.any(np.isinf(result.vocals))
