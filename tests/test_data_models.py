"""
Test suite for core/data_models.py - Pydantic models
Tests AudioFile, Genre, MediaType, DefectType
"""

import os
import sys
from datetime import datetime

import pytest

pytest.importorskip("pydantic")  # CI-Minimal-Umgebung (cross-platform)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.data_models import AudioFile, DefectType, Genre, MediaType


@pytest.mark.unit
def test_audio_file_basic():
    """Test AudioFile model with valid data"""
    audio = AudioFile(
        file_path="/test/audio.wav",
        file_hash="abc123def456",
        format="WAV",
        sample_rate=48000,
        bit_depth=24,
        channels=2,
        duration=120.0,
        file_size=5184000,
    )

    assert audio.file_path == "/test/audio.wav"
    assert audio.sample_rate == 48000
    assert audio.channels == 2
    assert audio.duration == 120.0


def test_audio_file_mono():
    """Test AudioFile with mono audio"""
    audio = AudioFile(
        file_path="/test/mono.wav",
        file_hash="mono123",
        format="WAV",
        sample_rate=44100,
        bit_depth=16,
        channels=1,
        duration=60.0,
        file_size=5292000,
    )

    assert audio.channels == 1
    assert audio.bit_depth == 16


def test_audio_file_created_at_auto():
    """Test AudioFile auto-generates created_at timestamp"""
    audio = AudioFile(
        file_path="/test/audio.wav",
        file_hash="hash",
        format="WAV",
        sample_rate=48000,
        bit_depth=24,
        channels=2,
        duration=60.0,
        file_size=1000,
    )

    # created_at should be auto-generated
    assert audio.created_at is not None
    assert isinstance(audio.created_at, datetime)


def test_audio_file_invalid_sample_rate():
    """Test AudioFile rejects invalid sample rate"""
    with pytest.raises(Exception):  # Pydantic ValidationError
        AudioFile(
            file_path="/test/audio.wav",
            file_hash="hash",
            format="WAV",
            sample_rate=0,  # Invalid: must be > 0
            bit_depth=24,
            channels=2,
            duration=60.0,
            file_size=1000,
        )


def test_audio_file_invalid_channels():
    """Test AudioFile rejects invalid channel count"""
    with pytest.raises(Exception):  # Pydantic ValidationError
        AudioFile(
            file_path="/test/audio.wav",
            file_hash="hash",
            format="WAV",
            sample_rate=48000,
            bit_depth=24,
            channels=0,  # Invalid: must be >= 1
            duration=60.0,
            file_size=1000,
        )


def test_audio_file_invalid_duration():
    """Test AudioFile rejects negative duration"""
    with pytest.raises(Exception):  # Pydantic ValidationError
        AudioFile(
            file_path="/test/audio.wav",
            file_hash="hash",
            format="WAV",
            sample_rate=48000,
            bit_depth=24,
            channels=2,
            duration=-10.0,  # Invalid: must be >= 0
            file_size=1000,
        )


def test_audio_file_json_serialization():
    """Test AudioFile can be serialized to JSON"""
    audio = AudioFile(
        file_path="/test/audio.wav",
        file_hash="hash",
        format="WAV",
        sample_rate=48000,
        bit_depth=24,
        channels=2,
        duration=60.0,
        file_size=1000,
    )

    json_data = audio.model_dump()

    assert json_data["file_path"] == "/test/audio.wav"
    assert json_data["sample_rate"] == 48000
    assert "created_at" in json_data


def test_genre_enum_values():
    """Test Genre enum has expected values"""
    assert Genre.CLASSICAL == "classical"
    assert Genre.JAZZ == "jazz"
    assert Genre.ROCK_METAL == "rock_metal"
    assert Genre.ELECTRONIC == "electronic"
    assert Genre.VOCAL_POP == "vocal_pop"
    assert Genre.VINTAGE_ANALOG == "vintage_analog"
    assert Genre.UNKNOWN == "unknown"


def test_genre_enum_iteration():
    """Test Genre enum can be iterated"""
    genres = list(Genre)
    assert len(genres) == 8  # CLASSICAL, JAZZ, ROCK_METAL, ELECTRONIC, VOCAL_POP, SCHLAGER, VINTAGE_ANALOG, UNKNOWN
    assert Genre.CLASSICAL in genres
    assert Genre.JAZZ in genres


def test_media_type_enum_values():
    """Test MediaType enum has expected values"""
    assert MediaType.VINYL == "vinyl"
    assert MediaType.TAPE == "tape"
    assert MediaType.CASSETTE == "cassette"
    assert MediaType.CD == "cd"
    assert MediaType.DIGITAL_NATIVE == "digital_native"
    assert MediaType.RADIO_BROADCAST == "radio_broadcast"
    assert MediaType.UNKNOWN == "unknown"


def test_media_type_enum_iteration():
    """Test MediaType enum can be iterated"""
    media_types = list(MediaType)
    assert len(media_types) == 7
    assert MediaType.VINYL in media_types


def test_defect_type_enum_values():
    """Test DefectType enum has expected values (kanonisch: defect_scanner)."""
    # Kern-Defekte vorhanden
    assert DefectType.CLICKS.value == "clicks"
    assert DefectType.CRACKLE.value == "crackle"
    assert DefectType.HUM.value == "hum"
    assert DefectType.WOW.value == "wow"
    assert DefectType.FLUTTER.value == "flutter"
    assert DefectType.CLIPPING.value == "clipping"
    assert DefectType.DROPOUTS.value == "dropouts"


def test_defect_type_enum_iteration():
    """Test DefectType enum can be iterated (54+ Defekttypen)."""
    defect_types = list(DefectType)
    assert len(defect_types) >= 54, f"Erwartete ≥54 Defekttypen, aber {len(defect_types)} vorhanden."
    assert DefectType.HUM in defect_types
    assert DefectType.CLIPPING in defect_types


def test_audio_file_high_sample_rate():
    """Test AudioFile with high sample rate (192kHz)"""
    audio = AudioFile(
        file_path="/test/hires.wav",
        file_hash="hires123",
        format="WAV",
        sample_rate=192000,
        bit_depth=32,
        channels=2,
        duration=60.0,
        file_size=46080000,
    )

    assert audio.sample_rate == 192000
    assert audio.bit_depth == 32


def test_audio_file_multi_channel_metadata_only():
    """
    Test AudioFile data model with multi-channel metadata (5.1 surround)

    NOTE: This tests the DATA MODEL only (metadata storage).
    Aurik 10.0.0 does NOT support multichannel PROCESSING.
    This test validates that metadata for multichannel files can be stored,
    but such files will be REJECTED by UnifiedRestorerV2.restore().
    """
    audio = AudioFile(
        file_path="/test/surround.wav",
        file_hash="surround123",
        format="WAV",
        sample_rate=48000,
        bit_depth=24,
        channels=6,  # 5.1 surround (metadata only - processing not supported)
        duration=120.0,
        file_size=103680000,
    )

    assert audio.channels == 6  # Metadata storage works
    # But: UnifiedRestorerV2.restore() would reject this (see test_unified_restorer.py)


def test_audio_file_flac_format():
    """Test AudioFile with FLAC format"""
    audio = AudioFile(
        file_path="/test/audio.flac",
        file_hash="flac123",
        format="FLAC",
        sample_rate=96000,
        bit_depth=24,
        channels=2,
        duration=180.0,
        file_size=83160000,
    )

    assert audio.format == "FLAC"
    assert audio.sample_rate == 96000
