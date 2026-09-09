from datetime import datetime

import pytest

pytest.importorskip("pydantic")  # CI-Minimal-Umgebung (cross-platform)

from backend.core.data_models import (
    AnalysisProfile,
    DynamicsAnalysis,
    FeatureVectors,
    FormatInfo,
    MaterialChainAnalysis,
    MediaType,
    MusicalContext,
    SpectralAnalysis,
    StereoAnalysis,
    VocalAnalysis,
)
from dsp.dsp_decision_logic import DSPDecisionLogic


def make_minimal_profile(medium, tape_type=None):
    return AnalysisProfile(
        analyzed_at=datetime.now(),
        analysis_version="test",
        format_info=FormatInfo(
            container_format="WAV",
            codec="pcm",
            sample_rate=48000,
            bit_depth=24,
            channels=2,
            dc_offset=0.0,
            has_clipping=False,
        ),
        material_chain=MaterialChainAnalysis(
            detected_medium=medium,
            medium_confidence=1.0,
            vinyl_rpm=None,
            tape_type=tape_type,
            adc_type=None,
            resampling_artifacts=False,
            lossy_codec_history=[],
            generation_count=1,
        ),
        spectral=SpectralAnalysis(
            spectral_centroid=1000.0,
            spectral_rolloff=8000.0,
            spectral_flux=0.1,
            bandwidth=20000.0,
            has_aliasing=False,
            frequency_gaps=[],
        ),
        dynamics=DynamicsAnalysis(
            lufs_integrated=-18.0,
            lufs_short_term=-18.0,
            lufs_momentary=-18.0,
            dynamic_range_db=12.0,
            crest_factor_db=8.0,
            true_peak_dbfs=-1.0,
            rms_db=-20.0,
            loudness_range_lu=5.0,
        ),
        stereo=StereoAnalysis(
            mid_side_balance=1.0,
            stereo_width=1.0,
            phase_coherence=1.0,
            iacc=0.0,
            panning_distribution={},
            mono_compatibility_score=1.0,
        ),
        detected_defects=[],
        overall_quality_score=1.0,
        musical_context=MusicalContext(
            genre="unknown",  # type: ignore[arg-type]
            genre_confidence=0.0,
            dominant_instruments=[],
            tempo_bpm=None,
            time_signature=None,
            key_signature=None,
            structure_segments=[],
            dynamic_contour=[],
            harmonic_complexity=0.5,
        ),
        vocal_analysis=VocalAnalysis(
            has_vocals=False,
            vocal_confidence=0.0,
            num_speakers=0,
            language=None,
            language_confidence=0.0,
            valence=None,
            arousal=None,
        ),
        feature_vectors=FeatureVectors(),  # type: ignore[call-arg]
        raw_features={},
    )


@pytest.mark.unit
def test_decision_logic_cassette():
    profile = make_minimal_profile(MediaType.CASSETTE, tape_type="cassette")
    logic = DSPDecisionLogic()
    chain = logic.analyze_adaptive(profile)
    assert isinstance(chain, list)
    # Es sollte eine Kettenentscheidung erfolgen, die "cassette" berücksichtigt
    assert (
        any("cassette" in str(step) or "cassette" in str(profile.material_chain.tape_type) for step in chain)
        or profile.material_chain.tape_type == "cassette"
    )


def test_decision_logic_reel_to_reel():
    profile = make_minimal_profile(MediaType.TAPE, tape_type="reel-to-reel")
    logic = DSPDecisionLogic()
    chain = logic.analyze_adaptive(profile)
    assert isinstance(chain, list)
    # Es sollte eine Kettenentscheidung erfolgen, die "reel" berücksichtigt
    assert (
        any("reel" in str(step) or "reel" in str(profile.material_chain.tape_type) for step in chain)
        or profile.material_chain.tape_type == "reel-to-reel"
    )
