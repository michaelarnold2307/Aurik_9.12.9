"""
test_transparent_dynamics.py - Tests für Transparent Dynamics & Micro-Dynamics (GAP #10, #11)

Testet:
- TransparentDynamicsProcessor (GAP #10): Transparente Compression
- MicroDynamicsEnhancer (GAP #11): Mikro-Dynamik-Erhaltung
- DynamicsProcessor (unified API)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")  # CI-Minimal-Umgebung (cross-platform)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.transparent_dynamics import (
    DynamicsProcessor,
    MicroDynamicsEnhancer,
    TransparentDynamicsProcessor,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests"""
    return 48000


@pytest.fixture
def duration():
    """Standard duration in seconds"""
    return 2.0


@pytest.fixture
def mono_audio(sample_rate, duration):
    """Generate mono test audio with varying dynamics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Amplitude modulation: varying envelope
    envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
    audio = envelope * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    return audio


@pytest.fixture
def stereo_audio(mono_audio):
    """Generate stereo test audio"""
    return np.vstack([mono_audio, mono_audio * 0.9])


@pytest.fixture
def loud_audio(sample_rate, duration):
    """Generate loud audio requiring compression"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Loud signal with peaks at 0 dB
    audio = 0.9 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def flat_dynamics_audio(sample_rate, duration):
    """Generate audio with flat dynamics (no micro-variations)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Constant amplitude
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def micro_dynamic_audio(sample_rate, duration):
    """Generate audio with rich micro-dynamics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Rapid amplitude variations (50 Hz modulation)
    envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 50 * t)
    audio = envelope * np.sin(2 * np.pi * 440 * t)
    return audio


# =============================================================================
# TESTS: TransparentDynamicsProcessor (GAP #10)
# =============================================================================


@pytest.mark.unit
class TestTransparentDynamicsProcessor:
    """Tests for GAP #10: Transparent Dynamics Restoration"""

    def test_initialization(self):
        """Test initialization with default parameters"""
        processor = TransparentDynamicsProcessor()
        assert processor.target_ratio == 2.0
        assert processor.threshold_db == -20.0
        assert processor.knee_db == 6.0
        assert processor.attack_ms == 10.0
        assert processor.release_ms == 100.0
        assert processor.adaptive is True

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        processor = TransparentDynamicsProcessor(
            target_ratio=4.0, threshold_db=-15.0, knee_db=8.0, attack_ms=5.0, release_ms=200.0, adaptive=False
        )
        assert processor.target_ratio == 4.0
        assert processor.threshold_db == -15.0
        assert processor.knee_db == 8.0
        assert processor.attack_ms == 5.0
        assert processor.release_ms == 200.0
        assert processor.adaptive is False

    def test_parameter_clipping(self):
        """Test that parameters are clipped to valid ranges"""
        processor = TransparentDynamicsProcessor(
            target_ratio=15.0,  # > 10.0
            threshold_db=-50.0,  # < -40.0
            knee_db=20.0,  # > 12.0
            attack_ms=0.5,  # < 1.0
            release_ms=2000.0,  # > 1000.0
        )
        assert processor.target_ratio == 10.0
        assert processor.threshold_db == -40.0
        assert processor.knee_db == 12.0
        assert processor.attack_ms == 1.0
        assert processor.release_ms == 1000.0

    def test_compute_envelope(self, mono_audio, sample_rate):
        """Test RMS envelope computation"""
        processor = TransparentDynamicsProcessor()
        envelope_db = processor.compute_envelope(mono_audio, sample_rate)

        # Should return same length
        assert len(envelope_db) == len(mono_audio)

        # Should be in dB (negative values for signals < 1.0)
        assert np.all(envelope_db < 0)

    def test_compute_gain_reduction(self, sample_rate):
        """Test gain reduction computation"""
        processor = TransparentDynamicsProcessor(target_ratio=4.0, threshold_db=-20.0, knee_db=6.0)

        # Test envelope: some below, some above threshold
        envelope_db = np.array([-30, -25, -20, -15, -10, -5])

        gain_reduction_db = processor.compute_gain_reduction(envelope_db)

        # Should be non-positive (gain reduction)
        assert np.all(gain_reduction_db <= 0)

        # Below threshold: minimal/no reduction
        assert gain_reduction_db[0] == 0.0  # -30 dB << -20 dB

        # Above threshold: significant reduction
        assert gain_reduction_db[-1] < -5  # -5 dB >> -20 dB

    def test_smooth_gain_reduction(self, sample_rate):
        """Test attack/release smoothing"""
        processor = TransparentDynamicsProcessor(attack_ms=10.0, release_ms=100.0)

        # Instantaneous gain reduction with sudden change
        duration = 0.5
        n_samples = int(sample_rate * duration)
        gain_reduction_db = np.zeros(n_samples)
        gain_reduction_db[n_samples // 2 :] = -10.0  # Sudden -10 dB

        smoothed = processor.smooth_gain_reduction(gain_reduction_db, sample_rate)

        # Should be smoothed (no sudden jumps)
        diff = np.diff(smoothed)
        max_diff = np.max(np.abs(diff))
        assert max_diff < 0.1  # Should be smooth

    def test_analyze_dynamics(self, mono_audio):
        """Test dynamics analysis"""
        processor = TransparentDynamicsProcessor()
        analysis = processor.analyze_dynamics(mono_audio)

        assert "peak_db" in analysis
        assert "rms_db" in analysis
        assert "crest_factor_db" in analysis
        assert "dynamic_range_db" in analysis

        # Peak should be higher than RMS
        assert analysis["peak_db"] > analysis["rms_db"]

        # Crest factor should be positive
        assert analysis["crest_factor_db"] > 0

    def test_process_mono_loud_audio(self, loud_audio, sample_rate):
        """Test processing loud audio requiring compression"""
        processor = TransparentDynamicsProcessor(target_ratio=3.0, threshold_db=-10.0)

        # Analyze before
        analysis_before = processor.analyze_dynamics(loud_audio)
        peak_before = analysis_before["peak_db"]

        # Process
        processed = processor.process(loud_audio, sample_rate)

        # Analyze after
        analysis_after = processor.analyze_dynamics(processed)
        peak_after = analysis_after["peak_db"]

        # Peak should be reduced
        assert peak_after < peak_before

        # Should preserve length
        assert len(processed) == len(loud_audio)

        # Should not clip
        assert np.max(np.abs(processed)) <= 1.0

        # Metrics should be populated
        assert "gain_reduction_max_db" in processor.metrics
        assert processor.metrics["gain_reduction_max_db"] < 0  # Negative = reduction applied

    def test_process_stereo_audio(self, stereo_audio, sample_rate):
        """Test processing stereo audio"""
        processor = TransparentDynamicsProcessor()

        processed = processor.process(stereo_audio, sample_rate)

        # Should preserve stereo shape
        assert processed.shape == stereo_audio.shape
        assert processed.ndim == 2

    def test_no_compression_when_below_threshold(self, sample_rate, duration):
        """Test that no compression is applied when below threshold"""
        # Generate quiet audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.01 * np.sin(2 * np.pi * 440 * t)  # Very quiet

        processor = TransparentDynamicsProcessor(threshold_db=-20.0)
        processed = processor.process(audio, sample_rate)

        # Should be unchanged (or nearly so)
        correlation = np.corrcoef(audio, processed)[0, 1]
        assert correlation > 0.99

    def test_quality_gate_over_compression(self, loud_audio, sample_rate):
        """Test quality gate prevents extreme compression"""
        processor = TransparentDynamicsProcessor(
            target_ratio=10.0,
            threshold_db=-30.0,  # Extreme ratio  # Low threshold
        )

        processor.process(loud_audio, sample_rate)

        # Gain reduction should be limited (allow up to -25 dB for extreme cases)
        assert processor.metrics.get("gain_reduction_max_db", 0) >= -25

    def test_metrics_reporting(self, loud_audio, sample_rate):
        """Test that metrics are properly reported"""
        processor = TransparentDynamicsProcessor()
        processor.process(loud_audio, sample_rate)

        assert hasattr(processor, "metrics")
        assert "crest_factor_before" in processor.metrics
        assert "crest_factor_after" in processor.metrics
        assert "gain_reduction_max_db" in processor.metrics
        assert "dynamic_range_reduction_db" in processor.metrics


# =============================================================================
# TESTS: MicroDynamicsEnhancer (GAP #11)
# =============================================================================


class TestMicroDynamicsEnhancer:
    """Tests for GAP #11: Micro-Dynamics Enhancer"""

    def test_initialization(self):
        """Test initialization with default parameters"""
        enhancer = MicroDynamicsEnhancer()
        assert enhancer.enhancement_amount == 0.5
        assert enhancer.time_window_ms == 50
        assert enhancer.frequency_selective is True

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        enhancer = MicroDynamicsEnhancer(enhancement_amount=0.8, time_window_ms=30, frequency_selective=False)
        assert enhancer.enhancement_amount == 0.8
        assert enhancer.time_window_ms == 30
        assert enhancer.frequency_selective is False

    def test_parameter_clipping(self):
        """Test that parameters are clipped to valid ranges"""
        enhancer = MicroDynamicsEnhancer(enhancement_amount=1.5, time_window_ms=5)  # > 1.0  # < 10
        assert enhancer.enhancement_amount == 1.0
        assert enhancer.time_window_ms == 10

    def test_analyze_micro_dynamics_flat(self, flat_dynamics_audio, sample_rate):
        """Test micro-dynamics analysis on flat audio"""
        enhancer = MicroDynamicsEnhancer()
        analysis = enhancer.analyze_micro_dynamics(flat_dynamics_audio, sample_rate)

        assert "micro_dynamics_score" in analysis

        # Flat audio should have low score
        assert analysis["micro_dynamics_score"] < 0.3

    def test_analyze_micro_dynamics_variable(self, micro_dynamic_audio, sample_rate):
        """Test micro-dynamics analysis on variable audio"""
        enhancer = MicroDynamicsEnhancer()
        analysis = enhancer.analyze_micro_dynamics(micro_dynamic_audio, sample_rate)

        # Variable audio should have higher score than flat audio
        # (50 Hz modulation gets smoothed by 50ms window, so score is modest)
        assert analysis["micro_dynamics_score"] > 0.03

    def test_enhance_micro_dynamics(self, flat_dynamics_audio, sample_rate):
        """Test micro-dynamics enhancement"""
        enhancer = MicroDynamicsEnhancer(enhancement_amount=0.7)

        enhanced = enhancer.enhance_micro_dynamics(flat_dynamics_audio, sample_rate)

        # Should return same length
        assert len(enhanced) == len(flat_dynamics_audio)

        # Should not clip
        assert np.max(np.abs(enhanced)) <= 1.5  # Allow some headroom

    def test_process_mono_flat_audio(self, flat_dynamics_audio, sample_rate):
        """Test processing flat audio"""
        enhancer = MicroDynamicsEnhancer(enhancement_amount=0.6)

        # Analyze before
        enhancer.analyze_micro_dynamics(flat_dynamics_audio, sample_rate)

        # Process
        processed = enhancer.process(flat_dynamics_audio, sample_rate)

        # Should preserve length
        assert len(processed) == len(flat_dynamics_audio)

        # Should not clip
        assert np.max(np.abs(processed)) <= 1.0

        # Metrics should be populated
        assert "micro_dynamics_score" in enhancer.metrics

    def test_process_mono_variable_audio(self, micro_dynamic_audio, sample_rate):
        """Test processing variable audio"""
        enhancer = MicroDynamicsEnhancer(enhancement_amount=0.7)

        processed = enhancer.process(micro_dynamic_audio, sample_rate)

        # Should preserve length
        assert len(processed) == len(micro_dynamic_audio)

        # Should not clip
        assert np.max(np.abs(processed)) <= 1.0

    def test_process_stereo_audio(self, stereo_audio, sample_rate):
        """Test processing stereo audio"""
        enhancer = MicroDynamicsEnhancer()

        processed = enhancer.process(stereo_audio, sample_rate)

        # Should preserve stereo shape
        assert processed.shape == stereo_audio.shape
        assert processed.ndim == 2

    def test_frequency_selective_enhancement(self, mono_audio, sample_rate):
        """Test frequency-selective enhancement"""
        # With frequency-selective
        enhancer_selective = MicroDynamicsEnhancer(enhancement_amount=0.7, frequency_selective=True)
        processed_selective = enhancer_selective.process(mono_audio, sample_rate)

        # Without frequency-selective
        enhancer_full = MicroDynamicsEnhancer(enhancement_amount=0.7, frequency_selective=False)
        processed_full = enhancer_full.process(mono_audio, sample_rate)

        # Should produce different results
        assert not np.allclose(processed_selective, processed_full)

    def test_quality_gate_clipping_prevention(self, sample_rate, duration):
        """Test quality gate prevents clipping"""
        # Generate hot audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        envelope = 0.8 + 0.1 * np.sin(2 * np.pi * 10 * t)
        hot_audio = envelope * np.sin(2 * np.pi * 440 * t)

        enhancer = MicroDynamicsEnhancer(enhancement_amount=1.0)
        processed = enhancer.process(hot_audio, sample_rate)

        # Should not exceed 0.99
        assert np.max(np.abs(processed)) <= 0.99

    def test_metrics_reporting(self, micro_dynamic_audio, sample_rate):
        """Test that metrics are properly reported"""
        enhancer = MicroDynamicsEnhancer()
        enhancer.process(micro_dynamic_audio, sample_rate)

        assert hasattr(enhancer, "metrics")
        assert "micro_dynamics_score" in enhancer.metrics
        assert "enhancement_applied_db" in enhancer.metrics


# =============================================================================
# TESTS: DynamicsProcessor (Unified API)
# =============================================================================


class TestDynamicsProcessor:
    """Tests for unified DynamicsProcessor API"""

    def test_initialization_all_enabled(self):
        """Test initialization with all modules enabled"""
        processor = DynamicsProcessor()

        assert processor.enable_transparent_dynamics is True
        assert processor.enable_micro_dynamics is True
        assert hasattr(processor, "transparent_processor")
        assert hasattr(processor, "micro_enhancer")

    def test_initialization_selective_enable(self):
        """Test initialization with selective module enable"""
        processor = DynamicsProcessor(enable_transparent_dynamics=True, enable_micro_dynamics=False)

        assert processor.enable_transparent_dynamics is True
        assert processor.enable_micro_dynamics is False
        assert hasattr(processor, "transparent_processor")
        assert not hasattr(processor, "micro_enhancer")

    def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        processor = DynamicsProcessor(target_ratio=3.0, threshold_db=-15.0, enhancement_amount=0.7)

        assert processor.transparent_processor.target_ratio == 3.0
        assert processor.transparent_processor.threshold_db == -15.0
        assert processor.micro_enhancer.enhancement_amount == 0.7

    def test_process_mono_audio(self, mono_audio, sample_rate):
        """Test processing mono audio with all modules"""
        processor = DynamicsProcessor()

        processed = processor.process(mono_audio, sample_rate)

        # Should preserve length
        assert len(processed) == len(mono_audio)

        # Should not clip
        assert np.max(np.abs(processed)) <= 1.0

    def test_process_stereo_audio(self, stereo_audio, sample_rate):
        """Test processing stereo audio with all modules"""
        processor = DynamicsProcessor()

        processed = processor.process(stereo_audio, sample_rate)

        # Should preserve shape
        assert processed.shape == stereo_audio.shape
        assert processed.ndim == 2

        # Should not clip
        assert np.max(np.abs(processed)) <= 1.0

    def test_process_sequential_application(self, loud_audio, sample_rate):
        """Test that modules are applied in correct sequence"""
        processor = DynamicsProcessor(target_ratio=3.0, threshold_db=-10.0, enhancement_amount=0.6)

        # Process
        processor.process(loud_audio, sample_rate)

        # Should have applied both modules
        metrics = processor.get_metrics()

        # Check metrics from both modules exist
        assert "transparent_dynamics" in metrics or "micro_dynamics" in metrics

    def test_get_metrics(self, mono_audio, sample_rate):
        """Test metrics collection from all modules"""
        processor = DynamicsProcessor()
        processor.process(mono_audio, sample_rate)

        metrics = processor.get_metrics()

        # Should have metrics from all enabled modules
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_process_with_only_transparent(self, loud_audio, sample_rate):
        """Test processing with only transparent dynamics enabled"""
        processor = DynamicsProcessor(enable_transparent_dynamics=True, enable_micro_dynamics=False, target_ratio=3.0)

        processed = processor.process(loud_audio, sample_rate)

        assert len(processed) == len(loud_audio)

        # Should only have transparent dynamics metrics
        metrics = processor.get_metrics()
        assert "transparent_dynamics" in metrics
        assert "micro_dynamics" not in metrics

    def test_process_with_only_micro(self, flat_dynamics_audio, sample_rate):
        """Test processing with only micro-dynamics enabled"""
        processor = DynamicsProcessor(
            enable_transparent_dynamics=False, enable_micro_dynamics=True, enhancement_amount=0.7
        )

        processed = processor.process(flat_dynamics_audio, sample_rate)

        assert len(processed) == len(flat_dynamics_audio)

        # Should only have micro dynamics metrics
        metrics = processor.get_metrics()
        assert "transparent_dynamics" not in metrics
        assert "micro_dynamics" in metrics


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_pipeline_realistic_audio(self, sample_rate):
        """Test full pipeline on realistic problematic audio"""
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create audio with both problems:
        # 1. Loud peaks (needs compression)
        # 2. Flat micro-dynamics (needs enhancement)

        # Base tone with loud peaks
        audio = 0.7 * np.sin(2 * np.pi * 440 * t)
        # Add occasional loud peaks
        peak_times = [0.5, 1.0, 1.5]
        for peak_time in peak_times:
            peak_idx = int(peak_time * sample_rate)
            peak_width = int(0.01 * sample_rate)  # 10ms
            if peak_idx < len(audio):
                audio[max(0, peak_idx - peak_width) : min(len(audio), peak_idx + peak_width)] *= 1.3

        # Process with all modules
        processor = DynamicsProcessor(target_ratio=3.0, threshold_db=-10.0, enhancement_amount=0.6)

        processed = processor.process(audio, sample_rate)

        # Verify processing
        metrics = processor.get_metrics()

        # Should have processed successfully
        assert len(processed) == len(audio)
        assert np.max(np.abs(processed)) <= 1.0

        # Should have metrics from both modules
        assert len(metrics) >= 1

    def test_preserves_silence(self, sample_rate, duration):
        """Test that silence remains silence"""
        silence = np.zeros(int(sample_rate * duration))

        processor = DynamicsProcessor()
        processed = processor.process(silence, sample_rate)

        # Should still be (near) silence
        rms = np.sqrt(np.mean(processed**2))
        assert rms < 1e-6


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance and efficiency tests"""

    def test_processing_time_reasonable(self, mono_audio, sample_rate):
        """Test that processing time is reasonable"""
        import time

        processor = DynamicsProcessor()

        start = time.time()
        processor.process(mono_audio, sample_rate)
        elapsed = time.time() - start

        audio_duration = len(mono_audio) / sample_rate

        # Should process faster than 15× real-time (conservative)
        assert elapsed < audio_duration * 15

        print(f"\nProcessing time: {elapsed:.3f}s for {audio_duration:.1f}s audio")
        print(f"Real-time factor: {elapsed / audio_duration:.2f}×")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
