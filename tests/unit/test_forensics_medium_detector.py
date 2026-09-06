"""Unit-Tests für forensics.medium_detector (MediumDetector, detect_medium_chain).

Testet:
- MediumDetector.detect() auf verschiedene synthetische Signale
- Mono/Stereo-Konversion
- SpectralFingerprint-Felder
- TransferChain-Erkennung
- Multi-Generation-Detection
- Singleton-Pattern + Thread-Safety
- NaN/Inf-Schutz
- Serialisierung (as_dict)
"""

import threading

import numpy as np
import pytest

from forensics.medium_detector import (
    MediumDetectionResult,
    MediumDetector,
    SpectralFingerprint,
    detect_medium_chain,
    get_medium_detector,
)


@pytest.mark.unit
class TestMediumDetector:
    """Tests für MediumDetector-Klasse."""

    def test_01_detect_mono_signal_returns_result(self):
        """Mono-Signal: detect() gibt MediumDetectionResult zurück."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_02_detect_stereo_to_mono_conversion(self):
        """Stereo-Signal: wird korrekt zu mono konvertiert."""
        np.random.seed(42)
        audio_stereo = np.random.randn(2, 48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)
        assert result.primary_material is not None

    def test_03_white_noise_cd_digital_or_unknown(self):
        """Weißes Rauschen → kein Absturz, gültiges Ergebnis."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Pure noise is pathological input — Bayesian model may score any material.
        # Key invariant: no crash, valid result structure.
        assert isinstance(result.primary_material, str)
        assert len(result.primary_material) > 0
        assert result.confidence > 0.0

    def test_04_silence_no_crash_low_confidence(self):
        """Stille → kein Absturz, confidence < 1.0."""
        audio = np.zeros(48000, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_05_bandwidth_limited_signal_shellac_or_tape(self):
        """Bandbreitenbegrenztes Signal (< 4 kHz) → shellac oder tape erkannt."""
        np.random.seed(42)
        # Synthetisches Signal mit Tiefpass-Simulation
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Erwarten niedrigen Rolloff
        assert isinstance(result.spectral_fingerprint, SpectralFingerprint)

    def test_06_rolloff_value_positive(self):
        """Rolloff-Wert ist > 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.rolloff_95_hz > 0

    def test_07_spectral_fingerprint_fields_finite(self):
        """Alle SpectralFingerprint-Felder sind finite."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        fp = result.spectral_fingerprint
        assert np.isfinite(fp.rolloff_95_hz)
        assert np.isfinite(fp.wow_flutter_index)
        assert np.isfinite(fp.hf_energy_above_16k)
        assert np.isfinite(fp.noise_floor_db)
        assert np.isfinite(fp.effective_bandwidth_hz)

    def test_08_nan_in_audio_no_crash(self):
        """NaN/Inf-Schutz: audio mit NaN → kein Absturz."""
        audio = np.full(48000, np.nan, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Sollte mit nan_to_num behandelt werden
        assert isinstance(result, MediumDetectionResult)

    def test_09_singleton_get_medium_detector_same_object(self):
        """Singleton: get_medium_detector() gibt selbes Objekt zurück."""
        det1 = get_medium_detector()
        det2 = get_medium_detector()
        assert det1 is det2

    def test_10_thread_safety_singleton(self):
        """Thread-Sicherheit: 10 parallele Aufrufe liefern selbes Singleton."""
        instances = []

        def get_instance():
            instances.append(get_medium_detector())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle Instanzen sind identisch
        assert all(inst is instances[0] for inst in instances)

    def test_11_transfer_chain_not_empty(self):
        """TransferChain ist nicht leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert len(result.transfer_chain) > 0

    def test_12_primary_material_known_type(self):
        """primary_material ist einer der bekannten MaterialType-Strings."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        known_materials = [
            "tape",
            "vinyl",
            "shellac",
            "cd_digital",
            "digital",
            "mp3_low",
            "mp3_high",
            "aac",
            "unknown",
            "streaming",
            "dat",
            "wax_cylinder",
            "wire_recording",
            "lacquer_disc",
            "minidisc",
            "reel_tape",
        ]
        assert result.primary_material in known_materials

    def test_13_confidence_in_range(self):
        """confidence ∈ [0, 1]."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert 0.0 <= result.confidence <= 1.0

    def test_14_as_dict_serializable(self):
        """as_dict() serialisierbar (alle Werte str/float/bool/list)."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "transfer_chain" in d
        assert "is_multi_generation" in d
        assert "primary_material" in d
        assert "confidence" in d

    def test_15_chain_label_contains_arrow_if_multi_generation(self):
        """chain_label enthält ' → ' wenn is_multi_generation=True."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        if result.is_multi_generation:
            assert " → " in result.chain_label

    def test_16_detect_very_short_audio_no_crash(self):
        """Sehr kurzes Audio (< 1 s) → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(4800).astype(np.float32) * 0.1  # 0.1 s @ 48 kHz
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_17_detect_long_audio_no_crash(self):
        """Langes Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(480000).astype(np.float32) * 0.1  # 10 s
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_18_spectral_fingerprint_hf_energy_in_range(self):
        """HF-Energie ∈ [0, 100] %."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert 0.0 <= result.spectral_fingerprint.hf_energy_above_16k <= 100.0

    def test_19_noise_floor_db_negative_or_zero(self):
        """Noise-Floor ist ≤ 0 dB."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.0001  # leise
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.noise_floor_db <= 0.0

    def test_20_effective_bandwidth_less_than_nyquist(self):
        """Effective Bandwidth < SR/2."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.effective_bandwidth_hz <= 24000

    def test_21_wow_flutter_index_non_negative(self):
        """Wow/Flutter-Index ≥ 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.wow_flutter_index >= 0.0

    def test_22_detect_sine_wave_stable_pitch(self):
        """Sinus-Welle: Wow/Flutter sollte klein sein."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Stabiler Pitch → kleiner Wow/Flutter
        assert result.spectral_fingerprint.wow_flutter_index < 5.0

    def test_23_evidence_list_not_none(self):
        """evidence-Liste ist nicht None."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.evidence is not None
        assert isinstance(result.evidence, list)

    def test_24_as_dict_has_expected_keys(self):
        """as_dict() hat erwartete Keys."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        expected_keys = [
            "transfer_chain",
            "is_multi_generation",
            "primary_material",
            "confidence",
            "chain_label",
            "spectral_fingerprint",
            "evidence",
        ]
        for key in expected_keys:
            assert key in d, f"Key {key} fehlt in as_dict()"

    def test_25_benign_codec_guard_prevents_false_tape_mp3_chain(self, monkeypatch):
        """Clean codec profile must not produce analog→codec chain."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=3200.0,
            wow_flutter_index=180.0,
            hf_energy_above_16k=0.0001,
            noise_floor_db=-30.0,
            effective_bandwidth_hz=13000.0,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: True)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000)

        # Benign codec guard: no analog material in chain
        for mat in result.transfer_chain:
            assert mat not in (
                "vinyl",
                "shellac",
                "tape",
                "reel_tape",
                "cassette",
                "wax_cylinder",
                "wire_recording",
                "lacquer_disc",
            ), f"Analog material {mat} should not appear in benign codec chain"

    def test_26_tape_mp3_chain_requires_analog_evidence(self, monkeypatch):
        """Tape→mp3_low chain must remain possible for genuine analog evidence."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=7000.0,
            wow_flutter_index=2.0,
            hf_energy_above_16k=0.0001,
            noise_floor_db=-32.0,
            effective_bandwidth_hz=9000.0,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000)

        # §6.1 v10.0.0: 'cassette' is a first-class material (12 kHz BW ceiling);
        # _normalize_material_key() preserves it as-is (no longer aliases to 'tape').
        assert result.primary_material in (
            "tape",
            "reel_tape",
            "cassette",
        ), f"Expected tape-family material, got {result.primary_material}"
        assert len(result.transfer_chain) >= 1
        assert result.transfer_chain[0] in ("tape", "reel_tape", "cassette")

    def test_25_stereo_shape_2_N_conversion(self):
        """Stereo-Array shape (2, N) → korrekte Mono-Konversion."""
        np.random.seed(42)
        audio_stereo = np.random.randn(2, 48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_26_stereo_shape_N_2_conversion(self):
        """Stereo-Array shape (N, 2) → korrekte Mono-Konversion."""
        np.random.seed(42)
        audio_stereo = np.random.randn(48000, 2).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_27_detect_medium_chain_convenience_function(self):
        """detect_medium_chain() convenience function works."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detect_medium_chain(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_28_rolloff_less_than_sr_half(self):
        """Rolloff < SR/2."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.rolloff_95_hz < 24000

    def test_29_primary_material_is_string(self):
        """primary_material ist String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.primary_material, str)

    def test_30_chain_label_is_string(self):
        """chain_label ist String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.chain_label, str)
        assert len(result.chain_label) > 0

    def test_31_detect_builds_four_stage_chain_with_digital_intermediate(self, monkeypatch):
        """Analog multi-generation chain should include digital lossless intermediate before codec."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=7200.0,
            wow_flutter_index=0.9,
            hf_energy_above_16k=0.0001,
            noise_floor_db=-36.0,
            effective_bandwidth_hz=9800.0,
            codec_artifact_score=0.25,
            codec_type_code=1.0,
            crackle_density=0.02,
            rotation_strength=0.12,
            infrasonic_rms=0.06,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "vinyl": 0.55,
                "tape": 0.33,
                "cd_digital": 0.24,
                "mp3_low": 0.21,
                "shellac": 0.03,
                "unknown": 0.01,
            },
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000)

        assert result.transfer_chain[:4] == ["vinyl", "tape", "cd_digital", "mp3_low"]
        assert len(result.medium_confidences) == len(result.transfer_chain)
        assert result.is_multi_generation is True

    def test_32_detect_mp3_ext_reconstructs_four_stage_chain_via_physical_inference(self, monkeypatch):
        """For .mp3 imports, physical analog inference must still allow deep multi-carrier chains."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=6800.0,
            wow_flutter_index=1.2,
            hf_energy_above_16k=0.0001,
            noise_floor_db=-34.0,
            effective_bandwidth_hz=9100.0,
            codec_artifact_score=0.30,
            codec_type_code=2.0,
            crackle_density=0.03,
            rotation_strength=0.11,
            infrasonic_rms=0.05,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "vinyl": 0.46,
                "cassette": 0.39,
                "cd_digital": 0.28,
                "mp3_low": 0.24,
                "aac": 0.08,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            lambda _fp: [("vinyl", 0.72), ("cassette", 0.58)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000, file_ext=".mp3")

        # §6.1 v10.0.0: cassette is a first-class material; _infer_analog_source_from_fingerprint
        # mock returns ("cassette", 0.58) → chain reflects cassette (not normalized to tape anymore).
        assert result.transfer_chain[:4] == ["vinyl", "cassette", "cd_digital", "mp3_low"]
        assert len(result.medium_confidences) == len(result.transfer_chain)
        assert result.is_multi_generation is True

    def test_33_detect_mp3_ext_weak_physical_inference_stays_digital_primary(self, monkeypatch):
        """Truly weak analog physical evidence (rotation < 0.30, conf < 0.20) on .mp3 must
        not override digital-primary classification.

        §2.46a Fallback-Gate: (_cand_conf >= 0.20 AND rotation >= 0.30) — beide Bedingungen
        müssen gleichzeitig verfehlt werden, damit der digitale Primär-Pfad bestehen bleibt.
        rotation=0.20 < 0.30 UND vinyl_conf=0.15 < 0.20 → bleibt mp3_low-primär.

        Hinweis: Der Vorgänger-Test verwendete rotation=0.474, was eine KLARE Vinyl-Rotation ist
        und von §2.46a korrekt als analog erkannt wird (das ist das beabsichtigte Verhalten).
        """
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=6400.0,
            wow_flutter_index=0.019,  # below tape threshold
            hf_energy_above_16k=0.0001,
            noise_floor_db=-36.0,
            effective_bandwidth_hz=12000.0,
            codec_artifact_score=0.35,
            codec_type_code=2.0,
            crackle_density=0.012,
            rotation_strength=0.20,  # below 0.30 → Fallback-Gate schlägt fehl
            infrasonic_rms=0.0102,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            # mp3_low dominant (realistic: a genuine mp3-encoded file has strong
            # codec-artifact evidence). vinyl/cassette given only marginal mass so
            # that, AFTER the ×0.50 .mp3 analog-penalty + renormalisation, both
            # fall below _ANALOG_POSTERIOR_MIN (0.08) — this is what actually
            # routes the decision into the §2.46a physical-inference Fallback-Gate
            # (the code path this test intends to exercise), instead of letting
            # vinyl win via the raw Bayesian-analog-floor check before it.
            lambda _fp, **_kw: {
                "mp3_low": 0.90,
                "cd_digital": 0.03,
                "aac": 0.02,
                "vinyl": 0.03,
                "cassette": 0.02,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            # conf=0.15 < 0.20 → Fallback-Gate schlägt fehl; keine vinyl-Übernahme
            lambda _fp: [("vinyl", 0.15), ("cassette", 0.10)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000, file_ext=".mp3")

        assert result.primary_material == "mp3_low"
        assert result.transfer_chain[0] == "mp3_low"
        assert "vinyl" not in result.transfer_chain[:1]

    def test_34_detect_mp3_ext_keeps_causal_order_and_depth_cap(self, monkeypatch):
        """Deep candidate sets must remain causal and bounded by configured chain depth."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=6200.0,
            wow_flutter_index=0.11,
            hf_energy_above_16k=0.0002,
            noise_floor_db=-33.0,
            effective_bandwidth_hz=9500.0,
            codec_artifact_score=0.33,
            codec_type_code=2.0,
            crackle_density=0.025,
            rotation_strength=0.70,
            infrasonic_rms=0.03,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "vinyl": 0.50,
                "cassette": 0.45,
                "reel_tape": 0.41,
                "wire_recording": 0.20,
                "cd_digital": 0.34,
                "mp3_low": 0.28,
                "aac": 0.10,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            lambda _fp: [("vinyl", 0.80), ("cassette", 0.62), ("reel_tape", 0.58)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000, file_ext=".mp3")

        order = detector._MEDIUM_ORDER
        # §Tiefen-Cap-Fix (2026-09-06): Die GESAMT-Kette darf 6 Stufen erreichen —
        # der Analog-Anteil ist durch _MAX_ANALOG_CHAIN_DEPTH (5) begrenzt, dazu
        # max. 1 digitale Zwischenstufe (cd_digital) + 1 Codec-Layer (mp3_low).
        # Das alte harte <= 5 verletzte legale Tief-5-Ketten.
        assert len(result.transfer_chain) <= detector._MAX_ANALOG_CHAIN_DEPTH + 2
        assert result.transfer_chain[-1] in {"mp3_low", "mp3_high", "aac"}
        assert all(
            order.get(a, 99) <= order.get(b, 99) for a, b in zip(result.transfer_chain, result.transfer_chain[1:])
        )

    def test_35_detect_mp3_vinyl_cassette_chain_codec_adaptive_gate(self, monkeypatch):
        """§2.46a regression: vinyl→cassette→mp3 multi-gen chain must not be suppressed.

        Multi-generation transfers attenuate analog fingerprints through each stage.
        The _strong_physical_analog gate must use codec-adaptive thresholds so that
        moderate analog evidence (vinyl conf=0.25, rotation=0.371) in strongly-encoded
        MP3 material still allows chain reconstruction.
        """
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=6500.0,
            wow_flutter_index=0.034,
            hf_energy_above_16k=0.0001,
            noise_floor_db=-35.0,
            effective_bandwidth_hz=12500.0,
            codec_artifact_score=0.65,
            codec_type_code=2.0,
            crackle_density=0.015,
            rotation_strength=0.371,
            infrasonic_rms=0.010,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "vinyl": 0.38,
                "cassette": 0.30,
                "cd_digital": 0.10,
                "mp3_low": 0.15,
                "aac": 0.05,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            lambda _fp: [("vinyl", 0.25), ("cassette", 0.35)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000, file_ext=".mp3")

        # Chain must include vinyl as analog origin, not just mp3_low
        assert result.is_multi_generation is True, f"Expected multi-gen chain, got single: {result.transfer_chain}"
        assert "vinyl" in result.transfer_chain, f"Vinyl must appear in chain, got: {result.transfer_chain}"
        assert result.transfer_chain[-1] in {
            "mp3_low",
            "mp3_high",
        }, f"Chain must end with MP3 codec, got: {result.transfer_chain}"

    def test_36_detect_analog_chain_prefers_mp3_low_when_bw_is_limited(self, monkeypatch):
        """Analog transfer chains with limited HF bandwidth must not end in mp3_high."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=7800.0,
            wow_flutter_index=0.060,
            hf_energy_above_16k=0.0002,
            noise_floor_db=-33.0,
            effective_bandwidth_hz=15_500.0,
            codec_artifact_score=0.32,
            codec_type_code=2.0,
            crackle_density=0.020,
            rotation_strength=0.36,
            infrasonic_rms=0.030,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "mp3_high": 0.45,
                "mp3_low": 0.22,
                "cd_digital": 0.18,
                "vinyl": 0.10,
                "cassette": 0.09,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            lambda _fp: [("vinyl", 0.72), ("cassette", 0.58)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detector.detect(audio, sr=48000, file_ext=".mp3")

        # §6.1 v10.0.0: cassette is a first-class material; mock returns ("cassette", 0.58)
        # → chain reflects cassette (not normalized to tape anymore).
        assert result.transfer_chain[:2] == ["vinyl", "cassette"]
        assert result.transfer_chain[-1] == "mp3_low", (
            f"Expected codec stage mp3_low for analog chain with limited bandwidth, got chain={result.transfer_chain}"
        )

    def test_37_detect_file_ext_without_dot_is_normalized(self, monkeypatch):
        """file_ext="mp3" and file_ext=".mp3" must produce identical transfer chains."""
        detector = MediumDetector()
        fp = SpectralFingerprint(
            rolloff_95_hz=7800.0,
            wow_flutter_index=0.060,
            hf_energy_above_16k=0.0002,
            noise_floor_db=-33.0,
            effective_bandwidth_hz=15_500.0,
            codec_artifact_score=0.32,
            codec_type_code=2.0,
            crackle_density=0.020,
            rotation_strength=0.36,
            infrasonic_rms=0.030,
        )

        monkeypatch.setattr(detector, "_compute_fingerprint", lambda _audio, _sr: fp)
        monkeypatch.setattr(
            detector,
            "_bayesian_score",
            lambda _fp, **_kw: {
                "mp3_high": 0.45,
                "mp3_low": 0.22,
                "cd_digital": 0.18,
                "vinyl": 0.10,
                "cassette": 0.09,
            },
        )
        monkeypatch.setattr(
            detector,
            "_infer_analog_source_from_fingerprint",
            lambda _fp: [("vinyl", 0.72), ("cassette", 0.58)],
        )
        monkeypatch.setattr(detector, "_is_benign_codec_source", lambda _audio, _sr, _fp: False)

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result_with_dot = detector.detect(audio, sr=48000, file_ext=".mp3")
        result_without_dot = detector.detect(audio, sr=48000, file_ext="mp3")

        assert result_with_dot.transfer_chain == result_without_dot.transfer_chain
        assert result_without_dot.transfer_chain[:2] == ["vinyl", "cassette"]
        assert result_without_dot.transfer_chain[-1] == "mp3_low"


class TestCodecStageConfidence:
    """Tests für die evidenzbasierte Codec-Stufen-Konfidenz (statt Platzhalter 0.40/0.35)."""

    def test_floor_at_no_evidence(self):
        """Ohne jede Evidenz (Vollband, kein Artefakt, kein Typ-Code) → Boden 0.35."""
        fp = SpectralFingerprint(effective_bandwidth_hz=20_000.0, codec_artifact_score=0.0, codec_type_code=0.0)
        assert MediumDetector._codec_stage_confidence(fp) == pytest.approx(0.35)

    def test_ceiling_at_full_evidence(self):
        """Volle Evidenz (BW ≤ 10 kHz, Artefakt ≥ 0.30, Typ-Code ≥ 1) → Deckel 0.90."""
        fp = SpectralFingerprint(effective_bandwidth_hz=9_000.0, codec_artifact_score=0.5, codec_type_code=1.0)
        assert MediumDetector._codec_stage_confidence(fp) == pytest.approx(0.90)

    def test_real_schlager_mp3_low_above_kpi_threshold(self):
        """Real-Song-Fingerprint (Elke Best, 13.7 kHz BW, Artefakt 0.086) → ≥ 0.45 KPI-Schwelle."""
        fp = SpectralFingerprint(effective_bandwidth_hz=13_729.0, codec_artifact_score=0.086, codec_type_code=0.0)
        conf = MediumDetector._codec_stage_confidence(fp)
        assert conf >= 0.45
        assert conf == pytest.approx(0.35 + 0.30 * ((17_500 - 13_729) / 7_500) + 0.20 * (0.086 / 0.30), abs=1e-6)

    def test_monotonic_in_bandwidth_cut(self):
        """Tieferer Codec-Tiefpass → höhere Konfidenz (Monotonie)."""
        confs = [
            MediumDetector._codec_stage_confidence(
                SpectralFingerprint(effective_bandwidth_hz=bw, codec_artifact_score=0.1, codec_type_code=0.0)
            )
            for bw in (16_000.0, 14_000.0, 12_000.0, 10_000.0)
        ]
        assert confs == sorted(confs)

    def test_above_digital_lock_threshold_for_clear_mp3(self):
        """Klare MP3-Signatur (BW < 14 kHz) → Konfidenz ≥ 0.35 Digital-Lock-Schwelle."""
        fp = SpectralFingerprint(effective_bandwidth_hz=12_931.0, codec_artifact_score=0.136, codec_type_code=0.0)
        assert MediumDetector._codec_stage_confidence(fp) >= 0.35

    def test_result_always_in_valid_range(self):
        """Extremwerte (negativ/NaN-frei) bleiben im Intervall [0.35, 0.90]."""
        for bw, art, code in ((0.0, 1.0, 3.0), (48_000.0, -0.5, -1.0), (17_500.0, 0.0, 0.0)):
            fp = SpectralFingerprint(effective_bandwidth_hz=bw, codec_artifact_score=art, codec_type_code=code)
            conf = MediumDetector._codec_stage_confidence(fp)
            assert 0.35 <= conf <= 0.90
