"""Unit-Tests für die Fixed-T-Chunked-Inferenz des DeepFilterNet-V3-II-Plugins.

§Fixed-T-Export (2026-09-07): Der enc-ONNX-Export fixiert die Zeitdimension
(T=100) — Ganzsignal-Feeds warfen InvalidArgument (index 2: Got 2999, Expected
100). Die Chunked-Inferenz (50-%-Überlappung + Hann-OLA im Spektralbereich)
muss für einen transparenten Spektral-Kern EXAKT das Ganzsignal-Ergebnis
reproduzieren — das belegt der Test ohne ONNX-Runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin


class _FakeChunkPlugin(DeepFilterNetV3Plugin):
    """Plugin ohne Modell-Load; Spektral-Kern ist ein 0.5-Passthrough."""

    def __init__(self, time_frames: int | None) -> None:
        self._enc = None  # type: ignore[assignment]
        self._dec = None  # type: ignore[assignment]
        self._erb_dec = None  # type: ignore[assignment]
        self._enc_time_frames = time_frames
        self._current_energy_bias_db = 0.0
        self._chunk_calls = 0

    def _infer_spectral_chunk(self, feat_erb, feat_spec, spec_cx):  # type: ignore[override]
        self._chunk_calls += 1
        assert spec_cx.shape[1] <= (self._enc_time_frames or spec_cx.shape[1])
        return spec_cx * 0.5


def _make_audio(dur_s: float) -> np.ndarray:
    sr = 48000
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)


def test_whole_signal_path_when_time_dimension_dynamic() -> None:
    """Dynamisches T → Ganzsignal-Pfad, exakt EIN Kern-Aufruf."""
    plugin = _FakeChunkPlugin(time_frames=None)
    audio = _make_audio(3.0)
    out = plugin._infer_onnx(audio)
    assert plugin._chunk_calls == 1
    assert len(out) == len(audio)
    # 0.5-Passthrough: Ausgabe ≈ 0.5 × Eingabe. Die ersten/letzten ~4 Samples
    # tragen den VORBESTEHENDEN STFT-Randfehler des Plugins (Analysis-Hann am
    # Signalrand) — daher Vergleich im Inneren.
    assert np.allclose(out[2000:-2000], 0.5 * audio[2000:-2000], atol=1e-3)


def test_chunked_equals_whole_signal_for_transparent_kernel() -> None:
    """Chunked-Pfad (T=100) reproduziert das Ganzsignal-Ergebnis EXAKT.

    Ein transparenter Spektral-Kern (×0.5) darf durch die Chunk-Grenzen +
    Hann-OLA keine Abweichung erzeugen — das ist der Beleg für
    „keine negativen Seiteneffekte“ der Chunked-Inferenz.
    """
    audio = _make_audio(4.0)  # S = (192000-960)/480+1 = 399 Frames > 2×T
    whole = _FakeChunkPlugin(time_frames=None)
    chunked = _FakeChunkPlugin(time_frames=100)
    out_whole = whole._infer_onnx(audio)
    out_chunked = chunked._infer_onnx(audio)
    assert chunked._chunk_calls > 2  # tatsächlich gechunkt
    assert np.allclose(out_chunked, out_whole, atol=1e-4)


def test_tail_chunk_padding_is_transparent() -> None:
    """Signal-Länge ohne T-Raster (399 Frames, T=100) → Rand-Padding unsichtbar."""
    audio = _make_audio(4.0)  # S=399: letzter Chunk 99 Frames → Padding nötig
    plugin = _FakeChunkPlugin(time_frames=100)
    out = plugin._infer_onnx(audio)
    ref = _FakeChunkPlugin(time_frames=None)._infer_onnx(audio)
    assert np.allclose(out, ref, atol=1e-4)
    assert np.isfinite(out).all()


def test_short_signal_below_t_uses_chunked_path() -> None:
    """§P1-6: Signal kürzer als T → Chunked-Pfad (Pad+Trim) statt Ganzsignal.

    Der Fixed-T-enc erwartet exakt T=100 Frames; der Ganzsignal-Pfad mit
    S<T warf InvalidArgument (Got 89, Expected 100) → stiller Fallback.
    """
    audio = _make_audio(0.9)  # S = (43200-960)/480+1 = 89 < 100
    plugin = _FakeChunkPlugin(time_frames=100)
    out = plugin._infer_onnx(audio)
    assert plugin._chunk_calls == 2  # pos=0 (l=89) + pos=50 (l=39), 50-%-Overlap
    assert len(out) == len(audio)
    ref = _FakeChunkPlugin(time_frames=None)._infer_onnx(audio)
    assert np.allclose(out, ref, atol=1e-4)
    assert np.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
