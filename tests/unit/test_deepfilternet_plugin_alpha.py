"""§P1-6 — DeepFilterNet dec.onnx ohne Alpha-Head (ML→DSP-Befund).

Befund 2026-09-08: IndexError in _infer_spectral_chunk (alpha = dec_out[1])
→ stiller OMLSA-Fallback. Root-Cause: DFN3-Export liefert nur coefs
(df_fc_a ist im trainierten Forward unbenutzt). Fix: alpha optional;
fehlendes alpha = pure DF wie im trainierten Forward (blend=1.0).
"""

from __future__ import annotations

import numpy as np

from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin


class _FakeSession:
    """ONNX-Session-Stub mit festen Outputs (nur run())."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outs = outputs

    def run(self, *_args, **_kwargs) -> list[np.ndarray]:
        return list(self._outs)


_S = 12  # Frames (realistisch >= DF_ORDER; Produktion: T=100)


def _make_plugin(dec_outputs: list[np.ndarray]) -> DeepFilterNetV3Plugin:
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)  # ohne Modell-Load
    p._enc = _FakeSession(
        [np.zeros((1, 16, _S, 32)), np.zeros((1, 16, _S, 16)), np.zeros((1, 16, _S, 8)),
         np.zeros((1, 16, _S, 8)), np.zeros((1, _S, 128)), np.zeros((1, 16, _S, 96)),
         np.zeros((1, _S, 1))]
    )
    p._erb_dec = _FakeSession([np.zeros((1, 1, _S, 32))])
    p._dec = _FakeSession(dec_outputs)
    return p


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_erb = np.zeros((1, 1, _S, 32), dtype=np.float32)
    feat_spec = np.zeros((1, 2, _S, 96), dtype=np.float32)
    spec_cx = np.zeros((481, _S), dtype=np.complex64)  # volle FFT-Bins
    return feat_erb, feat_spec, spec_cx


def test_infer_spectral_chunk_single_output_no_crash() -> None:
    """§P1-6: dec mit nur coefs → kein IndexError, pure DF."""
    p = _make_plugin([np.zeros((1, _S, 96, 10), dtype=np.float32)])
    out = p._infer_spectral_chunk(*_inputs())
    assert out.shape == (481, _S)


def test_infer_spectral_chunk_two_outputs_uses_alpha() -> None:
    """dec mit coefs + alpha → Alpha-Pfad bleibt funktional."""
    alpha = np.full((1, _S, 1), 0.5, dtype=np.float32)
    p = _make_plugin([np.zeros((1, _S, 96, 10), dtype=np.float32), alpha])
    out = p._infer_spectral_chunk(*_inputs())
    assert out.shape == (481, _S)


def test_apply_df_filter_none_alpha_is_pure_df() -> None:
    """alpha=None → blend=1.0 (pure DF, wie trainierter DFN3-Forward)."""
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)
    spec = np.ones((481, _S), dtype=np.complex64)
    coefs = np.zeros((_S, 96, 10), dtype=np.float32)
    coefs[:, :, 0] = 0.25  # nur 0. Koeffizient aktiv
    out_none = p._apply_df_filter(spec, coefs, None)
    out_explicit = p._apply_df_filter(spec, coefs, np.full((1, _S, 1), 1.0, dtype=np.float32))
    # pure DF: acc = coefs[:, :, 0].T * spec → 0.25 (nur erste 96 Bins)
    expected = np.ones((481, _S), dtype=np.complex64)
    expected[:96, :] = 0.25
    assert np.allclose(out_none, expected, atol=1e-6)
    assert np.allclose(out_none, out_explicit, atol=1e-6)


def test_apply_df_filter_alpha_blend_unchanged() -> None:
    """Alpha-Blend-Pfad (0.5) bleibt wie vor dem Fix."""
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)
    spec = np.ones((481, _S), dtype=np.complex64)
    coefs = np.zeros((_S, 96, 10), dtype=np.float32)
    coefs[:, :, 0] = 0.5
    alpha = np.full((1, _S, 1), 0.5, dtype=np.float32)
    out = p._apply_df_filter(spec, coefs, alpha)
    # blend 0.5: 0.5 * (0.5) + 0.5 * 1.0 = 0.75 (erste 96 Bins)
    expected = np.ones((481, _S), dtype=np.complex64)
    expected[:96, :] = 0.75
    assert np.allclose(out, expected, atol=1e-6)
