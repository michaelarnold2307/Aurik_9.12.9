"""Unit-Tests für den Wohlklang-Blend (§Muster 3).

Der Output-Blend Original + s·(Erstlauf − Original) ersetzt den teuren
Voll-Re-Run, wenn der MUSHRA-Proxy den Blend als ausreichend bewertet.
"""

from __future__ import annotations

import numpy as np

from backend.core.unified_restorer_v3 import _compute_wohlklang_blend


def test_blend_identity_at_full_strength() -> None:
    original = np.zeros(4800, dtype=np.float32)
    first = np.full(4800, 0.5, dtype=np.float32)
    blend = _compute_wohlklang_blend(original, first, 1.0)
    assert np.allclose(blend, first)
    assert blend.dtype == np.float32


def test_blend_original_at_zero_strength() -> None:
    original = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
    first = np.full(4800, 0.9, dtype=np.float32)
    blend = _compute_wohlklang_blend(original, first, 0.0)
    assert np.allclose(blend, original)


def test_blend_halfway_is_midpoint() -> None:
    original = np.zeros(4800, dtype=np.float32)
    first = np.full(4800, 0.4, dtype=np.float32)
    blend = _compute_wohlklang_blend(original, first, 0.5)
    assert np.allclose(blend, 0.2, atol=1e-6)


def test_blend_clips_to_unit_range() -> None:
    original = np.full(4800, -0.9, dtype=np.float32)
    first = np.full(4800, 1.5, dtype=np.float32)
    blend = _compute_wohlklang_blend(original, first, 0.5)
    assert float(np.max(blend)) <= 1.0
    assert float(np.min(blend)) >= -1.0


def test_blend_preserves_dtype_and_length() -> None:
    original = np.zeros(9600, dtype=np.float32)
    first = np.linspace(-1.0, 1.0, 9600, dtype=np.float32)
    blend = _compute_wohlklang_blend(original, first, 0.7)
    assert blend.shape == (9600,)
    assert blend.dtype == np.float32
