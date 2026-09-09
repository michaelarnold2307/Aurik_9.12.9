import numpy as np
import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st

    _hypothesis_available = True
except ImportError:
    _hypothesis_available = False

if not _hypothesis_available:
    pytest.importorskip("hypothesis")  # CI-Minimal-Umgebung (cross-platform)

from backend.core.core_utils import compute_rms


def arrays(dtype=np.float32, min_size=1, max_size=48000):
    return st.lists(st.floats(-1.0, 1.0), min_size=min_size, max_size=max_size).map(lambda l: np.array(l, dtype=dtype))


@given(audio=arrays())
def test_compute_rms_property(audio):
    rms = compute_rms(audio)
    assert np.isfinite(rms)
    assert rms >= 0.0
