import math

import numpy as np

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resampelt auf target_sr mit numba-Kompatibilitäts-Guard.

    In manchen Umgebungen (ROCm-Venv) ist der numba-Dispatcher von
    librosa.resample ein plain function ohne get_call_template → AttributeError
    („'function' object has no attribute 'get_call_template'“). Fallback:
    scipy.signal.resample_poly (phasenlinear, gleiche Polyphasen-Semantik wie
    librosa). NIEMALS pass-through bei falscher Samplerate — das korrumpiert
    ML-Embeddings (Genre/CLAP-Wert-Degradation, Befund 2026-08-16).
    """
    if orig_sr == target_sr:
        return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))
    if _HAS_LIBROSA:
        try:
            return cast(
                np.ndarray,
                np.asarray(
                    librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr),
                    dtype=np.float32,
                ),
            )
        except AttributeError as exc:
            if "get_call_template" not in str(exc):
                raise
            # numba-Dispatcher-Defekt → deterministischer SciPy-Ersatzpfad
    from scipy.signal import resample_poly as _rsp

    _g = math.gcd(int(orig_sr), int(target_sr))
    _up = int(target_sr) // _g
    _down = int(orig_sr) // _g
    return cast(
        np.ndarray, (np.asarray(_rsp(np.asarray(audio, dtype=np.float32), _up, _down, axis=-1), dtype=np.float32))
    )


def resample_to_48k(audio: np.ndarray, orig_sr: int) -> tuple[np.ndarray, int]:
    """
    Resample ein beliebiges Audiosignal auf 48 kHz (Mono oder Stereo).

    Verwendet resample_audio() — librosa mit numba-Guard,
    SciPy-Ersatzpfad bei Dispatcher-Defekt.
    """
    return resample_audio(audio, orig_sr, 48000), 48000


from typing import cast
