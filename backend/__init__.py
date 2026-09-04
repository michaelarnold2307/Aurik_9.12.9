"""Backend-Paket für Aurik 10.0.0 — DSP, ML-Modelle, Denker und API."""

# §v10.14 ATOMIC CACHE-CLEAR: Löscht ALLE __pycache__-Verzeichnisse
# rekursiv beim ersten Import. Garantiert dass JEDER Aurik-Start mit
# frischem Bytecode läuft — kein manuelles find/rm mehr nötig.
import pathlib as _bclear_path
import shutil as _bclear_shutil
import logging

logger = logging.getLogger(__name__)

_BACKEND_ROOT = _bclear_path.Path(__file__).parent
for _bclear_d in _BACKEND_ROOT.rglob("__pycache__"):
    _bclear_shutil.rmtree(_bclear_d, ignore_errors=True)

# Ermöglicht Import von backend als Paket für Tests

# ── §2.62 STFT Input-Length-Guard (zentral, schützt alle 62+ Aufrufer) ──
# scipy.signal.stft warnt bei nperseg > input_length ("using nperseg = 2").
# Aurik-DSP erzeugt in 2-Sample-Subbändern korrekte Kurz-Arrays; die Warnung
# ist harmlos aber flutet das Log. Statt 62+ Einzel-Guards patchen wir
# scipy.signal.stft einmalig mit einem transparenten Längen-Guard.
import warnings as _warnings

import numpy as _np
import scipy.signal as _scipy_signal

_original_stft = _scipy_signal.stft


def _safe_stft(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,  # type: ignore[no-untyped-def]
    nfft=None,
    detrend=False,
    return_onesided=True,
    scaling="spectrum",
    axis=-1,
    boundary=None,
    padded=True,
):
    """scipy.signal.stft mit transparentem Längen-Guard.

    Bei input_length < nperseg gibt scipy eine UserWarning aus und
    verwendet nperseg=input_length. Dieser Guard fängt den Fall ab,
    ohne die Warnung auszulösen — das Verhalten ist identisch.
    §v10.119: Normalisiert boundary=True → 'zeros' (scipy-kompatibel).
    """
    # §v10.119: boundary=True ist in scipy ungültig
    _boundary = "zeros" if boundary is True else boundary
    arr = _np.asarray(x)
    if arr.size == 0:
        nfft_val = nfft if nfft is not None else nperseg
        return _np.zeros(nfft_val // 2 + 1), _np.array([0.0]), _np.zeros((nfft_val // 2 + 1, 0), dtype=complex)
    if arr.size < nperseg:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return _original_stft(
                x,
                fs=fs,
                window=window,
                nperseg=arr.size,
                noverlap=max(0, min(noverlap or arr.size // 2, arr.size - 1)),
                nfft=nfft,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling,
                axis=axis,
                boundary=_boundary,
                padded=padded,
            )
    return _original_stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        boundary=_boundary,
        padded=padded,
    )


# §22/v10.115: Kein globaler scipy.signal.stft-Monkey-Patch mehr.
# Der globale Patch verfälschte die Frame-Geometrie für Fremd-Consumer
# (z. B. dsp/advanced_dereverb, scipy-Nutzer in Tests) → STFT/ISTFT-Versatz.
# Phasen verwenden explizit `from backend.core.audio_utils import safe_stft, safe_istft`.
# Der attribut-basierte Zugriff bleibt für Rückwärtskompatibilität erhalten:
_scipy_signal.safe_stft = _safe_stft


# §v10.303 FIX: safe_istft war fälschlich auf _safe_stft (STFT) gesetzt.
# Dadurch gab signal.safe_istft() 3 Werte zurück (f,t,Zxx) statt 2 (t,x).
# Betroffene Phasen: 03, 20, 23, 24 — alle mit "too many values to unpack".
def _safe_istft(
    Zxx,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    **kwargs,
):
    """§v10.303 Korrekte safe_istft-Monkey-Patch für Rückwärtskompatibilität.

    War vorher fälschlich auf _safe_stft gesetzt (STFT-Forward).
    """
    from scipy.signal import istft as _scipy_istft

    if kwargs.get("boundary") is True:
        kwargs["boundary"] = "zeros"
    # Clamp noverlap: 0 <= noverlap < min(nperseg, Zxx.shape[0])
    _eff_nperseg = min(nperseg, Zxx.shape[0] if hasattr(Zxx, "shape") else nperseg)
    if noverlap is None:
        _noverlap = _eff_nperseg // 2
    else:
        _noverlap = min(int(noverlap), max(0, _eff_nperseg - 1))
    try:
        return _scipy_istft(Zxx, fs=fs, window=window, nperseg=nperseg, noverlap=_noverlap, **kwargs)
    except ValueError as exc:
        logger.debug("§V6 scipy ISTFT Fallback aktiviert (ValueError): %s", exc)
        _noverlap = max(0, _eff_nperseg // 4)
        return _scipy_istft(Zxx, fs=fs, window="hann", nperseg=nperseg, noverlap=_noverlap)


_scipy_signal.safe_istft = _safe_istft
