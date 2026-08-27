"""SweetSpot-Baseline-Relativität: geerbte Quell-Artefakte nicht bestrafen (Spec 24).

Root-Fix-Regression 2026-08-16: find_sweet_spot maß Kammfilter/Kompression/
Maskierung ohne Original-Baseline — Vinyl-Rotation/mp3_low-Erbe wurde der
Pipeline angelastet (Befund: comb=0.10, SweetSpot-Loop 3× leer gedreht).
"""

from __future__ import annotations

import numpy as np

SR = 48000


def _comb_signal(n_samples: int, delay: int = 400, gain: float = 0.9, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(np.float32)
    x /= np.abs(x).max() + 1e-6
    out = x.copy()
    out[delay:] += gain * x[:-delay]
    return out


def test_inherited_comb_is_not_penalized() -> None:
    """Identisches Signal als Referenz UND Ergebnis: keine Comb-Warnung,
    Score >= Score ohne Referenz (der alte Abzug entfällt)."""
    from backend.core.sweet_spot_optimizer import find_sweet_spot

    sig = _comb_signal(SR * 2)
    with_ref = find_sweet_spot(sig, SR, reference=sig)
    without_ref = find_sweet_spot(sig, SR)
    assert not any("Kamm" in w for w in with_ref.warnings), (
        f"Geerbter Kammfilter darf nicht warnen: {with_ref.warnings}"
    )
    assert with_ref.score >= without_ref.score, (
        f"Baseline darf den Score nicht senken: {with_ref.score:.3f} < {without_ref.score:.3f}"
    )


def test_without_reference_comb_still_warns() -> None:
    """Ohne Referenz bleibt das alte Verhalten erhalten (Comb-Warnung)."""
    from backend.core.sweet_spot_optimizer import find_sweet_spot

    sig = _comb_signal(SR * 2)
    result = find_sweet_spot(sig, SR)
    assert any("Kamm" in w for w in result.warnings), "Ohne Referenz muss Comb warnen"


def test_short_reference_degrades_gracefully() -> None:
    """Kurze Referenz (< 4096 Samples) darf nicht crashen — Vererbung entfällt."""
    from backend.core.sweet_spot_optimizer import find_sweet_spot

    sig = _comb_signal(SR * 2)
    short_ref = sig[:2048].astype(np.float32)
    result = find_sweet_spot(sig, SR, reference=short_ref)
    assert result.score >= 0.0  # kein Crash, Ergebnis gültig


def test_clean_reference_path_does_not_crash() -> None:
    """Reference=None (alle bisherigen Aufrufer) bleibt abwärtskompatibel."""
    from backend.core.sweet_spot_optimizer import find_sweet_spot

    sig = _comb_signal(SR)
    result = find_sweet_spot(sig, SR, reference=None)
    assert 0.0 <= result.score <= 1.0
