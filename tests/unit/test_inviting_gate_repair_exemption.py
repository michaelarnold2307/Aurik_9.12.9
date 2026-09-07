"""Einladungs-Gate: Sharpness-Sprung-Exemption an Reparaturstellen (Punkt 4).

Produktionsbefund 2026-09-07: sharpness_jump=0.562acum → Einladungs-Gate
NICHT BESTANDEN, obwohl der Sprung aus einer lokalisierten Reparatur stammt
(beabsichtigte HF-Änderung). Das Jump-Kriterium (Hörordnung §6) zielt auf
unbeabsichtigte Diskontinuitäten — Sprünge, deren Fenster ein Reparatur-
Fenster überlappen, werden ausgenommen (kein neuer Schwellwert, das
0.2-acum-Limit bleibt normativ).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.inviting_sound_gate import check_inviting_gate


def _tone_segments(sr: int, seconds_per_window: int, sharpness_jump: float) -> np.ndarray:
    """Synthetisches Signal: mehrere Fenster; ein HF-Sprung zwischen Fenster 1→2.

    Sharpness wird über HF-Energie-Anteil angetrieben — Fenster 0 hat wenig HF,
    Fenster 1+ haben deutlich mehr (Sprung) oder gleichmäßig (kein Sprung).
    """
    n_win = 4
    seg = np.zeros(n_win * sr * seconds_per_window, dtype=np.float32)
    for wi in range(n_win):
        s = wi * sr * seconds_per_window
        e = s + sr * seconds_per_window
        t = np.arange(e - s) / sr
        # Grundton + HF-Anteil (HF steuert Sharpness-Proxy)
        base = 0.5 * np.sin(2 * np.pi * 220 * t)
        hf_amp = 0.02 if wi == 0 else 0.02 + sharpness_jump * 0.4
        hf = hf_amp * np.sin(2 * np.pi * 6000 * t)
        seg[s:e] = (base + hf).astype(np.float32)
    return seg


def test_jump_at_repair_window_is_exempted() -> None:
    sr = 48_000
    audio = _tone_segments(sr, 6, sharpness_jump=1.0)
    # Reparatur-Fenster deckt die Fenstergrenze zwischen Window 0 und 1 (6–12 s)
    repair_windows = [(5.0, 13.0)]

    res = check_inviting_gate(audio, sr, repair_windows=repair_windows)

    assert res.details.get("exempted_jumps", 0) >= 1
    assert res.sharpness_jump_max <= 0.20 + 1e-6, (
        f"effektiver Sprung muss unter Limit bleiben, war {res.sharpness_jump_max:.3f}"
    )
    assert res.passed or "sharpness_jump" not in str(res.details.get("failures"))


def test_jump_without_repair_window_fails_gate() -> None:
    sr = 48_000
    audio = _tone_segments(sr, 6, sharpness_jump=1.0)

    res = check_inviting_gate(audio, sr, repair_windows=None)

    assert res.details.get("exempted_jumps", 0) == 0
    assert res.sharpness_jump_max > 0.20
    assert not res.passed
    assert any("sharpness_jump" in f for f in res.details.get("failures", []))


def test_no_repair_windows_noop() -> None:
    """Ohne repair_windows bleibt das Verhalten exakt wie vorher."""
    sr = 48_000
    audio = _tone_segments(sr, 6, sharpness_jump=0.0)
    res = check_inviting_gate(audio, sr)
    assert res.sharpness_jump_max == pytest.approx(res.details.get("sharpness_jump_raw_max", -1.0), abs=0.01)
    assert res.details.get("exempted_jumps", 0) == 0
