"""§Hörordnung-Pre-Filter Regressionstests (Punkt 3, Produktionsbefund 2026-09-07).

FeedbackChain erzeugte Kandidaten, die brillanz (Stufe 4) auf Kosten von
waerme/natuerlichkeit (Stufe 1/2) verbesserten — der GPP-Abbruch kam erst
NACH der Messung (11–23 Verstöße pro Audit). Der Pre-Filter überspringt
solche Kandidaten VOR der Messung, solange ein niederrangiges Goal im
Defizit ist (lexikografische Ordnung, hoerordnung.instructions.md §5).
"""

from __future__ import annotations

from backend.core.feedback_chain import FeedbackChain


def _phases() -> list:
    """Fake-Phasenliste wie im FC-Loop: (num, fn, kw)."""
    return [
        (7, lambda a, sr, **kw: a, {}),   # Harmonic Restoration → waerme/natuerlichkeit
        (16, lambda a, sr, **kw: a, {}),  # Final EQ → brillanz/waerme
        (17, lambda a, sr, **kw: a, {}),  # Mastering polish → brillanz/groove
        (48, lambda a, sr, **kw: a, {}),  # Stereo imaging → spatial_depth/separation_fidelity
        (99, lambda a, sr, **kw: a, {}),  # unbekannt → konservativ behalten
    ]


def test_filter_drops_high_tier_boost_when_low_tier_deficit() -> None:
    """waerme (Stufe 2) im Defizit → phase_48 (Stufe 3/4, kein Defizit-Ziel) fällt."""
    fc = FeedbackChain()
    fc.adaptive_goal_thresholds = {
        "waerme": 0.8,
        "brillanz": 0.7,
        "natuerlichkeit": 0.9,
    }
    scores = {"waerme": 0.6, "brillanz": 0.5, "natuerlichkeit": 0.95}

    kept = fc._filter_phases_by_hoerordnung_tiers(_phases(), scores)

    kept_ids = [int(pid) for pid, _, _ in kept]
    assert 48 not in kept_ids, "Stereo-Imaging (höhere Stufe, kein Defizit-Ziel) muss übersprungen werden"
    # Phase 17 (brillanz) bedient ein Defizit-Goal direkt → bleibt
    assert 17 in kept_ids
    # Phase 7 adressiert Stufe 1/2 → bleibt
    assert 7 in kept_ids
    # Unbekannte Phase bleibt (konservativ)
    assert 99 in kept_ids


def test_filter_keeps_all_when_no_deficits() -> None:
    fc = FeedbackChain()
    fc.adaptive_goal_thresholds = {"waerme": 0.5, "brillanz": 0.5}
    scores = {"waerme": 0.9, "brillanz": 0.9}

    kept = fc._filter_phases_by_hoerordnung_tiers(_phases(), scores)
    assert len(kept) == len(_phases())


def test_filter_noop_without_thresholds() -> None:
    fc = FeedbackChain()
    kept = fc._filter_phases_by_hoerordnung_tiers(_phases(), {"waerme": 0.1})
    assert len(kept) == len(_phases())


def test_filter_noop_on_empty_scores() -> None:
    fc = FeedbackChain()
    fc.adaptive_goal_thresholds = {"waerme": 0.8}
    kept = fc._filter_phases_by_hoerordnung_tiers(_phases(), {})
    assert len(kept) == len(_phases())


def test_filter_keeps_deficit_targeting_high_tier() -> None:
    """brillanz (Stufe 4) ist selbst im Defizit → brillanz-Phasen bleiben (GPP entscheidet)."""
    fc = FeedbackChain()
    fc.adaptive_goal_thresholds = {"brillanz": 0.8}
    scores = {"brillanz": 0.4}

    kept = fc._filter_phases_by_hoerordnung_tiers(_phases(), scores)
    kept_ids = [int(pid) for pid, _, _ in kept]
    assert 17 in kept_ids
    assert 55 not in kept_ids  # phase_55 nicht in der Fake-Liste — nur zur Doku
    # phase_16 bleibt (brillanz-Defizit wird bedient)
    assert 16 in kept_ids
