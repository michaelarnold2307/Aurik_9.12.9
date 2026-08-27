# AERO-Challenger-Round — Bandbreiten-Extension 12 kHz → 48 kHz

- Datum: 2026-08-15 | Challenger: AERO (slp-rl/aero, MIT; Checkpoint `models/aero/checkpoint_12-48_hl256.th`)
- Status: **Kandidaten-Audio für 3 Items erzeugt**; Incumbent-Ausgaben und Hörurteile stehen aus.

## Ehrliche Aufgaben-Rahmung

Spec 04 positioniert AERO unter „Musik-NR (spezialisiert)“. Das ist eine
Kategorie-Unschärfe: AERO ist **Bandbreiten-Extension** (Super-Resolution im
Spektralbereich, 12 kHz → 48 kHz), kein Rauschunterdrücker. Diese Runde misst
AERO daher in seiner tatsächlichen Aufgabe — gegen den aktuellen
Bandbreiten-Extension-Pfad (Incumbent: FlashSR). Die NR-Frage gehört in eine
eigene Challenger-Runde (MP-SENet).

## Aufbau

- Quelle je Item: degradierte Golden-Set-Quelle, auf 12 kHz heruntergerechnet.
- Challenger: AERO-Ausgabe (48 kHz) — erzeugt für `vinyl_rock_1960s_clicks`,
  `shellac_jazz_1920s_crackle`, `digital_pop_2000s_clicks` unter
  `candidates/<item_id>_candidate.wav`.
- Incumbent: FlashSR-Ausgabe (48 kHz) — noch zu erzeugen und unter
  `incumbent/<item_id>_incumbent.wav` abzulegen.
- Bewertung: `scripts/challenger_round.py prepare` (Trial-Paket) und
  `decide` (ADOPT/REJECT) auf den Hörurteilen der Runde.

## Offen

1. FlashSR-Incumbent-Ausgaben für dieselben Items erzeugen.
2. Hörrunde durchführen (≥ 10 Hörer/Item, Verdicts im Gate-Schema).
3. `challenger_round.py decide` — erst dann ist eine Aufnahme-Entscheidung zulässig.

## Vermerke

- Plugin: `plugins/aero_plugin.py` (nicht in der Produktions-Routing-Pipeline
  verdrahtet — Aufnahme erst nach ADOPT).
- Vendor: `plugins/_vendor_aero/` (MIT, unverändert kopiert, LICENSE beiliegend).
- Determinismus: eval-Modus, fixe Chunking, Test `tests/unit/test_aero_plugin.py`
  prüft Bit-Identität zweier Läufe (§G5).
