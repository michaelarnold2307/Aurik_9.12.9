# Code-Schwachstellen-Report (Watchdog)

- Erzeugt: 2026-09-04T19:48:36.174926
- Geprüfte Dateien: 1548 (Dauer: 13.493s)
- Befunde gesamt: **11**
  - critical: 0
  - high: 0
  - medium: 0
  - low: 11
- Pro Regel: determinism_time_usage=10, print_in_production=1
- Unterdrückte Befunde (unter Schwelle/Kappung, bewusst sichtbar): determinism_time_usage=357, print_in_production=12
  (Schwellen: AST-Cap 3/Datei, time.time ≥ 2, print ≥ 3, Top-N 10, max_findings — unterdrückt heißt nicht: nicht vorhanden.)

## LOW (11)

- `backend/core/forensics/training/train_models.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `9 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/parallel/module_parallel.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `7 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_01_click_removal.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `18 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_12_wow_flutter_fix.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `7 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_19_de_esser.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `9 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_20_reverb_reduction.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `7 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_29_tape_hiss_reduction.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `8 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_31_speed_pitch_correction.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `7 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/phases/phase_34_mid_side_processing.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `7 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/unified_restorer_v3.py:1` — **determinism_time_usage** (§G5 (AGENTS.md §3 / copilot-instructions.md))
  - time.time() im Produktions-Code — Determinismus-Risiko
  - Evidenz: `13 Vorkommen von time.time()`
  - Empfehlung: Prüfen, ob Wall-Clock-Zeit in Entscheidungslogik einfließt; für Messungen time.monotonic()/perf_counter() verwenden. (§G5 Determinism)
- `backend/core/scripts/lint_peak_guard_conformity.py:1` — **print_in_production** (Logger-Pflicht (§III DSP, AGENTS.md §3))
  - print() in Produktions-Modul statt Logger
  - Evidenz: `7 Vorkommen von print()`
  - Empfehlung: Ausgaben über logging umleiten (Logger-Pflicht).
