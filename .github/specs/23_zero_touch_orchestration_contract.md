# Spec 23: Zero-Touch-Orchestrierungsvertrag — Der eine Dirigent, additiv verdrahtet

> **Version:** Aurik 10.0.23 · **Scope:** Systemische Stabilität
> **Status:** In Umsetzung
> **Erstellt:** 2026-08-16 · **Abgeschlossen:** —

## Prämisse

Aurik besitzt mehrere konkurrierende „Dirigenten“ (UnifiedRestorerV3, AurikDenker,
AurikOrchestrator, CompletionEngine). Der AurikOrchestrator (Pfeiler P1–P5) war nur
teilweise verdrahtet: P2/P4 wurden in UV3 bzw. Denker aufgerufen, aber `preflight()`
lief nirgends — dadurch war der P2-Watchdog in UV3 ein stiller No-Op
(`self.watchdog is None`) und das P3-Session-Memory wurde nie zum richtigen
Zeitpunkt persistiert. Ziel: EIN koordinierender Zero-Touch-Pfad, ohne dass eine
bestehende Komponente (PMGG, HPE-Gate, CompletionEngine, Denker-Stufen)
deaktiviert oder umgangen wird.

## Maßnahme

Additive Verdrahtung des AurikOrchestrators in den produktiven AurikDenker-Pfad:
P1/P5 laufen als Advisory-Stufe (Empfehlung + Watchdog-Initialisierung, kein
Phasenplan-Eingriff), P2 wird durch diese Initialisierung in UV3 scharf geschaltet,
P3 persistiert direkt nach dem finalen P4-Urteil. Alle bestehenden Gates bleiben
unverändert aktiv — keine dauerhafte Deaktivierung benötigter Komponenten.

### Implementierung

1. `backend/core/aurik_orchestrator.py`: `resolve()` bekommt `warnings`-Passthrough;
   `after_phase()` ohne Preflight loggt einmalig statt still zu no-oppen.
2. `denker/aurik_denker.py::_orchestriere`: `preflight()` wird vor der
   RestaurierDenker-Stufe mit echten Messwerten (Restorability, Transfer-Tiefe,
   bw_loss, SNR, Codec, PID-Phasenplan) aufgerufen; Ergebnis landet in
   `stage_notes["orchestrator_preflight"]`; bei passthrough oder
   konservativ/aggressiv-Konflikt nur Warnung — kein Eingriff in den Phasenplan.
3. `denker/aurik_denker.py`: `close_session()` direkt nach dem finalen
   `resolve()` (idempotent) — Session-Memory wird sofort persistiert.
4. Kontrakt-Tests (`tests/unit/test_aurik_orchestrator_wiring.py`) sichern
   P1–P5-Verhalten und den Verdrahtungsvertrag per AST-Check.

### Erfolgskriterium

- `preflight()` wird im Produktionspfad vor der Restaurierung aufgerufen (AST-Test grün).
- `AurikOrchestrator.watchdog` ist nach Preflight nicht `None` (Unit-Test).
- Alle 5 Pfeiler in einem Lauf aktiv; kein bestehendes Gate deaktiviert
  (bestehende Denker-/PMGG-Tests bleiben grün).
- `after_phase()` ohne Preflight = dokumentierter No-Op mit Debug-Log.

### Aufwand

3h | **Wohlklang-Wirkung:** Indirekt

### Risiken & Gegenmaßnahmen

| Risiko | Eintrittswkt. | Gegenmaßnahme |
|--------|---------------|---------------|
| Doppel-Gating mit PMGG/HPE (widersprüchliche Stopps) | Mittel | Stufe 1: P1/P5 rein advisory; P2 stoppt erst nach 3 konsekutiv schädlichen Phasen |
| Parallel laufende Songs teilen den Orchestrator-Singleton | Niedrig | No-Competing-Instances-Protokoll (AurikDenker) gilt weiter; Stufe 2: per-Song-Instanzen |
| Phasenplan-Kürzung entfernt benötigte Reparatur-Phasen | Mittel | Kein Pruning in Stufe 1; Enforcement erst nach MUSHRA-Kalibrierung (Spec 13) |
| Session-Memory-Wachstum | Niedrig | Cap 200 Einträge (bestehend) |

---

## Ziel-Matrix

| Ziel | Betroffen? | Wie? |
|------|-----------|------|
| Hörbarer Wohlklang | Nein | Indirekt: aktiver Watchdog und Session-Memory schützen künftige Läufe |
| Systemische Stabilität | Ja | EIN koordinierter Pfad; stiller No-Op wird sichtbar; P2 ist aktiv |
| Nachhaltige Wartbarkeit | Nein | Sekundär: Kontrakt-Tests verhindern erneutes Auseinanderdriften der Pfeiler |

> **Regel:** Eine Maßnahme adressiert GENAU EIN Ziel als primäres Ziel.
> Die anderen beiden dürfen als sekundäre Ziele profitieren, aber nicht
> im Fokus stehen. Keine Maßnahme adressiert alle drei gleichzeitig.

---

## Nachtrag (Rev. 2026-08-16): §v10.53-Invariante + AMRB-Verifikation

- **§v10.53-Fix**: §DENKER- und §3.0-CrossPhase-Modulation in
  `_profiled_phase_call` sind mit `not _strength_explicit` geschützt — der
  Consensus-Cap überschreibt explizite Stärke nicht mehr (zuvor überschrieb der
  modulglobale CPC-Singleton-Zustand explizite 0.99; Invarianten-Test +
  Singleton-Reset-Fixture).
- **AMRB-Frisch-Baseline mit aktivem Orchestrator** (16.08.2026): 82.5/100,
  2/2 Szenarien ≥ 80, RESOLVE „improved (72/100)“, Session-Memory persistiert.
  Laufzeit-Gate FAILED (3413 s > 1800 s) → Qualitäts-Roadmap Phase 1 (Budget-
  Disziplin; FeedbackChain-Formel seither auf die Normtabelle korrigiert).
- **Kontrakt-Ausbau**: Watchdog-No-Op-Warnung und warnings-Passthrough in
  resolve() dokumentiert (Spec 24 für die ML-Fallback-Architektur ergänzt).
