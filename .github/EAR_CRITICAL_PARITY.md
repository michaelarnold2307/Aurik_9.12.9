# Ear-Critical-Paritäts-Registry — Ohr-Schutzregeln müssen Code und Test haben

Dieser Bestand ist die fail-closed-Brücke zwischen den normativen Ohr-Schutzregeln
und dem Code. `scripts/ear_critical_parity_check.py` (Pre-Commit-Hook
`aurik-ear-parity`) erzwingt:

- **enforced**: Implementierungs- und Test-Datei existieren und enthalten das
  Prüf-Token. Damit ist die §0j-Klasse von Fehlern ausgeschlossen: eine Regel,
  die spezifiziert, aber im Code wirkungslos oder ungetestet ist.
- **deferred**: nur mit expliziter Begründung — kein stiller Skip.
- Mindestens 8 enforced-Zeilen (Schutz gegen versehentliches Leeren).

Spalten: Regel-ID | Quelle | Ohr-Grund | Prüf-Token | Implementierung | Test | Status | Defer-Begründung

| Regel-ID | Quelle | Ohr-Grund | Prüf-Token | Implementierung | Test | Status | Defer-Begründung |
|---|---|---|---|---|---|---|---|
| §0j | .github/instructions/dsp.instructions.md §0j | Bias verhindert, dass Harmonik als Rauschen abgetragen wird | §0j | plugins/deepfilternet_v3_ii_plugin.py | tests/test_deepfilternet_v3_plugin.py | enforced | — |
| §G71 | .github/copilot-instructions.md §G71 | Wirksamkeitsgarantie strength×wet ≥ 0.15 — Phasen bleiben hörbar wirksam | §G71 | backend/core/signal_flow_tracer.py | tests/unit/test_restoration_memory.py | enforced | — |
| §V6 | .github/copilot-instructions.md §V6 | ML→DSP-Fallbacks dürfen nie still degradieren | §V6 | backend/quality_metrics_manager.py | tests/test_phase_01_ml_hybrid.py | enforced | — |
| §0a | .github/copilot-instructions.md §0a | NaN/Inf und verbotene Phasen = hörbare Klicks/Artefakte | §0a | backend/core/adaptive_phase_rescheduler.py | tests/normative/test_section_0a_restoration_guard.py | enforced | — |
| Verbotene Phasen 21/35/42 | .github/copilot-instructions.md §0a | Exciter/MBC/Vocal-Enhancement im Restoration-Modus gefährden Wohlklang | phase_21_exciter | backend/core/adaptive_phase_rescheduler.py | tests/test_autonomous_restoration_engine.py | enforced | — |
| §V5 Dither | .github/copilot-instructions.md §V5 | Quantisierungsrauschen bei <32 bit hörbar | POW-r | backend/exporter.py | tests/unit/test_exporter_dither.py | enforced | — |
| Soft-Knee | .github/instructions/dsp.instructions.md §III | Hard-Clamp erzeugt hörbare Clipping-Artefakte | soft_knee | backend/core/phases/phase_19_de_esser.py | tests/unit/test_phase42_psychoacoustics.py | enforced | — |
| §7.7/§2.29a Inference-Caching | .github/specs/06_phases_system.md §7.7 | Retries ohne Re-Inferenz: deterministisch, kein doppeltes Artefaktrisiko | §2.29a | backend/core/per_phase_musical_goals_gate.py | tests/normative/test_full_pipeline_determinism.py | enforced | — |
| HallucinationGuard | .github/specs/02_pipeline_architecture.md | Halluzinierter Inhalt ist das deutlichste hörbare Artefakt | HallucinationGuard | backend/core/dsp/hallucination_guard.py | tests/unit/test_guard_self_test.py | enforced | — |
| HNR-Guard | .github/instructions/dsp.instructions.md | Vokal-Dumpfheit/Stimmverlust nach NR | hnr_guard | backend/core/vocal_no_harm_gate.py | tests/unit/test_miipher_plugin.py | enforced | — |
| §2.70 nur Erfolge | .github/instructions/pipeline.instructions.md §2.70 | Lernen aus Fehlern verschlechtert künftige Restaurierungen hörbar | restoration_memory | backend/core/restoration_memory.py | tests/unit/test_restoration_memory.py | enforced | — |
| Golden-Set-Gate | docs/guides/GOLDEN_LISTENING_SET.md | misst das Ohr direkt; fail-closed ohne Hörurteile | non_inferiority_gate | scripts/non_inferiority_gate.py | tests/unit/test_non_inferiority_gate.py | enforced | — |
| V01/V08 ERROR-Gate | AGENTS.md §3 | dokumentierte Production-Verstöße; ERROR-Gate-Test skipped | — | — | — | deferred | Maintainer-Sign-off nötig (AGENTS.md §3: nicht „nebenbei“ fixen) |
| SNR<10/Codec-Task-Split | .github/specs/04_dsp_standards.md (Rev. 2026-08-15) | verhindert Task-Vertragsbruch: Deep-Noise-Vokal → SGMSE+/DFN-Legacy-Router; Codec-Degradation → DiT | miipher_deepfilternet_v3_ii | plugins/miipher_plugin.py | tests/unit/test_miipher_plugin.py | enforced | — |
| Pitch-Hierarchie FCPE primär | .github/specs/04_dsp_standards.md (Pitch-Zeile) | CREPE (2018) ist spec-verworfen und gemessen 4× schlechter (39 % GPE) — FCPE primär | FCPE | backend/core/unified_restorer_v3.py | tests/normative/test_hybrid_release_mode.py | enforced | — |
| Decrackle-ML verifiziert & domänenkonform | .github/specs/04_dsp_standards.md (Rev. 2026-08-16) | verhindert erfundene Modell-Zitate (RBME-Net existiert nicht) und Sprach-Modelle im Musik-Decrackle-Pfad | banquet | backend/core/phases/phase_09_crackle_removal.py | tests/unit/test_banquet_gate.py | enforced | — |
| v10.22 Modell-Gating | .github/specs/v10.22_model_orchestration.md | per-Modell-Bypass bei Hörverschlechterung | — | — | — | deferred | Nicht implementiert; großer Eingriff, Maintainer-Sign-off |

## Hinweis: obsolete Regel-Dokumente

`.github/VERBOTE.md` und `.github/GEBOTEN.md` tragen Veraltet-/Referenz-Banner
und sind nicht normativ (AGENTS.md §6); historische Pfad-Referenzen liegen in
`.github/ID_REGISTRY.md`. Sie werden nicht durch diesen Check geprüft.
