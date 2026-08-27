# SOTA-Migrationsplan — verbleibende Legacy-Modelle (Messstand 2026-08-15)

Ziel: Die nach der MIIPHER-Korrektur verbleibenden Legacy-Modelle auf den
gemessenen SOTA-Stand heben. Jeder Schritt mit Beleg, kein Training großer
Modelle nötig — die Nachfolger sind im Haus.

## 1. CREPE (2018) → FCPE (2023) — gemessen, umgesetzt (Rev. 2026-08-16)

**Messung** (`scripts/pitch_tracker_benchmark.py`, objektive F0-Ground-Truth,
15 Bedingungen: steady/vibrato/glide/low/high × clean/snr10/wow):

| Tracker | Cents-RMSE | GPE-Rate | Oktav-Fehler | Anmerkung |
|---|---|---|---|---|
| **FCPE** | **5,9** | **0,0 %** | 0,0 % | ONNX, 15/15 |
| CREPE | 24,6 | 38,9 % | 0,0 % | ONNX, 15/15 — 4× schlechter, 39 % Grobfehler |
| RMVPE | 25,9 | 17,7 % | 0,4 % | **nur 1/15 ONNX — 14/15 PESTO-DSP-Fallback!** |

**Umgesetzt (Rev. 2026-08-16):**
- `vocal_harmonic_decomp.py` (letzte CREPE-Primär-Verdrahtung) auf FCPE
  geroutet; Parameter `use_crepe` → `use_fcpe`.
- Bereits FCPE-primär (verifiziert): `hybrid_wow_flutter.py`,
  `hybrid_speed_pitch_ml.py`, `harmonic_preservation_guard.py`,
  `analysis_and_modules.py`, `phase_56_spectral_band_gap_repair.py`.
- CREPE verbleibt nur als FCPE-interner Delegate und in
  `_PHASE_REQUIRED_MODELS` (Legacy-Kompatibilität, Spec 04 Z. 1129).
- Registry-Zeile „Pitch-Hierarchie FCPE primär“ ist enforced (fail-closed).

## 2. RMVPE (2023) — ONNX-Lebenszyklus-Bug behoben (Rev. 2026-08-16)

**Messung (vorher)**: 1× `rmvpe_onnx`, 14× `pesto_dsp_fallback` — der §4.4-Pitch-
Tracker degradierte still auf DSP (§V6-Fund).

**Umgesetzt:**
1. **Session-Selbstheilung** (`rmvpe_plugin.py::_ensure_session`): Nach
   PLM-Unload (`_session = None`) lädt `analyze()` transparent neu — kein
   stiller PESTO-Downgrade mehr. Regressionstest:
   `test_sota_gap_closures.py::TestRmvpeSessionSelfHealing`.
2. **Race-Fix (lokale Session-Referenz)**: Der PLM-Evict setzt nur
   `self._session = None`; die Inferenz hält jetzt eine lokale Referenz,
   die mitten im Call nicht mehr annulliert werden kann — das war die
   eigentliche Ursache des `'NoneType' object has no attribute 'get_inputs'`
   (Benchmark-Fund Rev. 2026-08-16).
3. **phase_56-Attribut-Bug**: Die RMVPE-Stufe griff auf `voiced_prob`/`f0_hz`
   zu (CrepeResult-API) — RmvpeResult liefert `f0`/`voiced_flag`. Die Stufe
   fiel dadurch IMMER durch (AttributeError → stiller DSP-Downgrade).
   Korrigiert + Regressionstest `TestPhase56RmvpeTier`.

**Messstand nach Fix** (`models/pitch_benchmark_report_rev20260816b.json`):
15/15 Bedingungen `model_used="rmvpe_onnx"`, 0 Inferenzfehler (vorher 14/15
PESTO). Echte ONNX-Zahlen auf der synthetischen Suite: stark in hoher Lage
(1,2–6,1 Cents-RMSE, f1 0,99–1,0), schwach bei Glide/Vibrato (GPE bis 1,0)
und stimmhaft-arm bei Steady/Low (Salience-Gate 0,5) — FCPE bleibt damit
messbar der klare Primär (5,9 Cents-RMSE, 0 % GPE, f1 0,999).

## 3. Whisper-Denoiser → DFN/SGMSE+ — umgesetzt (Rev. 2026-08-16)

Methodisches Ad-hoc-Konstrukt ohne SOTA-Pendant. Denoising tragen in Aurik
DeepFilterNet/SGMSE+/OMLSA (Spec 04).

**Umgesetzt:**
- `phase_66_stem_targeted_nr.py::_get_dfn`: Whisper-Hop entfernt — Kette ist
  jetzt DFN (primär) → OMLSA-DSP-Fallback. Regressionstest
  `TestPhase66NoWhisperFallback`.
- `plugins/whisper_denoiser_plugin.py`: DEPRECATED-Banner (Rev. 2026-08-16),
  ladbar nur für A/B-Vergleiche (`music_model_flags.use_whisper_denoiser=False`).
- `ml_model_readiness.py`: Label „(deprecated, Rev. 2026-08-16)“.

## 4. DiffWave (2020) / HiFi-GAN (2020) → BigVGAN v2 (2024) — umgesetzt (Rev. 2026-08-16)

**Umgesetzt** (`backend/core/vocoder_chain.py`, neu auf Spec-04-[RELEASE_MUST]-Kaskade):

- Studio-2026:  Vocos 48 kHz nativ → BigVGAN-v2 → HiFi-GAN → PGHI
- Restoration:  BigVGAN-v2 → HiFi-GAN → PGHI (Vocos verboten, §1.4)
- PGHI-Endfall nutzt jetzt die reale §4.5-Schnittstelle `pghi_reconstruct`
  (der alte Import `pghi_istft` existierte nicht — der Endfall war defekt und
  gab still das Original zurück, §V6-Fund). Längen-Invariante gewahrt (§G5).
- DiffWave bleibt reine Inpainting-Aufgabe (phase_55), kein Vocoder-Tier.
- Regressionstests: `TestVocoderChainSpecCascade`.

## 5. MLMediumDetector → trainiertes flaches Artefakt

**Messung** (CV auf 56 kuratierten Items, `scripts/train_medium_classifier.py`):

| Aufgabe | Flacher Klassifikator | MediumDetector | Faktor |
|---|---|---|---|
| Material (6 Klassen) | 64,3 % | 10,7 % | 6× |
| Depth (4 Klassen) | 85,7 % | 51,8 % | 1,65× |

**Schritt**: Artefakt (`models/medium_shallow_v1.joblib`) als
Querprüfungs-Kanal im Golden-Gate — umgesetzt als `golden_set_tool.py
crosscheck` (deterministisch, vergleicht beide gegen die kuratierten Labels).
Produktions-Integration des Klassifikators bleibt Maintainer-Entscheidung.

## 8. Audit stille ML→DSP-Downgrades + aurik_restore-UV3-Angleich (Rev. 2026-08-16)

**Behoben (waren still bzw. spec-verletzend):**
- `aurik_restore.py`: ML-Plugins jetzt lazy geladen; alle ML→DSP-Fallbacks
  warnen mit Begründung (§V6), Dry-Passthrough nie still; UTMOS-Ausfall →
  Quality-Gate fail-closed. Kanonischer Vertrag (LEGACY_NON_RELEASE,
  NaN-Guard → tmp → os.replace) unverändert — Gate grün.
- `phase_03_denoise.py`: SGMSE+→OMLSA-Fallback warnt jetzt (§V6).
- `feedback_chain.py`: VERSA-Scorer-Ladefehler → PQS/RMS-Ersatzpfad warnt (§V6).
- `stem_level_restorer.py`: SOTA-Router-Ablehnung → DSP-Bandpass warnt (§V6).
- `hybrid_wow_flutter.py` + `hybrid_speed_pitch_ml.py`: CREPE als Tier-4
  entfernt (Spec 04 Z. 1129) — Kaskade ist jetzt FCPE → RMVPE → PESTO → pYIN
  mit §V6-Warnung am DSP-Endfall.

**Geprüft und konform:**
- `phase_49_advanced_dereverb.py` (§V6-Warnung vorhanden),
  `vocal_quality_index.py` Resemblyzer (§V6-Warnung vorhanden),
  `stem_level_restorer.py` Exception-Pfad (§V6),
  `phase_56`/RMVPE/`phase_55` (§V6 via Call-Site).
- `perceptual_validator.py`: OSError→DSP-Fallback bewusst auf DEBUG
  (erwarteter Deployment-Zustand „Modell nicht gebündelt", Kommentar im
  Code) — dokumentierte Ausnahme.

**Regressionstests:** `tests/unit/test_aurik_restore_uv3_alignment.py` (7 Tests),
`tests/unit/test_fallback_auditor_wiring.py` (10 Tests).

**FallbackAuditor-Verdrahtung (§v10.17, Rev. 2026-08-16):**
- `FallbackAuditor.reset()` ergänzt; `AurikDenker.restauriere()` setzt den
  Auditor pro Song zurück (§V8/§G1) — ohne das hätten session-global
  akkumulierte Events ab 8 Stück fälschlich Exporte blockiert.
- `pre_export_validator` gibt den konsolidierten Bericht aus („Aurik:
  DEGRADED — …") und blockiert weiterhin ab Kaskaden-Limit.
- Alle sechs behobenen Sites registrieren jetzt explizit Events
  (phase_03→OMLSA, FeedbackChain→PQS/RMS, StemLevelRestorer→DSP-Bandpass,
  beide Hybrid-Pitch-Kaskaden→pYIN, aurik_restore→Dry-Passthrough) —
  defensiv gekapselt (nie blockierend).

## Governance

- Alle Produktions-Eingriffe (Routing, Plugins) erfordern Maintainer-Sign-off
  (AGENTS.md §3); die Messungen und Pläne hier sind die Evidenz dafür.
- Der Pitch-Benchmark ist objektiv reproduzierbar (§G5, fixe Seeds) — er wird
  zum Challenger-Gate für jeden künftigen Pitch-Tracker-Kandidaten.

## 6. DSP-Benchmark — veraltete DSPs gemessen (scripts/dsp_benchmark.py)

| Methode | ref_snr_mean | edge_peak_ratio_max | Befund |
|---|---|---|---|
| spectral_gating_ref (korrigiert) | 0,6 dB | 1,3 | Referenz ohne Kantenspike |
| wiener_ref (korrigiert) | 1,2 dB | 0,8 | Referenz ohne Kantenspike |
| aurik_spectral_gating (operativ) | 0,6 dB | 1,0 | boundary-Paar korrigiert ✅ |
| **aurik_omlsa (operativ)** | **+5,2 dB** | **1,0** | **boundary-Fix angewendet (Rev. 2026-08-16) ✅** |

- **Fix angewendet (Rev. 2026-08-16)**: `boundary`-Paarung der scipy-STFT/ISTFT
  im OMLSA-Fallback und im operativen Spectral-Gating korrigiert.
  Gemessen: edge_peak_ratio 393,0 → 1,0 und ref_snr −3,4 dB → +5,2 dB
  (+8,6 dB) — OMLSA ist damit der zweitbeste gemessene Pfad nach Banquet.
- Der ehemalige `xfail`-Test ist ein permanenter Regressionstest
  (`tests/unit/test_dsp_benchmark.py::test_aurik_omlsa_edge_artifact_bounded`).

## 7. Decrackle-Pfad — Banquet gemessen + B6-Gate verifiziert

- **Banquet ist objektiv der beste gemessene Pfad**: ref_snr 5,7 dB (Legacy-DSPs
  0,6/−3,4 dB), LSD 86,6 (niedrigste), Kanten sauber (edge_ratio 1,0) —
  gemessen in `scripts/dsp_benchmark.py` (aurik_banquet).
- **B6-Gate ist runtime-enforced und Chain-aware**: `tests/unit/test_banquet_gate.py`
  (Digital → nie Banquet; Vinyl → ML-Pfad; Vinyl in der Kette → ML-Pfad).
- **B11 fehlt**: HF-Rauschfloor-Check (v10.900) ist in phase_09 nicht
  implementiert — Nachrüstung empfohlen.
- **RBME-Net (Spec-04-Decrackle-Primary) hat kein Plugin** und ist unter dem
  Namen nicht auf GitHub auffindbar — Beschaffung offen (siehe
  `audit/listening_study/round_2026-08-15/DECRACKLE_RBME_ROUND.md`).
