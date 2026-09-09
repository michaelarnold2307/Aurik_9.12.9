# Tiefenanalyse: Aurik vs. Stand der Wissenschaft und Technik (2026-09)

> **Frage**: Entspricht der Aufbau von Aurik dem fortschrittlichsten, qualitativ
> hochwertigsten und performantesten Restaurierungssystem, das nach dem heutigen
> Stand der wissenschaftlichen Erkenntnisse möglich ist?
>
> **Methode**: Inventar des realen Aurik-Stands (Modelle, Phasen, Metriken,
> Performance — belegt mit Dateipfaden) gegen die etablierte wissenschaftliche
> Literatur zum Audio-Restaurierungs-SOTA. Ehrliche Kennzeichnung: Die
> Literaturkenntnis reicht bis zum Wissensstand des Analysemodells; brandneue
> 2026-Veröffentlichungen können Lücken erzeugen, die hier als solche markiert sind.

---

## 1. Kurz-Verdikt

**Aurik ist in mehreren Kern-Domänen auf dem Niveau des publizierten SOTA**
(Quellentrennung: Mel-Band-RoFormer/BS-RoFormer; Bandbreiten-Erweiterung: NVSR;
Diffusions-Inpainting: SGMSE+/Diffusion-Phase 55; perceptuelle Metriken:
MERT-MUSHRA-Proxy, DNSMOS, ViSQOL) und übertrifft praktisch alle bekannten
Systeme in **Determinismus (bit-exakte Reproduzierbarkeit), psychoakustisch
maskierungsgetriebener Eingriffssteuerung (Hörordnung/ERB-JND) und
wissenschaftlicher Hör-Evaluation (Blindtest-Framework + GO/NO-GO-Protokoll)**.

**Aber „das höchstmögliche System“ ist es noch nicht.** Vier messbare Lücken:
(1) Declipping ist klassisch (PCHIP-Interpolation), nicht neuronal (A-SPADE-Klasse);
(2) der Denoiser-Kern ist DeepFilterNet3 (kausal, 15 MB) — nicht-kausale
Vollband-Modelle (TF-GridNet/MP-SENet-Klasse) erreichen höhere DNSMOS-Werte;
(3) Wow/Flutter-Korrektur ist klassisch pitch-tracking-basiert (kein neuronales
Modell — das ist aber auch im Forschungsstand kaum besser gelöst);
(4) die 30-s-Chunk-Pipeline verhindert Song-globale Entscheidungen (siehe
Tiefenanalyse Restaurierungsablauf).

---

## 2. Domänen-Matrix: SOTA-Methode vs. Aurik

| Domäne | Wissenschaftlicher SOTA (etabliert) | Aurik-Implementierung | Urteil |
|---|---|---|---|
| **Musik-Quellentrennung** | BS-RoFormer (Lu et al., AAAI 2024), Mel-Band-RoFormer (2024), HTDemucs (Rouard et al. 2023), BSRNN (Luo & Yu 2023) | `plugins/bs_roformer_plugin.py` + MelBandRoformer-Top-Stufe (~860 MB, Tier-2), Demucs-Plugin, MDX23C-Fallback | ✅ **SOTA-Niveau** |
| **Breitband-Rauschunterdrückung (Musik)** | TF-GridNet (Wang et al., Interspeech 2023), MP-SENet (Lu et al., ICASSP 2024) — nicht-kausal, Vollband | DeepFilterNet v3.II finetuned (~15 MB, kausal) + MelBandRoformer-Enhancement + OMLSA-DSP-Fallback (`backend/core/noise_reduction.py`) | ⚠️ **Solide, aber nicht Spitze**: kausales 15-MB-Modell liegt unter nicht-kausalen Vollband-Modellen; MelBandRoformer gleicht teilweise aus |
| **Declipping** | A-SPADE (Gaultier et al., 2023), DCUNet-artige lernende Declipper — übertreffen AR/PCHIP deutlich | `phase_07_declipper.py`: PCHIP-Interpolation (Déger & Duhamel 2002; Välimäki et al. 2016) + Soft-Saturation-Erkennung | ❌ **Klassisch — größte Einzel-Lücke** |
| **Bandbreiten-Erweiterung** | NVSR (Liu et al., 2024), AudioSR (2024), Nu-wave-Familie | `backend/core/hybrid/hybrid_nvsr.py` + `plugins/nvsr_plugin.py`, integriert in `phase_06_frequency_restoration.py` (8–16-kHz-Gap Vinyl/MP3-128k) | ✅ **SOTA-Niveau** |
| **Dereverberation** | WPE (Nakatani et al. 2010) als klassische Basis; MISO; TF-GridNet-basiert; SGMSE+-Diffusion | Phase 20 + `plugins/sgmse_plugin.py` (SGMSE+, ~12 MB) — Diffusions-Dereverb/Denoise | ✅ **SOTA-Niveau** (Diffusion) |
| **Diffusions-Inpainting** | SGMSE+ (Richter et al., 2023), StoRM (2024) | `phase_55_diffusion_inpainting.py` | ✅ **SOTA-Niveau** |
| **De-Essing** | klassisch: spektral/phase-aware; SOTA weiterhin DSP-dominiert (wenig publizierte neuronale De-Esser) | Phase 19 + Phase 43 (ML-De-Esser), gender-aware (pYIN+LPC-Formanten+Contralto, Spec 19) | ✅ **SOTA-Niveau** (DSP-Spitze) |
| **Clicks/Crackle/Pops** | Klassisch dominant (Interpolation, AR-Modelle; neuere NN-Click-Detektoren existieren, aber kein klar etablierter Gewinner) | Phasen 01/09/27 (Median, kammfilter-geführt, adaptive Crossfades) | ✅ **SOTA-Niveau** |
| **Bandhiss** | Spektrale Subtraktion / adaptive Filter (LMS) / DFN-artige NN | Phase 29 (LMS-Adaptivfilter) + DeepFilterNet-Stufe | ✅ **SOTA-Niveau** |
| **Wow/Flutter** | Forschung dünn; Industrie-SOTA: Celemony Capstan (pitch-tracking-basiert, proprietär) | Phase 12 (Pitch-Tracking-Korrektur, Polyphonic Circuit-Breaker) | ✅/⚠️ **Branchenniveau; neuronale Forschung existiert kaum** |
| **Pitch/F0-Schätzung** | CREPE (Kim et al. 2018), pYIN (Mauch & Dixon 2014), RMVPE (2023), FCPE (2024) | Kaskade FCPE→RMVPE→PESTO→pYIN (`bridge.py:1558`) | ✅ **SOTA-Niveau** |
| **Stimmklang-Analyse** | WORLD-Vocoder (Morise et al. 2016), LPC/Burg-Formanten | WORLD-Quervalidierung + Burg-LPC + VQI (`musical_goals/vocal_quality_index.py`) | ✅ **SOTA-Niveau** |

## 3. Metrik- und Psychoakustik-Schicht

| Ebene | Wissenschaftlicher SOTA | Aurik | Urteil |
|---|---|---|---|
| Objektive Metriken | PESQ (ITU-T P.862), STOI (Taal et al. 2011), DNSMOS (Reddy et al. 2021), ViSQOL (Chinen et al. 2020) | PESQ/STOI (`blindtest_framework.py`, `psychoacoustic_metrics.py`), DNSMOS (`dsp/quality_predictors.py`), ViSQOL (`enhanced_metrics.py`) | ✅ |
| Perzeptuelle Proxies | MERT-basierte Qualitäts-Proxies (Li et al. 2023), Lernbare DNSMOS (GSEP, 2024) | **MERT-MUSHRA-Proxy** (`mert_mushra_proxy.py`) + MUSHRA-Proxy (`mushra_proxy.py`) | ✅ **Vorreiter** |
| Hör-Evaluation | MUSHRA (ITU-R BS.1534), Blindtests als Goldstandard | **Blindtest-Framework** (`blind_test_framework.py`, Scorer) + `GO_NO_GO_DECISION_PROTOCOL.md` | ✅ **Methodisch vorbildlich** |
| Psychoakustik | Zwicker/Fastl-Lautheit (ISO 532), ERB/Gammatone-Maskierungsmodelle, PEMO-Q | ERB-Filterbank + Maskierungsschwellen pro Phase (`erb_auditory_masking.py`), JND-Gates (`perceptual_tuning.py`), Hörordnung (lexikografische Wohlklang-Ordnung) | ✅ **Einzigartig tief** |
| Loudness | EBU R128 / LUFS | Phase 40 (LUFS-Normierung), True-Peak-Limiter (Phase 47) | ✅ |

**Hervorhebung**: Die „Hörordnung“ (Maskierungsschwelle statt Mess-Null; Eingriff nur wenn
hörbar; JND-basierte Stärke-Entscheidungen über `global_scalar`) setzt die
Zwicker/Fastl-Maskierungstheorie konsequenter um als jedes uns bekannte
kommerzielle Restaurierungswerkzeug — dort dominieren reine Messwert-Ziele.

## 4. Performance- und Engineering-Schicht

| Aspekt | SOTA-Praxis | Aurik | Urteil |
|---|---|---|---|
| Inferenz | ONNX-Runtime + GPU (CUDA/ROCm), Quantisierung | ONNX-Sessions, `ml_device_manager.py` (ROCm), RAM-Budgets, LRU-Eviction (`plugin_lifecycle_manager`) | ✅ |
| Speicher | Batch/Chunked-Inferenz | Chunked-Streaming RAM O(1) (30/60-s-Chunks) | ✅ speicherseitig; ⚠️ qualitätsseitig (Song-Kontext, eigene Analyse) |
| Real-Time | RT-Faktor-Budgets, progressive Modi | RT-Budget-Kaskade (Denker, Spec §9.5), Quality/Maximum-Modi | ✅ |
| **Determinismus** | In der ML-Forschung praktisch inexistent | **Bit-exakte Reproduzierbarkeit (§G5)**: Seeds pro Session, kein `time.time()` in Entscheidungen, B3-Chunk-Determinismus, Referenzlauf-Vertrag | ✅ **Alleinstellungsmerkmal** |
| Fehlerkultur | — | Silent-Failure-Verbot (§V6), Fallback-Auditor, NaN-Guards pro Phase, Rollback-Gates | ✅ |

## 5. Wo Aurik die Wissenschaft übertrifft (Differenzierer)

1. **Deterministische ML-Pipeline** — reproduzierbare Referenzläufe (Bit-Identität)
   sind im Forschungsstand unbekannt; Aurik macht sie zum Release-Gate.
2. **Maskierungsgetriebene Eingriffssteuerung** — Phasen entscheiden über
   Hörbarkeits-Schwellen (ERB-Maskierung, JND), nicht über Mess-Nullen.
3. **Kausales Defekt-Modell** (62 DefectTypes, CausalGraph, per-Defekt-Phasen)
   statt generischem „Denoise-Alles“.
4. **Wissenschaftliches Evaluations-Regime**: Blindtest-Framework, MERT-MUSHRA-
   Proxy, Hörordnung-Kalibrierung, GO/NO-GO — übertrifft die übliche
   PESQ/DNSMOS-Einzelzahl-Kultur.
5. **FeedbackChain** (MOS-geführte Nachjustierung) + m1b-Hörbarkeits-Retry —
   Pipeline-Level-Äquivalent zu MetricGAN-Philosophie, aber gate-gebunden und deterministisch.

## 6. Prioritärer Rückstand (ehrliche Lückenliste)

| # | Lücke | Empfohlene Maßnahme | Erwarteter Gewinn |
|---|---|---|---|
| 1 | **Neuronaler Declipper fehlt** (nur PCHIP) | A-SPADE-Klasse (oder DCUNet-artig) als Phase-07-Upgrade; DSP-Fallback behalten | Größter Einzelgewinn bei stark geclipptem Material (Pop der 90er–2000er) |
| 2 | **Denoiser nicht-kausal unterdimensioniert** | TF-GridNet/MP-SENet-Klasse als optionaler Quality-Mode (nicht-kausal, Vollband); DeepFilterNet bleibt Echtzeit-/Fallback-Pfad | +DNSMOS, weniger Musik-Artefakte |
| 3 | **Ganzsong-Pipeline statt 30-s-Chunks** | Geplanter Refactor (Tiefenanalyse Restaurierungsablauf §6) | Song-globale Loudness/Formant/Gender/Gate-Entscheidungen |
| 4 | **Wow/Flutter neuronal** | Forschungslücke: kein etabliertes öffentliches Modell; Capstan-Klasse + eigene Pitch-Tracking-Verbesserung bleibt Stand | Begrenzt — Industriegeheimnisse, wenig publizierte Gewinne |
| 5 | **Per-Stem-Restaurierung als Standard** (statt Env-Opt-in) | `source_aware_restorer` nach Referenzlauf-Verifikation zum Standard für Vocal-lastige Tracks machen | Bessere Stimmbehandlung ohne Instrumenten-Artefakte |

## 7. Fazit

Aurik kombiniert bereits heute **SOTA-Modelle (Mel-Band-RoFormer, NVSR, SGMSE+,
DeepFilterNet3, MERT-Proxy) mit einer wissenschaftlich tieferen psychoakustischen
Steuerung und einer in der Branche einzigartigen Deterministik- und Evaluations-
Kultur**. Damit ist es in der Breite (65+ Phasen, 62 Defekttypen, Kausalmodell,
Hörordnung) mindestens auf Augenhöhe mit den besten bekannten Systemen und in
mehreren Dimensionen darüber.

„Das höchstmögliche System“ ist es noch nicht — die **Lücken 1–3** (neuronaler
Declipper, nicht-kausaler Denoiser, Ganzsong-Pipeline) sind klar benennbar und
umsetzbar; danach wäre die verbleibende Differenz zum theoretischen Maximum im
Wesentlichen Modell-Finetuning- und Daten-Arbeit, nicht Architektur.
