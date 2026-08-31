# GEBOTE-/VERBOTEN-Integrations-Matrix

> **Status: Aktiv — maschinell geprüft durch `audit/spec_integration_scanner.py`.**
> Jede normative ID der GEBOTE.md (Referenzkatalog) und VERBOTEN.md muss entweder
> extern zitiert sein (Code/Tests/Instructions/Registry/Skripte/Specs — ohne das
> jeweilige Definitionsdokument und ohne diese Matrix selbst) oder hier einen
> dokumentierten Status haben. Fehlt beides, meldet der Scanner einen ERROR im
> Fehlerprotokoll. `katalog` = Referenzkatalog-Eintrag ohne externes Zitat;
> `integriert` = extern zitiert; VERBOTEN-IDs zusätzlich mit Linter-Status.

Stand: 227 IDs (166 integriert, 61 katalog).
Regeneriert mit `python scripts/gen_integration_matrix.py`.

| ID | Kategorie | Titel | Status |
|---|---|---|---|
| §G1 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Pro-Song-Kalibrierung | integriert |
| §G2 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Defekt-Vollständigkeit | integriert |
| §G3 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Gesangsintegrität | integriert |
| §G4 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Ghost-Echo-Freiheit | integriert |
| §G5 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Konsistenz-Mandat | integriert |
| §G6 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Null-Toleranz für Phasen-Leckage | integriert |
| §G7 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Interchannel-Lag | integriert |
| §G8 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | CD-Rauschprofil-Pflicht | integriert |
| §G9 | Kategorie I — Individuelle Song-Maximierung (§G1–§G9) | Quellmaterial-Unabhängigkeit | integriert |
| §G10 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | ERB-Masking-First | katalog |
| §G11 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Natürlicher Wohlklang | integriert |
| §G12 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Lautheitskonsistenz | integriert |
| §G13 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Multi-Point-Lag | integriert |
| §G14 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Spectral-Tilt-Guard | integriert |
| §G15 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Rauschprofil-Maskierung | integriert |
| §G16 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Rauschprofil-Charakteristik | integriert |
| §G17 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Stille-Respekt | integriert |
| §G18 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Spektrale Kohärenz | integriert |
| §G19 | Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19) | Dither-Doppelung-Verbot | integriert |
| §G20 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Bridge-Bypass-Verbot | katalog |
| §G21 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Denker-Zentralität | integriert |
| §G22 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Determinismus | katalog |
| §G23 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | ML-Fallback-Logging | integriert |
| §G24 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | NaN/Inf-Schutz | integriert |
| §G25 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Logger-Pflicht | integriert |
| §G26 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Guard-Counter-Lebendigkeit | integriert |
| §G27 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Messschleifen-Plateau | katalog |
| §G28 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | PIM-first, RLP-last | katalog |
| §G29 | Kategorie III — Architektur & Datenfluss (§G20–§G29) | Artistic Intent vor Defect-Scan | katalog |
| §G30 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | L/R-Unkorreliertheit | integriert |
| §G31 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | Maskierungs-Kanten-Glättung | katalog |
| §G32 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | ML-Device-Detection | katalog |
| §G33 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | ML-Recovery-API-Äquivalenz | katalog |
| §G34 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | Test-Assertion-Konvention | katalog |
| §G35 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | Export-Atomizität | katalog |
| §G36 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | True-Peak-Grenze | katalog |
| §G37 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | Feedback-Chain-Guards | katalog |
| §G38 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | Modus-Parameter-Isolation | katalog |
| §G39 | Kategorie IV — CD-Rauschprofil & Export (§G30–§G39) | Rauschprofil-Monitoring | integriert |
| §G40 | Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45) | Rauschprofil-Zeitpunkt | integriert |
| §G41 | Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45) | Übergangs-Verifikation | integriert |
| §G42 | Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45) | CD-Produktions-Kohärenz | integriert |
| §G43 | Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45) | Rauschprofil-Pegel-Anpassung | integriert |
| §G44 | Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45) | Maskierungs-Wissenschaft | integriert |
| §G45 | Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45) | Digital-Black-Integrität | integriert |
| §G46 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Harmonic Preservation Score | integriert |
| §G47 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Transient Preservation Score | integriert |
| §G48 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Formant Preservation Score | integriert |
| §G49 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | ABX Test Harness | integriert |
| §G50 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | MUSHRA Proxy Scorer | integriert |
| §G51 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Statistical Report | integriert |
| §G52 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Micro-Dynamics Score | integriert |
| §G53 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Artifact Detector | integriert |
| §G54 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Emotional Arc Score | integriert |
| §G55 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Blind Reference-Free Quality | integriert |
| §G56 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Noise Floor Continuity | integriert |
| §G57 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Sliding ERB Gain | integriert |
| §G58 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Vocal Repair Module | integriert |
| §G59 | Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59) | Restoration Quality Report | integriert |
| §G60 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | STCG Multi-Point-Primär | integriert |
| §G61 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | Chunk-Phasen-STCG-Pflicht | integriert |
| §G62 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | Sub-Sample-Lag-Korrektur | integriert |
| §G63 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | Lag-Messung-Orientierungsfrei | integriert |
| §G64 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | STCG-Singleton-Konsistenz | integriert |
| §G65 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | Post-Chunk-Global-STCG | integriert |
| §G66 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | Keine konkurrierenden Lag-Fixes | integriert |
| §G67 | Kategorie VII — Stereo-Lag-Integrität (§G60–§G67) | STFT-Input-Length-Guard | integriert |
| §G68 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | SFT-Novelty-Schwelle adaptiv pro Song | integriert |
| §G69 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | Defekt-Reparatur-Phasen-Klassifikation | katalog |
| §G70 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | SFT-Prioritätskette: Zerstörung vor Neuheit | integriert |
| §G71 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | Unhörbare Defekte als Qualitätsziel | integriert |
| §G72 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | Keine pauschalen Wet-Werte | integriert |
| §G73 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | Joint-Calibration Minimum | integriert |
| §G74 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | OneTakeExport-Garantie | integriert |
| §G75 | Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75) | Tuple-ndim Recovery | integriert |
| §G76 | Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81) | Zentraler Kalibrierungs-Kontext | integriert |
| §G77 | Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81) | Kontinuierliche Ableitung | integriert |
| §G78 | Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81) | Vollständigkeit der Kalibrierung | integriert |
| §G79 | Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81) | Kalibrierungs-Audit | integriert |
| §G80 | Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81) | Unkalibrierter-Fallback-Warnung | integriert |
| §G81 | Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81) | Einzige Quelle der Wahrheit | integriert |
| §G82 | Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86) | Lebendiger CalibrationContext | integriert |
| §G83 | Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86) | NOVELTY_CRIT-Rekalibrierung | integriert |
| §G84 | Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86) | Phasen-Stärke-Drift-Korrektur | integriert |
| §G85 | Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86) | Rekalibrierungs-Audit | integriert |
| §G86 | Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86) | Monotonie-Garantie | integriert |
| §G183 | Kategorie XI-b — Maschinelle Durchsetzung (§G183–§G187, früher §G122–§G126) | CalibrationContext-Dataclass | integriert |
| §G184 | Kategorie XI-b — Maschinelle Durchsetzung (§G183–§G187, früher §G122–§G126) | Linter-Baseline | katalog |
| §G185 | Kategorie XI-b — Maschinelle Durchsetzung (§G183–§G187, früher §G122–§G126) | Cross-Depth-Validierung | katalog |
| §G186 | Kategorie XI-b — Maschinelle Durchsetzung (§G183–§G187, früher §G122–§G126) | Kalibrierte Konstanten | katalog |
| §G187 | Kategorie XI-b — Maschinelle Durchsetzung (§G183–§G187, früher §G122–§G126) | Blindtest-Pflicht | integriert |
| §G87 | Kategorie XII — Noise-Floor-Brücke Phase_03→Phase_26 (§G87) | Phase_26 Per-Band-Noise-Floor-Guard | integriert |
| §G88 | Kategorie XIII — Defektbehebungs-Module auf höchster Qualitätsstufe (§G88) | Defektbehebung mit Depth-adaptiven DSP-Fallbacks | integriert |
| §G89 | Kategorie XIV — Unsichtbare Signalintegrität (§G89) | Soft-Clipping-Pflicht für alle 71 Phasen (68 + 3 Phase-0) | katalog |
| §G90 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Blinder-Referenz-Vektor-Pflicht | integriert |
| §G91 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Embedding-basierte-Referenz-Pflicht | integriert |
| §G92 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Material-adaptive-Confidence-Pflicht | integriert |
| §G93 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Exception-Proxy-Pflicht | integriert |
| §G94 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Cross-Phase-Metadata-Pflicht | integriert |
| §G95 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Phase-02-vor-Phase-03-Pflicht | integriert |
| §G96 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | HPI-NaN-Guard-Pflicht | integriert |
| §G97 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | log10-Null-Guard-Pflicht | integriert |
| §G98 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | AUTHENTIC_CHARACTER-Vollständigkeit | integriert |
| §G99 | Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99) | Equality-of-Materials-Pflicht | integriert |
| §G100 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Hörbarkeit vor Mathematik | integriert |
| §G101 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Perzeptueller Wet/Dry-Blend | integriert |
| §G102 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Bark-Band-Verarbeitung | integriert |
| §G103 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | LUFS-basierte Lautheit | integriert |
| §G104 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | JND-Gate nach jeder Phase | integriert |
| §G105 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | ISO-226-Hörschwellen-Integration | integriert |
| §G106 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Perzeptuelle Qualitätsgewichtung | integriert |
| §G107 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Ermüdungsfreier Klang | integriert |
| §G108 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Stille als psychoakustischer Raum | katalog |
| §G109 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Binaurale Natürlichkeit | katalog |
| §G110 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Transiente Hörbarkeit | katalog |
| §G111 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Adaptiver Frequenzgang | katalog |
| §G112 | Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112) | Perzeptuelles Monitoring | integriert |
| §G113 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | Universal RMS-Guard | katalog |
| §G114 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | Transient-Shift-Detektion | katalog |
| §G115 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | Hallucination-Guard | katalog |
| §G116 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | Formant-Stabilitäts-Guard | katalog |
| §G117 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | Groove-Guard | katalog |
| §G118 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | HPI-Gate im Goosebumps-Recovery | katalog |
| §G119 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | FeedbackChain-Silence-Guard | katalog |
| §G120 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | ExcellenceOptimizer RX11-Kalibrierung | katalog |
| §G121 | Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120) | Mode-Differenzierung: RESTORATION = Do No Harm | katalog |
| §G122 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | LUFS-Δ-Cap-Pflicht | integriert |
| §G123 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | Closed-Loop-Empfindlichkeit | integriert |
| §G124 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | ExcellenceOptimizer-Hysterese | integriert |
| §G125 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | MDEM-Per-5-Phasen-Prüfung | integriert |
| §G126 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | De-Esser-Soft-Saturation-Skip | katalog |
| §G127 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | Unbound-Variable-Scope-Garantie | katalog |
| §G128 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | GDD-Budget-Proaktivität | katalog |
| §G129 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | Rollback-Sanity-Pflicht | katalog |
| §G130 | Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130) | PresenceEmbedding-Export-Pflicht | integriert |
| §G131 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Perzeptuelle-Verbesserungs-Metrik-Pflicht | integriert |
| §G132 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Composite-Score-Schwelle | integriert |
| §G133 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Per-Defekt-Reduktions-Pflicht | integriert |
| §G134 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Defekt-Transparenz | integriert |
| §G135 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Chunked-Streaming-Determinismus-Pflicht | integriert |
| §G136 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Wiederholungs-Reproduzierbarkeit | integriert |
| §G137 | Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G137) | Full-Song-Defekt-Presence-Pflicht | integriert |
| §G138 | Kategorie XX — Perzeptueller Autopilot: Wohlklang-Garantien (§G138–§G141) | BlindQuality-Verbot-im-Gate | integriert |
| §G139 | Kategorie XX — Perzeptueller Autopilot: Wohlklang-Garantien (§G138–§G141) | Defekt-Countdown-Pflicht | integriert |
| §G140 | Kategorie XX — Perzeptueller Autopilot: Wohlklang-Garantien (§G138–§G141) | Export-Gate-Pflicht | integriert |
| §G141 | Kategorie XX — Perzeptueller Autopilot: Wohlklang-Garantien (§G138–§G141) | Wohlklang-Garantie-Pflicht | integriert |
| §G142 | Kategorie XXI — Perzeptueller Closed-Loop: Per-Band-Hören (§G142–§G145) | Per-Band-MUSHRA-Pflicht | integriert |
| §G143 | Kategorie XXI — Perzeptueller Closed-Loop: Per-Band-Hören (§G142–§G145) | Bark-Band-Blend-Pflicht | integriert |
| §G144 | Kategorie XXI — Perzeptueller Closed-Loop: Per-Band-Hören (§G142–§G145) | MUSHRA-Proxy-Pflicht | integriert |
| §G145 | Kategorie XXI — Perzeptueller Closed-Loop: Per-Band-Hören (§G142–§G145) | Perzeptueller-Rollback-Pflicht | integriert |
| §G150 | Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155) | Metrik-Hierarchie-Pflicht | integriert |
| §G151 | Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155) | MUSHRA-Primat | katalog |
| §G152 | Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155) | BlindQuality-Diagnostik-Verbot | integriert |
| §G153 | Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155) | Guard-Phasen-Whitelist-Pflicht | integriert |
| §G154 | Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155) | Adaptive-Schwellwert-Pflicht | integriert |
| §G155 | Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155) | Quality-Entscheidungs-Narrativ-Pflicht | integriert |
| §G156 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Depth+Restorability-adaptiver HPI-Gate | katalog |
| §G157 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Sample-Axis-Robustheit für B3-Phase-2 | katalog |
| §G158 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | MUSHRA/HPI-Forwarding an MQA | katalog |
| §G159 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | De-Esser-Dynamics-Threshold | katalog |
| §G160 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Chunked-Mode-Längenwarnung | katalog |
| §G161 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | P5-Exception-Traceback-Pflicht | katalog |
| §G162 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | HPI-Gate-Restorability-Kontinuität | katalog |
| §G163 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Floor-Absolut-Garantie | katalog |
| §G164 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Studio-Master-Floor-Invariante | katalog |
| §G165 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Spec-Constitution-Synchronisation | katalog |
| §G166 | Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166) | Drei-Quellen-Synchronisation | katalog |
| §G167 | Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706) | Export-Gate-B30-Komplettierung | katalog |
| §G168 | Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706) | SourceMediumProfile-Kalibrierungspflicht | katalog |
| §G169 | Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706) | Per-Phase-SMP-Cap | katalog |
| §G170 | Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706) | Chain-Depth-Budget-Adaption | katalog |
| §G171 | Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706) | Material-Fremdlauf-Transparenz | katalog |
| §G172 | Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706) | OneTakeExport-ISP-Margin | katalog |
| §G173 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Event-Garantie | integriert |
| §G174 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Lock-freie Importe | integriert |
| §G175 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Plugin-Namen-Validierung | katalog |
| §G176 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Watchdog-Selbsttest | katalog |
| §G177 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Cache-Safety | katalog |
| §G178 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Happy-Path-Gate | katalog |
| §G179 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Startup-Smoke-Test | katalog |
| §G180 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Import-Check | katalog |
| §G181 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | GPU-Detection Safety | katalog |
| §G182 | Kategorie XXIV — Startup-Integration & Kommunikation (§G173–§G182, früher §SC-G71–§SC-G80, §v10.305) | Unified Progress | integriert |
| V01 | VERBOTEN.md | `backend/`, `plugins/` | integriert (Linter) |
| V02 | VERBOTEN.md | `backend/`, `plugins/` | integriert (Linter) |
| V03 | VERBOTEN.md | `plugins/` | integriert (Linter) |
| V04 | VERBOTEN.md | `backend/core/` | integriert (Linter) |
| V05 | VERBOTEN.md | `backend/core/phases/` | integriert (Linter) |
| V08 | VERBOTEN.md | `backend/`, `plugins/` | integriert (Linter) |
| V09 | VERBOTEN.md | `backend/core/` | integriert (Linter) |
| V11 | VERBOTEN.md | `backend/core/phases/` | integriert (Linter) |
| V12 | VERBOTEN.md | `backend/core/causal_defect_reasoner.py` | integriert (Linter) |
| V13 | VERBOTEN.md | `backend/core/unified_restorer_v3.py` | integriert (Linter) |
| V27 | VERBOTEN.md | `backend/core/unified_restorer_v3.py`, `causal_defect_reasoner.py`, `defect_phase_mapper.py` | integriert (Linter) |
| V28 | VERBOTEN.md | | ⏳ V28 DFN-MUSIK: `NR_BREATHING_ARTIFACT` → `phase_03_denoise` / `phase_29` | NR-Atmen/Pumpen entsteht durch  | integriert (Linter) |
| V29 | VERBOTEN.md | `backend/core/causal_defect_reasoner.py`, `defect_phase_mapper.py` | integriert (Linter) |
| V30 | VERBOTEN.md | | ⏳ V30 DFN-MUSIK: `ALIASING` → `phase_03_denoise` | Aliasing-Spiegelfrequenzen sind kohärente Signalspiegelun | integriert (Linter) |
| V31 | VERBOTEN.md | `backend/core/defect_phase_mapper.py`, `causal_defect_reasoner.py` | integriert (Linter) |
| V32 | VERBOTEN.md | `backend/core/cumulative_interaction_guard.py` | integriert (Linter) |
| V33 | VERBOTEN.md | `backend/core/phases/phase_*.py` | integriert (Linter) |
| V38 | VERBOTEN.md | `backend/core/phases/phase_*.py` | integriert (zitiert) |
| V39 | VERBOTEN.md | `backend/core/causal_defect_reasoner.py`, `defect_phase_mapper.py` | integriert (Linter) |
| V40 | VERBOTEN.md | `backend/core/phases/phase_03*.py`, `phase_29*.py` (NR-Phasen) | integriert (zitiert) |
| V41 | VERBOTEN.md | `backend/core/phases/phase_*.py` (additive Phasen, `panns_singing ≥ 0.25`) | integriert (zitiert) |
| V42 | VERBOTEN.md | `backend/core/phases/phase_03*.py`, `phase_29*.py` | integriert (zitiert) |
| V43 | VERBOTEN.md | `backend/core/phases/phase_*.py`, `backend/core/dsp/lpc_formant_tracker.py` | integriert (zitiert) |
| V44 | VERBOTEN.md | `backend/core/musical_goals/musical_goals_metrics.py` | integriert (Linter) |
| V45 | VERBOTEN.md | `backend/core/musical_goals/musical_goals_metrics.py` | integriert (zitiert) |
| V46 | VERBOTEN.md | `backend/core/dsp/noise_texture_resynth.py`, sowie jede DSP-Datei die dBFS-Werte skaliert | integriert (Linter) |
| V47 | VERBOTEN.md | `backend/core/clipping_detection.py` | integriert (Linter) |
| V48 | VERBOTEN.md | `backend/core/goal_applicability_filter.py`, `backend/core/unified_restorer_v3.py` | integriert (Linter) |
| V49 | VERBOTEN.md | `denker/exzellenz_denker.py`, `denker/aurik_denker.py` | integriert (Linter) |
| V50 | VERBOTEN.md | `denker/exzellenz_denker.py` | integriert (zitiert) |
| V51 | VERBOTEN.md | `denker/restaurier_denker.py`, `denker/aurik_denker.py` | integriert (zitiert) |
| V52 | VERBOTEN.md | `backend/core/goal_applicability_filter.py` | integriert (zitiert) |
| V53 | VERBOTEN.md | `backend/core/unified_restorer_v3.py` | integriert (Linter) |
| V54 | VERBOTEN.md | `backend/core/unified_restorer_v3.py` | integriert (zitiert) |
| V55 | VERBOTEN.md | `backend/core/dsp/lpc_formant_tracker.py`, `backend/core/phases/phase_42_vocal_enhancement.py`, `backend/core/ | integriert (zitiert) |
| V56 | VERBOTEN.md | `Aurik10/ui/modern_window.py` | integriert (zitiert) |
| V57 | VERBOTEN.md | `backend/core/phases/phase_*.py` (alle additiven Phasen mit `panns_singing ≥ 0.25`) | integriert (zitiert) |
| V58 | VERBOTEN.md | `backend/core/unified_restorer_v3.py` (und alle ndarray-Return-Funktionen) | integriert (Linter) |
| V70 | VERBOTEN.md | `denker/`, `backend/core/`, `Aurik10/ui/` | integriert (zitiert) |
| V71 | VERBOTEN.md | `Aurik10/ui/`, `backend/core/pre_analysis.py` | integriert (zitiert) |
| V72 | VERBOTEN.md | `backend/core/pre_analysis.py`, `Aurik10/ui/` | integriert (zitiert) |
| V73 | VERBOTEN.md | `Aurik10/ui/` | integriert (Linter) |
| V74 | VERBOTEN.md | Alle `.py`-Dateien | integriert (Linter) |
| V75 | VERBOTEN.md | `Aurik10/ui/` | integriert (Linter) |
