# Hörordnung — Psychoakustische Wahrheits-Ordnung (normativ)

> **Status:** Normative Spitze für Hör-Entscheidungen. Diese Datei steht in der
> normativen Kette (AGENTS.md §1) **über** `musical_goals.instructions.md`,
> `dsp.instructions.md` und den Goal-Specs — aber **nur** für die Frage, welche
> Größe bei Konflikt den Ausschlag gibt. Die Berechnung der Messgrößen bleibt
> unverändert dort normativ, wo sie definiert ist.
>
> **Grundsatz (vgl. 01_musical_goals.md, Einleitung):** „Der CausalDefectReasoner
> kann irren — das Ohr nicht." Diese Datei operationalisiert diesen Satz.

Arbeitssprache: Deutsch. Alle §-Zitate mit Quelle.

---

## 1. Die Rolle der Messwerte: Zeugen, nicht Richter

Technische Metriken (MUSHRA/NSIM/MCD, HPI, die 15 Musical Goals, VQI,
Goosebumps, HPE) sind **Evidenz**. Kein einzelner Messwert darf allein eine
Entscheidung mit destruktiver Wirkung tragen (Re-Run, Veto, Rollback, Phase-
Strip). Solche Entscheidungen laufen immer über die Ebenen in §2.

Ausnahme: Die Härte-Invarianten der Ebene 1 (§3) sind selbst die oberste
Entscheidungsinstanz — sie sind die operationalisierte Form dessen, was das
Ohr niemals akzeptiert.

## 2. Die vier Ebenen

Jede Ebene dominiert die darunterliegende. Eine Entscheidung auf Ebene n darf
niemals eine Invariante der Ebene < n verletzen.

1. **Hör-Invarianten** — unverhandelbar (siehe §3)
2. **Audibility-Schicht** — nur Hörbares wird repariert (§4)
3. **Wohlklang-Ordnung** — lexikografische Ziel-Hierarchie (§5)
4. **Einladungs-Gate** — positiver Wohlklang-Nachweis (§6)

## 3. Ebene 1 — Hör-Invarianten (Veto-Bereich)

Verletzt eine Phase eine dieser Invarianten, wird die **verursachende Phase
zurückgenommen oder geblendet** — nicht erst am Pipeline-Ende „wiederhergestellt".

| Invariante | Messung (bestehend, normativ dort) | Schwelle |
|---|---|---|
| Stimm-Identität | `singer_identity_cosine` (Spec 01 §2.35c/d) | ≥ 0.92 (kein Unterschreiten durch irgendeine Phase) |
| Konsonanten-/Atem-Energie | `consonant_clarity` (Spec 01 §2.35d-ii) | ≥ 0.85 |
| Vibrato-Erhalt | `vibrato_precision` (Spec 01 §2.35d-i) | Rate-Fehler ≤ 0.3 Hz, Tiefen-Erhalt ≥ 0.85 |
| Dynamikbogen | EmotionalArc (Spec 01 §2.35e) | Arc-Korrelation ≥ Schwelle aus §2.35e-ii |
| Atem-Zeitstruktur | BreathEmotionClassifier-Segmente | Anzahl/Position der Atemer dürfen sich durch NR nicht um > 10 % ändern |

**Verboten:** Eine Ebene-1-Verletzung durch „Recovery-Kaskade" zu kaschieren,
wenn die Ursache eine identifizierbare Einzelphase ist (Ursache statt Symptom —
§V7 (copilot-instructions.md)).

## 4. Ebene 2 — Audibility: Reparaturziel ist die Maskierungsschwelle

Reparatur gilt als **abgeschlossen**, wenn ein Defekt **unter der psychoakustischen
Maskierungsschwelle** liegt — nicht wenn sein Messwert Null ist.

- Maßgeblich: Masking-Modell nach ISO 11172-3 (Bark), zeitlich **und** spektral
  lokal (vgl. §2.62 (dsp.instructions.md)).
- Jede Reparatur-Entscheidung (Phase an/aus, Stärke, Re-Scan) fragt zuerst:
  **„Ist dieser Defekt über der Maskierungsschwelle hörbar?"** Nur dann wird
  repariert.
- Berichte (z. B. §v10.703 Defekt-Countdown, §B2 Per-Defekt-Reduktion) müssen
  „hörbar" als **„über Maskierungsschwelle"** ausweisen. Ein Re-Scan, der nach
  erfolgreicher Reparatur neue „Defekte" unterhalb der Schwelle meldet, darf
  nicht als Misserfolg gezählt werden.
- **Mindestanforderung an PerceptualSalience:** Der Salience-Filter muss real
  maskieren. Läuft er über einen Abschnitt mit einem Anteil `salient ≥ 0.99`
  bei breitbandigem Musikmaterial, ist er als Pass-Through zu behandeln und
  darf keine Audibility-Entscheidung tragen (Befund 2026-08-23: 12969/12969
  salient, mean=1.000).

## 5. Ebene 3 — Lexikografische Wohlklang-Ordnung

Bei Zielkonflikt gilt die Reihenfolge strikt — ein höherrangiges Ziel darf für
kein niederrangiges gesenkt werden. Der Pareto-Tie-Break nach Hörpriorität
(Spec 01 §2.36) wird von der End-Gate-Ausnahme zur Regel für **alle** Phasen
und für die FeedbackChain-Boosts.

1. **Natürlichkeit** (P1: DNSMOS/SingMOS, HNR, Mikrodynamik — Spec 01 §2.34, §MKK (dsp.instructions.md))
2. **Wärme** (§WBG 200–800 Hz-Balance — §V25 (dsp.instructions.md), waerme P4)
3. **Klarheit / Durchhörbarkeit** (Maskierungs-SMR — transparenz, sep_fidelity)
4. **Brillanz** (bounded durch Material-BW-Ceiling — §6.2c, §2.46a Chain-End-Ceiling)

**Verboten:** Ein Brillanz-/Transparenz-Boost, der Wärme oder Natürlichkeit
senkt — unabhängig davon, ob der Einzel-Score des Boost-Ziels steigt
(Teamwork- statt Dominanz-Prinzip, Spec 01 §1.2c).

## 6. Ebene 4 — Einladungs-Gate (positiver Nachweis)

„Wohlklang, in den sich das Ohr hineinlegt" ist ein **positives** Kriterium und
wird als Fenster-Gate gemessen, nicht als Einzelwert:

- **Zeitverlauf-Gate:** Roughness (Zwicker), Sharpness (Bismarck), Loudness
  (ERB, ISO 532-1) über Fenster (z. B. 5 s, überlappend). Das Gate ist erfüllt,
  wenn keine Roughness-Spitze `asper > 0.5` in Stimmen-/Klimax-Zonen liegt und
  der Sharpness-Verlauf keine Sprünge > 0.2 acum zwischen benachbarten Fenstern
  aufweist. (Die Größen sind bereits als Gewichtungs-Input in Spec 01 §2.56
  Stufe 3 definiert — hier werden sie zum eigenständigen Gate.)
- **Ermüdungs-Abbruch:** Fatigue (experience_runtime) > 0.40 beendet die
  Optimierung (gilt schon in OneTakeExport; hier generell für End-Gate/FC).
- **Bezugspunkt:** Die Goosebumps-/HPE-Bewertung bleibt erhalten, aber ein
  „EXZELLENT" ersetzt das Einladungs-Gate nicht — beide müssen unabhängig halten.

## 7. Konfliktregel — die Hör-Instanz entscheidet

Wenn zwei Messgrößen einander widersprechen, gilt:

1. Hält Ebene 1 (alle Hör-Invarianten erfüllt)?
2. Hält eine **Hör-Instanz** (GoosebumpsQualityChecker oder HPE) auf hohem
   Niveau (Goosebumps ≥ 0.75 bzw. HPE stabil hoch)?

Wenn beides ja: Die widersprechende automatische Metrik wird als
**Messartefakt-Verdacht** behandelt. Sie darf dann **keinen** Re-Run, kein
Veto, keinen Rollback und keinen Phase-Strip auslösen; stattdessen wird eine
Warnung geloggt und die Messung ggf. auf einem korrekten Fenster wiederholt
(Bezug: Alignment-Artefakt-Guard der Wohlklang-Garantie, 2026-08-23).

Wenn Ebene 1 verletzt ist, hat die Hör-Instanz **keinen** Vorrang — die
Invariante dominiert.

## 8. Maschinen-Wahrheit — wo die Ebenen leben

Die Ebenen binden Komponenten an ihre Rolle (Berechnung bleibt bei den
jeweiligen Mess-Definitionen; hier zählt nur der Entscheidungsfluss):

| Ebene | Träger |
|---|---|
| 1 | VQI-Gate + singer_identity-Rollback + EmotionalArc (UV3, Spec 01 §2.35c–e), ConsonantClarity; **§SCK-R/§WBG-R** Phasen-Rücknahme im zentralen Phasen-Call (`unified_restorer_v3.py`, Konkretisierung zu V24/V25 in dsp.instructions.md) |
| 2 | `compute_masking_threshold_iso11172` (dsp §2.62), PerceptualSalience (Pass-Through-Guard), `residuum_masking.py` (3. Blend-Term), `_should_skip_masked_phase` (Stufe B), §v10.703-Countdown (Stufe A: ERB-maskierte Events) |
| 3 | GoalPriorityProtocol (Spec 01 §2.34) + `HEARING_TIER_MAP`/`hearing_tier()` (Hörordnungs-Dominanzstufe), `goal_weights` (§2.56), PhaseConductor/PMGG; Guards in FeedbackChain (intern + UV3-Callback) und End-Gate-Ranking |
| 4 | `inviting_sound_gate.py` (Fenster-Gate), experience_runtime (fatigue_index), GoosebumpsQualityChecker, OneTakeExport |
| §7 Konfliktregel | Wohlklang-Garantie-Alignment-Guard, af-false-positive-Handling, MQA-Verdict-Kennzeichnung („Messartefakt-Verdacht“) |

Änderungen an diesen Trägern bleiben der jeweiligen Spec unterworfen; diese
Datei regelt nur den Entscheidungsfluss zwischen ihnen.

## 8a. Kalibrierungs-Wächter

`scripts/horordnung_calibration.py` prüft die psychoakustischen Invarianten der
Hörordnungs-Module gegen synthetische Referenz-Signale mit bekannten
Eigenschaften (Roughness-/Sharpness-/Residuum-Monotonie, Maskierungs-Richtung,
Hörstufen-Konsistenz). Verletzung ⇒ Exit 1.

- Verdrahtet als Pre-Commit-Hook `aurik-horordnung-calibration`
  (läuft bei Änderungen an den psychoakustischen Trägern).
- Neue Schwellwerte oder Modell-Änderungen an Ebene 2/4 MÜSSEN im Harness
  eine Invariante ergänzen — sonst verschiebt sich die Kalibrierung still.
- Das Harness ist die maschinelle Vorstufe der Panel-Kalibrierung (§7, Abschluss
  der echten Hörertests bleibt menschliche Prozessarbeit).

## 9. Verhältnis zur normativen Kette

- Kein Widerspruch zu den Mess-Definitionen in `musical_goals.instructions.md`,
  `dsp.instructions.md` oder Spec 01 — diese bleiben für die **Berechnung**
  normativ.
- Bei Konflikt zwischen einer Rollen-Regel dieser Datei und einer
  Mess-Definition gilt: Die Messung bleibt wie definiert, die
  **Entscheidungsnutzung** folgt dieser Datei. Der Konflikt ist im
  PR-Evidenzblock zu dokumentieren (AGENTS.md §4).
- Regeländerungen hier ⇒ betroffene Gates/Verifier in Skripten nachziehen
  (AGENTS.md §2, Absatz „Regeländerung ⇒ Skript nachziehen").
