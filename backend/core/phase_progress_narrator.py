"""PhaseProgressNarrator — Klare, für jeden verständliche Fortschrittsmeldungen.

Jede Meldung erklärt in einfacher Sprache, was Aurik gerade tut und warum.
Kein Fachchinesisch — so verständlich, dass es jeder Mensch sofort begreift.

Prinzipien:
  - Alltagssprache, keine Fachbegriffe aus Programmierung oder Tontechnik
  - Konkrete Vergleiche aus der Alltagswelt (wie ein Restaurator, wie ein Archäologe)
  - Jede Aktion wird mit ihrem Zweck erklärt (was + warum)
  - 10 Sekunden Mindesteinblenddauer — Zeit zum Lesen
  - Persönliche Ansprache, als würde ein Freund erklären
"""

from __future__ import annotations

import hashlib
import time as _time
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Tonträger-Namen — für jeden verständlich
# ═══════════════════════════════════════════════════════════════════════════════
_TRÄGER_NAMEN: dict[str, str] = {
    "vinyl": "einer Vinyl-Schallplatte",
    "shellac": "einer alten Schellackplatte",
    "reel_tape": "einem Tonband (grosse Spulen)",
    "cassette_tape": "einer Musikkassette",
    "cd": "einer CD",
    "mp3_low": "einer MP3-Datei (stark verkleinert)",
    "mp3_high": "einer MP3-Datei",
    "aac": "einer AAC-Datei",
    "streaming": "einem Streaming-Dienst",
    "digital": "einer digitalen Datei",
    "unknown": "einem unbekannten Träger",
}
_TRÄGER_KURZ: dict[str, str] = {
    "vinyl": "Vinyl-Platte",
    "shellac": "Schellackplatte",
    "reel_tape": "Tonband",
    "cassette_tape": "Kassette",
    "cd": "CD",
    "mp3_low": "MP3",
    "mp3_high": "MP3",
    "aac": "AAC",
    "streaming": "Stream",
    "digital": "Digital",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Tonträgerketten-Erzählung — wie Aurik herausfindet, woher die Musik stammt
# ═══════════════════════════════════════════════════════════════════════════════
_KETTEN_ERZAEHLUNG: list[str] = [
    "Aurik hat genau hingehört und erkannt, dass Deine Musik von "
    "{chain} stammt. Jede Art von Musikträger — ob Platte, Band oder "
    "Datei — hinterlässt nämlich ganz bestimmte Spuren im Klang. "
    "So wie ein Detektiv anhand von Fingerabdrücken weiss, wer am Tatort war, "
    "erkennt Aurik an diesen Spuren, woher Deine Musik kommt.",
    "Deine Musik hat eine kleine Reise hinter sich: {chain}. "
    "Das sind {num_stages} verschiedene Stationen. Aurik hat das herausgefunden, "
    "indem es das Klangbild ganz genau untersucht hat. Jede dieser Stationen "
    "hat ihre eigenen, typischen Merkmale — ähnlich wie verschiedene "
    "Fotos desselben Motivs je nach Kamera anders aussehen.",
    "Spannend: Deine Aufnahme war ursprünglich {first_stage} "
    "und hat im Lauf der Zeit {num_stages} weitere Stationen durchlaufen. "
    "Das ist wie bei einem alten Foto, das mehrfach kopiert wurde — "
    "jede Kopie verliert ein bisschen an Schärfe. Aurik weiss genau, "
    "wie es diese Verluste wieder ausgleichen kann.",
    "Aurik vergleicht Deine Musik mit 76 verschiedenen Mustern, "
    "die typisch für verschiedene Musikträger sind. Dabei kam heraus: "
    "{chain}. So wie ein erfahrener Uhrmacher am Klang erkennt, "
    "welches Uhrwerk tickt, so erkennt Aurik, woher Deine Musik stammt.",
    "Die Untersuchung zeigt: {chain}. Aurik hat dafür über 60 verschiedene "
    "Merkmale Deiner Aufnahme geprüft — vom leisesten Rauschen bis zur "
    "höchsten Höhe. Das Ergebnis: ein klares Bild davon, welchen Weg "
    "Deine Musik genommen hat, bevor sie zu Dir kam.",
    "Deine Musik begann ihr Leben als {first_stage}. Seitdem hat sie "
    "eine spannende Reise hinter sich. Aurik weiss jetzt genau, "
    "wie es sie am besten behandelt — so wie ein Restaurator weiss, "
    "ob ein Gemälde auf Leinwand oder Holz gemalt wurde.",
    "Wie erkennt Aurik eigentlich, woher Deine Musik kommt? "
    "Ganz einfach: Es hört sich die leisesten Geräusche an — "
    "das feine Knistern einer Platte, das sanfte Rauschen eines Bandes, "
    "die typischen Verluste einer komprimierten Datei. "
    "Jeder dieser Klänge verrät Aurik etwas über die Herkunft.",
    "Die Spurensuche ergab: {chain}. Stell Dir das vor wie eine "
    "Geschichte, die Deine Musik erzählt — von ihrer Geburt als "
    "{first_stage} bis zu dem Moment, als sie bei Dir ankam. "
    "Aurik hat diese Geschichte gelesen und verstanden.",
]

_WARUM_KETTE_WICHTIG: list[str] = [
    "Warum ist das wichtig? Weil Aurik jede Art von Musik anders "
    "behandeln muss. Eine Schallplatte hat andere Probleme als eine "
    "Kassette — und eine MP3-Datei wieder ganz andere. Wenn Aurik "
    "weiss, woher Deine Musik kommt, kann es die richtigen Werkzeuge "
    "auswählen. So wie ein Arzt eine andere Behandlung braucht als ein "
    "Mechaniker — Aurik braucht für Vinyl andere Methoden als für MP3.",
    "Weil Aurik jetzt genau weiss, dass Deine Musik von {chain} stammt, "
    "kann es jeden einzelnen Arbeitsschritt perfekt darauf abstimmen. "
    "So wie ein Koch weiss, ob ein Steak medium oder well-done sein soll — "
    "Aurik weiss jetzt, ob es sanft oder kräftig vorgehen muss.",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Phase-Erklärungen: WARUM dieser Schritt für DIESEN Song
# ═══════════════════════════════════════════════════════════════════════════════
_WARUM_DIESE_PHASE: dict[str, dict[str, list[str]]] = {
    "phase_01": {
        "_default": [
            "Deine Musik hat kleine Knackser und Klicks — das sind winzige "
            "Störungen, die wie ein kurzes Knacksen klingen. Aurik entfernt "
            "sie jetzt ganz gezielt, ohne den Rest der Musik anzutasten.",
        ],
        "vinyl": [
            "Deine Vinyl-Platte hat die typischen Knackser und Klicks, "
            "die beim Abspielen mit der Nadel entstehen. Aurik sucht jetzt "
            "jeden einzelnen dieser Störer und entfernt ihn — "
            "so vorsichtig, dass die Musik unberührt bleibt.",
            "Die Nadel einer Schallplatte hinterlässt Spuren: winzige Knackser. "
            "Aurik ist darin geübt, sie zu finden und zu beseitigen — "
            "wie ein Restaurator, der Staub von einem Gemälde pustet.",
        ],
    },
    "phase_03": {
        "_default": [
            "Jetzt geht es ans Eingemachte: Aurik entfernt das Grundrauschen. "
            "Das ist wie das leise Hintergrundgeräusch, das man bei alten "
            "Aufnahmen oft hört. Aurik trennt es sauber von der Musik.",
        ],
        "vinyl": [
            "Deine Vinyl-Platte hat ein ganz eigenes, feines Rauschen — "
            "jede Platte klingt da etwas anders. Aurik erkennt dieses "
            "Rauschen und zieht es behutsam aus der Musik heraus.",
        ],
        "reel_tape": [
            "Tonbänder haben ein charakteristisches Rauschen — ein sanftes "
            "Zischen, das im Hintergrund mitschwingt. Aurik entfernt es, "
            "ohne die Wärme des Bandklangs zu verlieren.",
        ],
        "cassette_tape": [
            "Kassetten rauschen von Natur aus stärker als andere Tonträger. "
            "Aurik geht hier besonders geschickt vor — es holt das Rauschen "
            "raus, aber lässt die Musik schön klar klingen.",
        ],
    },
    "phase_05": {
        "_default": [
            "Ganz tiefe Töne, die man kaum hört, können trotzdem stören — "
            "zum Beispiel das Rumpeln eines Plattentellers. Aurik filtert "
            "diese tiefen Störungen jetzt heraus.",
        ],
    },
    "phase_06": {
        "_default": [
            "Mit der Zeit verliert Musik oft ihre hohen Töne — die Brillanz "
            "geht verloren. Aurik stellt diese Höhen wieder her, und zwar "
            "so, als wären sie nie verschwunden gewesen.",
        ],
        "mp3_low": [
            "MP3-Dateien werfen beim Verkleinern viele hohe Töne einfach weg. "
            "Das spart Platz, kostet aber Klang. Aurik holt diese verlorenen "
            "Höhen jetzt Stück für Stück zurück.",
        ],
    },
    "phase_09": {
        "_default": [
            "Das feine Knistern auf der Oberfläche — bei {material} "
            "ganz normal — wird jetzt entfernt. Aurik geht dabei so "
            "behutsam vor, dass die Musik nicht stumpf oder glatt klingt.",
        ],
    },
    "phase_12": {
        "_default": [
            "Bei Bandaufnahmen kommt es manchmal vor, dass die Tonhöhe "
            "leicht schwankt — wie bei einem Plattenspieler, der nicht "
            "ganz rund läuft. Aurik gleicht das jetzt aus, sodass alles "
            "wieder stabil und sauber klingt.",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Phase-Aktivitäten — was Aurik konkret tut (Alltagssprache)
# ═══════════════════════════════════════════════════════════════════════════════
_AKTIVITAETEN: dict[str, list[str]] = {
    "phase_01": [
        "Entfernt Knackser — so präzise wie ein Chirurg …",
        "Jeder Klick wird geortet und sanft entfernt …",
        "Kleine Störer verschwinden, die Musik bleibt …",
        "Was geknackst hat, klingt gleich sauber …",
        "Die Oberfläche wird von Knacksern befreit …",
        "Knackser für Knackser — es wird immer sauberer …",
        "Kurzzeit-Störungen werden entfernt …",
    ],
    "phase_02": [
        "Entfernt das tiefe Brummen aus dem Hintergrund …",
        "Das störende Netzbrummen verschwindet …",
        "Ein sauberer Klang entsteht — ohne Brummen …",
        "Die 50-Hz-Störung wird beseitigt …",
    ],
    "phase_03": [
        "Das Grundrauschen weicht — die Musik atmet auf …",
        "Rauschen und Musik werden voneinander getrennt …",
        "Schicht für Schicht geht das Rauschen weg …",
        "Wie ein Archäologe, der eine Vase ausgräbt — behutsam wird die Musik freigelegt …",
        "Die Details der Aufnahme kommen langsam zum Vorschein …",
        "Was vorher im Rauschen unterging, wird hörbar …",
        "Aurik arbeitet sich durch das Rauschen — wie ein Taucher, der zum Meeresgrund hinabsteigt …",
        "Langsam wird es stiller im Hintergrund …",
    ],
    "phase_04": [
        "Bringt Bässe, Mitten und Höhen ins Gleichgewicht …",
        "Die Klangfarbe wird ausbalanciert — nichts dröhnt, nichts fehlt …",
        "Passt den Klang an wie ein Optiker eine Brille — so lange, bis alles perfekt scharf ist …",
        "Was zu dumpf klang, wird klarer. Was zu scharf war, wird weicher …",
        "Die richtige Balance für Deine Musik …",
    ],
    "phase_05": [
        "Entfernt tiefes Rumpeln, das man kaum hört …",
        "Das Wummern des Plattentellers verschwindet …",
        "Alles, was nur die Boxen vibrieren lässt, wird entfernt …",
    ],
    "phase_06": [
        "Stellt verlorene Höhen wieder her …",
        "Die Brillanz kehrt zurück …",
        "Was dumpf war, bekommt wieder Glanz …",
        "Fehlende Klangfarben werden ergänzt …",
        "Die Musik wird wieder luftig und offen …",
        "Wie beim Öffnen eines Fensters — frische Luft für Deine Musik …",
    ],
    "phase_07": [
        "Bringt Wärme und Fülle zurück …",
        "Die feinen Klangfarben jedes Tons leben wieder auf …",
        "Die Musik klingt wieder natürlich und voll …",
        "Obertöne, die verloren gingen, werden wieder hörbar …",
    ],
    "phase_08": [
        "Schützt die knackigen Anfänge jedes Tons …",
        "Trommeln und Gitarrenanschläge bleiben lebendig …",
        "Nichts klingt verwaschen — alles bleibt präzise …",
    ],
    "phase_09": [
        "Das Knistern der Oberfläche wird entfernt …",
        "Die Plattenoberfläche klingt gleich viel sauberer …",
        "Feinste Störgeräusche verschwinden …",
    ],
    "phase_12": [
        "Gleicht leichte Tonhöhen-Schwankungen aus …",
        "Der Ton bekommt wieder sicheren Stand …",
        "Die Musik schwebt nicht mehr — alles sitzt fest …",
        "Stabile Tonhöhe von Anfang bis Ende …",
    ],
    "phase_13": [
        "Verbessert das räumliche Klangbild …",
        "Links und rechts werden perfekt ausbalanciert …",
        "Die Musik bekommt Tiefe und Raum …",
    ],
    "phase_17": [
        "Der letzte Feinschliff — wie beim Polieren eines Edelsteins …",
        "Alles wird noch einmal verfeinert und abgerundet …",
        "Die finale Politur für den perfekten Klang …",
    ],
    "phase_18": [
        "Befreit die stillen Momente vom Rauschen …",
        "Zwischen den Tönen herrscht jetzt Ruhe …",
        "Die Pausen in der Musik werden wirklich still …",
    ],
    "phase_19": [
        "Macht scharfe Zischlaute angenehmer …",
        "Die S-Laute werden weicher, ohne dumpf zu werden …",
        "Angenehmer zu hören — weniger scharf …",
    ],
    "phase_20": [
        "Verringert unerwünschten Hall …",
        "Die Musik klingt direkter und näher …",
        "Weniger Nachhall — mehr Klarheit …",
    ],
    "phase_23": [
        "Repariert beschädigte Stellen im Klang …",
        "Lücken werden geschlossen — nichts fehlt mehr …",
        "Was beschädigt war, wird wieder ganz …",
    ],
    "phase_24": [
        "Füllt kurze Aussetzer im Ton …",
        "Wenn der Ton mal weg war, holt Aurik ihn zurück …",
        "Fehlende Momente werden rekonstruiert …",
    ],
    "phase_29": [
        "Das Bandrauschen wird leiser …",
        "Das Zischen des Bandes tritt in den Hintergrund …",
        "Mehr Musik, weniger Rauschen …",
    ],
    "phase_31": [
        "Korrigiert die Geschwindigkeit …",
        "Alles wieder im richtigen Tempo …",
        "Zu schnell oder zu langsam? Jetzt stimmt's …",
    ],
    "phase_40": [
        "Stellt die ideale Lautstärke ein …",
        "Weder zu leise noch zu laut — genau richtig …",
        "Deine Musik klingt auf jedem Gerät optimal …",
    ],
    "phase_42": [
        "Verbessert die Klarheit der Stimme …",
        "Jedes Wort wird deutlicher und präsenter …",
        "Die Stimme steht jetzt im besten Licht …",
    ],
}

_ALLGEMEINE_AKTIVITAETEN: list[str] = [
    "Arbeitet mit höchster Sorgfalt an Deiner Musik …",
    "Jeder Rechenschritt bringt besseren Klang …",
    "Gute Restaurierung braucht ein wenig Zeit — und die geben wir ihr …",
    "Deine Musik verdient diese Aufmerksamkeit …",
    "Im Hintergrund laufen komplexe Berechnungen — alles für Deine Musik …",
    "Qualität vor Geschwindigkeit — immer …",
    "So sorgfältig wie ein Uhrmacher …",
    "Deine Musik in guten Händen …",
]


class PhaseProgressNarrator:
    """Erklärt jeden Schritt — für jeden verständlich."""

    def __init__(self) -> None:
        self._session_key = hashlib.md5(str(id(self)).encode(), usedforsecurity=False).hexdigest()[:6]
        self._used: dict[str, list[int]] = {}
        self._last_ts: dict[str, float] = {}
        self._rotate_every_s: float = 10.0
        self._context: dict[str, Any] = {}
        self._chain_story_told: bool = False
        self._chain_story_index: int = 0
        # §v10.203 S1: Live-Chapter-Tracking
        self._chapters_emitted: set[str] = set()
        # §v10.203 S2: Chapter-Transition — letztes Kapitel
        self._last_chapter: str = ""
        # §v10.203 S4: Discovery-Tracking
        self._discoveries_emitted: set[str] = set()
        # §v10.203 S7: Phase-Start-Zeit für Entertainment-Timing
        self._phase_start: dict[str, float] = {}
        self._intermezzo_emitted: dict[str, float] = {}  # phase_key -> last emission time

    # ── Kontext setzen ──────────────────────────────────────────────────────

    def set_context(
        self,
        *,
        material: str = "",
        era_decade: int | None = None,
        transfer_chain: list[str] | None = None,
        defects: list[str] | None = None,
        restorability: float | None = None,
    ) -> None:
        self._context = {
            "material": str(material or "").lower(),
            "era_decade": era_decade,
            "transfer_chain": list(transfer_chain or []),
            "defects": list(defects or []),
            "restorability": restorability,
        }
        self._chain_story_told = False
        self._chain_story_index = 0

    # ── Tonträgerketten-Erzählung ──────────────────────────────────────────

    def chain_narrative(self) -> str:
        """Eine verständliche Erzählung, wie Aurik die Herkunft der Musik erkannt hat."""
        ctx = self._context
        chain = ctx.get("transfer_chain") or []
        if not chain:
            return ""

        kurz = [_TRÄGER_KURZ.get(c, c) for c in chain]
        chain_str = " → ".join(kurz)
        num_stages = len(chain)
        first_stage = kurz[0] if kurz else "eine unbekannte Quelle"

        idx = self._chain_story_index % len(_KETTEN_ERZAEHLUNG)
        self._chain_story_index += 1
        return _KETTEN_ERZAEHLUNG[idx].format(chain=chain_str, num_stages=num_stages, first_stage=first_stage)

    def chain_summary(self) -> str:
        """Abschliessende Zusammenfassung der Tonträgerkette."""
        ctx = self._context
        chain = ctx.get("transfer_chain") or []
        material = ctx.get("material", "")

        if not chain and not material:
            return ""

        lang = [_TRÄGER_NAMEN.get(c, c) for c in chain]
        chain_lang = " → ".join(lang) if lang else _TRÄGER_NAMEN.get(material, material)

        teile = [
            "📀 So kam Deine Musik zu Dir: " + chain_lang,
            "",
            "Wie Aurik das herausgefunden hat:",
        ]

        if len(chain) >= 2:
            teile.append(
                f"Deine Musik hat {len(chain)} Stationen durchlaufen. "
                f"So wie ein altes Foto, das mehrfach kopiert wurde, "
                f"hat jede Kopie ihre eigenen Spuren hinterlassen. "
                f"Aurik erkennt diese Spuren und weiss genau, "
                f"wie es Deine Musik am besten behandeln kann."
            )
        elif chain:
            teile.append(
                f"Deine Musik stammt von {lang[0]}. Aurik hat das "
                f"an den typischen Merkmalen dieses Trägers erkannt — "
                f"so wie man eine Geige von einem Klavier unterscheiden kann."
            )
        elif material:
            teile.append(f"Deine Musik zeigt alle typischen Eigenschaften von {_TRÄGER_NAMEN.get(material, material)}.")

        if ctx.get("era_decade"):
            teile.append("")
            teile.append(
                f"Die Aufnahme stammt aus den {ctx['era_decade']}er Jahren — "
                f"eine Zeit mit einem ganz eigenen Klangcharakter."
            )

        return "\n".join(teile)

    # ── Fortschrittsmeldung erzeugen ────────────────────────────────────────

    def message_for(self, phase_id: str, phase_name: str = "", progress_pct: int = 0) -> str:
        now = _time.monotonic()
        phase_key = phase_id or "_unbekannt"
        ctx = self._context

        # ── Tonträgerketten-Interlude bei sehr langen Phasen (>20s) ────────
        chain = ctx.get("transfer_chain") or []
        zeit_seit_letztem = now - self._last_ts.get(phase_key, 0.0)
        if chain and zeit_seit_letztem > 20.0 and not self._chain_story_told:
            self._chain_story_told = True
            self._last_ts[phase_key] = now
            return f"📀 {self.chain_narrative()}"

        # ── "Warum diese Phase?" — nur beim ersten Mal ────────────────────
        erster_aufruf = phase_key not in self._used or not self._used[phase_key]
        if erster_aufruf and progress_pct < 15:
            warum = self._warum(phase_id)
            if warum:
                self._used.setdefault(phase_key, []).append(-1)
                self._last_ts[phase_key] = now
                return warum

        # ── Aktivitäts-Templates durchwechseln ─────────────────────────────
        templates = self._aktivitaeten(phase_id)
        used = self._used.setdefault(phase_key, [])

        if len(used) >= len(templates) + 1:
            used[:] = [i for i in used if i >= 0]
            if len(used) >= len(templates):
                used.clear()

        verfuegbar = [i for i in range(len(templates)) if i not in used]
        if not verfuegbar:
            used.clear()
            verfuegbar = list(range(len(templates)))

        if zeit_seit_letztem >= self._rotate_every_s:
            idx = verfuegbar[hash(f"{phase_key}_{int(now)}") % len(verfuegbar)]
            used.append(idx)
            self._last_ts[phase_key] = now
        else:
            idx = used[-1] if used and used[-1] >= 0 else verfuegbar[0]

        vorlage = templates[idx % len(templates)]

        # ── Präfix ─────────────────────────────────────────────────────────
        symbol = self._symbol(phase_id)
        if phase_name:
            praefix = f"{symbol} {phase_name}: "
        elif progress_pct < 12:
            _anfaenge = ["Starte: ", "Beginne: ", "Bereite vor: "]
            praefix = _anfaenge[idx % len(_anfaenge)]
        elif progress_pct >= 92:
            _enden = ["Fast fertig: ", "Letzter Schliff: ", "Abschluss: "]
            praefix = _enden[idx % len(_enden)]
        else:
            praefix = ""

        return f"{praefix}{vorlage}"

    # ── §v10.203 S1: Live-Chapter-Narration ────────────────────────────────

    def live_chapter(self, progress_pct: float) -> str | None:
        """Liefert bei Fortschritt-Schwellen ein narratives Kapitel.

        Returns None wenn keine Schwelle erreicht wurde oder das Kapitel
        bereits erzählt wurde. Andernfalls einen mehrzeiligen narrativen Text,
        der dem Nutzer erklärt, was Aurik gerade tut und WARUM.

        Kapitel:
          5%  — „Was wir vorgefunden haben" (Forensik-Erzählung)
          18% — „Was wir reparieren" (Defekt-Geschichten)
          45% — „Was wir verbessern" (Klang-Erzählung)
          78% — „Der Feinschliff" (Mastering-Narrative)
          95% — „Das Ergebnis" (Qualitäts-Vorschau)
        """
        pct = float(progress_pct)
        ctx = self._context
        chain = ctx.get("transfer_chain") or []
        material = ctx.get("material", "")
        defects = ctx.get("defects", [])
        era = ctx.get("era_decade")
        rest = ctx.get("restorability")

        _schwellen = [
            (5.0, "finding"),
            (18.0, "repairing"),
            (45.0, "enhancing"),
            (78.0, "polishing"),
            (95.0, "result"),
        ]

        for _schwelle, _kapitel_id in _schwellen:
            if pct >= _schwelle and _kapitel_id not in self._chapters_emitted:
                self._chapters_emitted.add(_kapitel_id)
                return self._build_chapter(_kapitel_id, chain, material, defects, era, rest)

        return None

    def _build_chapter(
        self,
        chapter_id: str,
        chain: list[str],
        material: str,
        defects: list[str],
        era: int | None,
        rest: float | None,
    ) -> str:
        """Baut ein narratives Kapitel aus Kontext-Daten."""
        import random as _random

        _rng = _random.Random(hash(chapter_id + self._session_key))

        _mat_name = _TRÄGER_NAMEN.get(material, "diesem Tonträger")
        _mat_kurz = _TRÄGER_KURZ.get(material, material or "Audio")
        _era_str = f"aus den {era}ern" if era else ""
        _rest_str = ""
        if rest is not None:
            if rest >= 70:
                _rest_str = "Die Aufnahme ist in einem Zustand, der eine sehr gute Restaurierung erwarten lässt."
            elif rest >= 45:
                _rest_str = "Die Aufnahme hat einige Herausforderungen, aber Aurik ist zuversichtlich."
            else:
                _rest_str = "Die Aufnahme ist stark degradiert. Aurik wird das Beste herausholen, aber nicht jeder Schaden lässt sich vollständig beheben."
        _chain_str = " → ".join([_TRÄGER_KURZ.get(c, c) for c in chain]) if chain else _mat_kurz
        _num_stages = len(chain) if chain else 1
        _first = _TRÄGER_NAMEN.get(chain[0], _mat_name) if chain else _mat_name

        _defect_list = ""
        if defects:
            _def_names = {
                "clicks": "Knackser",
                "crackle": "Knistern",
                "hum": "Netzbrummen",
                "noise_level": "Grundrauschen",
                "wow": "Gleichlaufschwankungen",
                "dropout": "Aussetzer",
                "clipping": "Übersteuerungen",
                "sibilance": "Zischlaute",
                "pops": "Pops",
            }
            _def_human = [_def_names.get(d, d) for d in defects[:5]]
            _defect_list = ", ".join(_def_human)

        if chapter_id == "finding":
            return self._chapter_finding(_mat_name, _chain_str, _num_stages, _first, _era_str, _rest_str, _rng)
        elif chapter_id == "repairing":
            return self._chapter_repairing(_mat_name, _defect_list, _era_str, _rng)
        elif chapter_id == "enhancing":
            return self._chapter_enhancing(_mat_name, _era_str, _chain_str, _rng)
        elif chapter_id == "polishing":
            return self._chapter_polishing(_mat_name, _era_str, _rng)
        elif chapter_id == "result":
            return self._chapter_result(_mat_name, _era_str, _rest_str, _rng)
        return ""

    def _chapter_finding(self, mat, chain, stages, first, era, rest, rng) -> str:
        _templates = [
            (
                f"📀 Kapitel 1: Was wir vorgefunden haben\n\n"
                f"Deine Musik stammt von {mat}. {era} — eine Zeit mit einem ganz "
                f"eigenen, unverwechselbaren Klang. {rest}\n\n"
                f"Deine Aufnahme hat {stages} Stationen durchlaufen: {chain}. "
                f"Das ist wie ein altes Foto, das mehrfach kopiert wurde — "
                f"jede Kopie hinterlässt ihre Spuren. Aurik hat diese Spuren "
                f"erkannt und weiss jetzt genau, wie es Deine Musik behandeln muss."
            ),
            (
                f"🔍 Kapitel 1: Die Spurensuche\n\n"
                f"So wie ein Archäologe die Geschichte einer antiken Vase entschlüsselt, "
                f"hat Aurik die Hörspuren in Deiner Musik untersucht. Das Ergebnis: "
                f"Deine Aufnahme begann als {first} und durchlief {stages} weitere "
                f"Stationen — {chain}.\n\n"
                f"{era} — jedes Jahrzehnt hat seinen eigenen Klang-Charakter. "
                f"{rest}"
            ),
            (
                f"🎧 Kapitel 1: Deine Musik hat eine Geschichte\n\n"
                f"{mat} {era} — das ist der Ursprung Deiner Aufnahme. "
                f"Seitdem hat sie einen weiten Weg zurückgelegt: {chain}.\n\n"
                f"Jede dieser Stationen hat Spuren hinterlassen — ein feines "
                f"Rauschen hier, ein leichtes Knistern da. Aurik hat diese "
                f"Spuren gelesen wie ein Detektiv. {rest}"
            ),
        ]
        return _templates[rng.randint(0, len(_templates) - 1)]  # type: ignore[no-any-return]

    def _chapter_repairing(self, mat, defects, era, rng) -> str:
        _def_str = f"{defects}" if defects else "verschiedene altersbedingte Störungen"
        _templates = [
            (
                f"🔧 Kapitel 2: Die Reparatur beginnt\n\n"
                f"Jetzt geht es ans Eingemachte. Aurik hat bei Deiner {mat} "
                f"{_def_str} entdeckt — die Spuren der Zeit.\n\n"
                f"Wie ein Restaurator, der ein Gemälde Schicht für Schicht reinigt, "
                f"entfernt Aurik jetzt jede einzelne Störung. Nicht mit Gewalt, "
                f"sondern mit chirurgischer Präzision. Nichts von der Musik "
                f"geht verloren — nur das, was nicht hingehört, wird entfernt."
            ),
            (
                f"🩹 Kapitel 2: Chirurgische Präzision\n\n"
                f"Deine {mat} hat {_def_str}. Aurik weiss genau, "
                f"wie es diese spezifischen Störungen behandeln muss. "
                f"{era} — das bedeutet besondere Sorgfalt.\n\n"
                f"Jeder Arbeitsschritt ist so kalibriert, dass nur die Störung "
                f"verschwindet. Die Musik dahinter bleibt unberührt — "
                f"als wäre die Zeit spurlos an ihr vorübergegangen."
            ),
        ]
        return _templates[rng.randint(0, len(_templates) - 1)]  # type: ignore[no-any-return]

    def _chapter_enhancing(self, mat, era, chain, rng) -> str:
        _templates = [
            (
                "✨ Kapitel 3: Die Wiederbelebung\n\n"
                "Die gröbsten Störungen sind entfernt. Jetzt kommt der "
                "magische Moment: Aurik bringt zurück, was die Zeit genommen hat.\n\n"
                "Verlorene Höhen werden rekonstruiert, die natürliche Wärme "
                "kehrt zurück, und die Musik bekommt wieder den Raum und die "
                "Tiefe, die sie einmal hatte. Es ist, als würde man ein "
                "Fenster öffnen und frische Luft hereinlassen."
            ),
            (
                f"🎚️ Kapitel 3: Der Klang erblüht\n\n"
                f"Nach der Reinigung kommt die Wiederherstellung. Deine {mat} "
                f"hat durch den Transfer über {chain} viel von ihrer ursprünglichen "
                f"Brillanz verloren. Aurik holt diese verlorenen Klangfarben "
                f"jetzt zurück — Frequenz für Frequenz, Obertöne für Obertöne.\n\n"
                f"{era} — Aurik weiss, wie Musik aus dieser Zeit "
                f"geklungen hat, und stellt diesen Charakter wieder her."
            ),
        ]
        return _templates[rng.randint(0, len(_templates) - 1)]  # type: ignore[no-any-return]

    def _chapter_polishing(self, mat, era, rng) -> str:
        _templates = [
            (
                "💎 Kapitel 4: Der Feinschliff\n\n"
                "Die grobe Arbeit ist getan. Jetzt wird Deine Musik poliert — "
                "wie ein Diamant, der den letzten Schliff bekommt.\n\n"
                "Die Lautstärke wird für jedes Wiedergabegerät optimiert, "
                "das Stereobild bekommt die perfekte Balance, und jedes "
                "kleine Detail wird noch einmal geprüft und verfeinert. "
                "Das ist der Unterschied zwischen 'gut' und 'atemberaubend'."
            ),
            (
                f"🎯 Kapitel 4: Die Perfektion\n\n"
                f"Deine {mat} aus den {era}ern — fast fertig. "
                f"Jetzt geht es um die Details, die den Unterschied machen: "
                f"Perfekte Lautstärke auf jedem Gerät, optimale Stereo-Balance, "
                f"letzte Anpassungen der Klangfarbe. Aurik arbeitet jetzt mit "
                f"der Sorgfalt eines Uhrmachers an den letzten Feinheiten."
            ),
        ]
        return _templates[rng.randint(0, len(_templates) - 1)]  # type: ignore[no-any-return]

    def _chapter_result(self, mat, era, rest, rng) -> str:
        _templates = [
            (
                f"🎵 Kapitel 5: Das Ergebnis\n\n"
                f"Deine {mat} aus den {era}ern ist fertig restauriert. "
                f"Gleich siehst Du das Qualitätsergebnis.\n\n"
                f"{rest}\n\n"
                f"Egal wie das Ergebnis ausfällt — Aurik hat sein Bestes getan. "
                f"Manche Aufnahmen sind zu stark beschädigt für eine perfekte "
                f"Wiederherstellung. Aber Aurik hat nichts verschlechtert — "
                f"das ist das wichtigste Prinzip."
            ),
        ]
        return _templates[rng.randint(0, len(_templates) - 1)]  # type: ignore[no-any-return]

    # ── Ende Live-Chapter-Narration ──────────────────────────────────────

    # ── §v10.203 S2: Chapter-Übergänge ──────────────────────────────────

    def chapter_transition(self, progress_pct: float) -> str | None:
        """Liefert einen Übergangstext wenn ein Kapitel endet und das nächste beginnt."""
        _transitions = {
            ("finding", "repairing"): [
                "Die Analyse ist abgeschlossen. Jetzt beginnt die eigentliche Arbeit — "
                "Aurik macht sich an die Reparatur.",
                "Aurik weiss jetzt genau, was zu tun ist. Die Reparatur-Phasen laufen an.",
            ],
            ("repairing", "enhancing"): [
                "Die gröbsten Störungen sind beseitigt. Jetzt geht es darum, "
                "den ursprünglichen Klang wiederherzustellen — die Seele der Musik.",
                "Reparatur abgeschlossen. Aurik wechselt jetzt in den "
                "Wiederherstellungs-Modus — hier entsteht der magische Klang.",
            ],
            ("enhancing", "polishing"): [
                "Die Musik klingt schon viel besser. Jetzt kommt der Feinschliff — "
                "der Unterschied zwischen 'gut' und 'hervorragend'.",
                "Fast geschafft. Aurik poliert jetzt jedes einzelne Detail "
                "auf Hochglanz. Das ist die Kür nach der Pflicht.",
            ],
            ("polishing", "result"): [
                "Der letzte Schliff ist getan. Aurik prüft jetzt das Ergebnis "
                "mit höchster Sorgfalt. Gleich siehst Du, was erreicht wurde.",
            ],
        }
        # Nur auslösen wenn wir zwischen Kapiteln wechseln
        _current = self._last_chapter
        for (_from, _to), _msgs in _transitions.items():
            if _current == _from and _to in self._chapters_emitted:
                self._last_chapter = _to
                import random as _rnd

                return _msgs[_rnd.randint(0, len(_msgs) - 1)]
        return None

    # ── §v10.203 S3: Post-Processing & Denker Narrative ──────────────────

    def post_processing_message(self, stage: str) -> str:
        """Narrative für Post-Processing und Denker-Phasen."""
        _msgs = {
            "precision_dropout": [
                "Präzisions-Feinabstimmung: Aurik entfernt letzte Artefakte, "
                "die nur in extrem hochauflösender Analyse sichtbar sind.",
            ],
            "vocal_scratch": [
                "Gesangs-Optimierung: Aurik verfeinert die Stimmklarheit — "
                "jedes Wort, jede Nuance wird herausgearbeitet.",
            ],
            "tape_head": [
                "Band-Kopf-Entzerrung: Aurik korrigiert die typischen "
                "Frequenzgang-Veränderungen, die jeder Bandkopf verursacht.",
            ],
            "smart_tape": [
                "Intelligente Band-Analyse: Aurik modelliert das exakte Bandverhalten und gleicht es perfekt aus.",
            ],
            "azimuth": [
                "Azimut-Präzisions-Korrektur: Aurik richtet die Stereo-Spuren "
                "haargenau aus — für maximale räumliche Tiefe.",
            ],
            "echo_removal": [
                "Echo-Entfernung: Aurik beseitigt unerwünschte Reflexionen, "
                "ohne den natürlichen Raumklang zu zerstören.",
            ],
            "export": [
                "Export-Vorbereitung: Aurik bereitet die Ausgabe im gewählten "
                "Format vor. Jedes Sample wird mit höchster Präzision berechnet.",
            ],
            "goosebumps": [
                "Gänsehaut-Optimierung: Aurik analysiert die emotionale "
                "Wirkung und verstärkt die Momente, die unter die Haut gehen.",
            ],
            "mdem": [
                "Multi-dimensionale Entzerrung: Aurik bearbeitet die letzte Ebene der Klang-Verbesserung.",
            ],
            "denker": [
                "Der Aurik-Denker prüft jetzt das gesamte Ergebnis: "
                "MUSHRA-Qualitätsbewertung, VERSA-Klangtreue, HPI-Verbesserung. "
                "Das ist der aufwendigste Teil — er dauert ein paar Minuten, "
                "aber ohne ihn gäbe es keine Qualitätsgarantie.",
            ],
            "excellence": [
                "Exzellenz-Prüfung: Aurik vergleicht das Ergebnis mit "
                "Referenzaufnahmen aus der gleichen Ära und dem gleichen Genre. "
                "Nur was diesen Vergleich besteht, wird ausgeliefert.",
            ],
        }
        _opts = _msgs.get(stage, [f"Post-Processing: {stage} — Aurik verfeinert das Ergebnis."])
        return _opts[hash(stage + self._session_key) % len(_opts)]

    # ── §v10.203 S4: Entdeckungs-Narrative ──────────────────────────────

    def discovery(self) -> str | None:
        """Liefert faszinierende narrative Entdeckungen aus der Forensik."""
        ctx = self._context
        chain = ctx.get("transfer_chain") or []
        material = ctx.get("material", "")
        era = ctx.get("era_decade")
        defects = ctx.get("defects", [])

        _discoveries = []

        if len(chain) >= 3 and "discovery_chain" not in self._discoveries_emitted:
            self._discoveries_emitted.add("discovery_chain")
            _chain_kurz = " → ".join([_TRÄGER_KURZ.get(c, c) for c in chain])
            _discoveries.append(
                f"🔎 Spannende Entdeckung: Deine Aufnahme hat {len(chain)} Generationen "
                f"durchlaufen — {_chain_kurz}. Das ist selten! Aurik hat spezielle "
                f"Mehrgenerationen-Algorithmen, die genau für solche Fälle entwickelt wurden."
            )

        if material == "mp3_low" and "discovery_mp3" not in self._discoveries_emitted:
            self._discoveries_emitted.add("discovery_mp3")
            _discoveries.append(
                "🔎 Interessant: Deine Aufnahme ist eine stark komprimierte MP3-Datei. "
                "Das bedeutet, dass beim Kodieren viele Klangdetails entfernt wurden. "
                "Aurik hat spezielle Algorithmen, um diese verlorenen Details zu "
                "rekonstruieren — fast wie ein Puzzle, bei dem die fehlenden Teile "
                "aus dem vorhandenen Bild erschlossen werden."
            )

        if era and era <= 1970 and "discovery_era" not in self._discoveries_emitted:
            self._discoveries_emitted.add("discovery_era")
            _discoveries.append(
                f"🔎 Deine Aufnahme stammt aus den {era}ern — einer Zeit, in der Musik "
                f"noch mit völlig anderen Mitteln aufgenommen wurde als heute. "
                f"Aurik berücksichtigt die damalige Studiotechnik, die typischen "
                f"Mikrofone und Mischpulte dieser Ära — und stellt den Klang so "
                f"wieder her, wie er im Studio geklungen haben muss."
            )

        if defects and len(defects) >= 4 and "discovery_defects" not in self._discoveries_emitted:
            self._discoveries_emitted.add("discovery_defects")
            _discoveries.append(
                "🔎 Deine Aufnahme zeigt gleich mehrere Arten von Schäden — "
                "das deutet auf eine bewegte Geschichte hin. Aurik behandelt "
                "jeden Schadenstyp mit einem eigenen, spezialisierten Werkzeug. "
                "Kein One-Size-Fits-All — sondern 43 verschiedene Spezialisten."
            )

        if _discoveries:
            import random as _rnd

            return _discoveries[_rnd.randint(0, len(_discoveries) - 1)]
        return None

    # ── §v10.203 S5: Handlungsempfehlungen ──────────────────────────────

    # ── §v10.203 S7: Entertainment-Intermezzo für lange Phasen ──────────

    def entertainment_intermezzo(self, phase_id: str, phase_duration_s: float) -> str | None:
        """Liefert bei langen Phasen ein unterhaltsames, song-individuelles Intermezzo.

        Das Intermezzo fühlt sich wie ein natürlicher Teil der Erzählung an —
        kein aus dem Kontext gerissenes Entertainment. Es knüpft an den aktuellen
        Pipeline-Abschnitt an, nutzt die konkreten Song-Daten (Era, Genre, Material)
        und leitet am Ende zurück zur laufenden Arbeit.

        Triggerschwellen (damit es nicht nervt):
          - Erstes Intermezzo nach 30s
          - Weiteres alle 60s, aber nie zweimal dasselbe Thema
        """
        ctx = self._context
        material = ctx.get("material", "")
        era = ctx.get("era_decade")
        chain = ctx.get("transfer_chain") or []
        defects = ctx.get("defects", [])

        # Nur bei wirklich langen Phasen (>30s)
        if phase_duration_s < 30.0:
            return None

        # Nicht zu oft — mindestens 45s Abstand zwischen Intermezzi
        _now = _time.monotonic()
        _last_emission = self._intermezzo_emitted.get(phase_id, 0.0)
        if _now - _last_emission < 45.0:
            return None
        self._intermezzo_emitted[phase_id] = _now

        # Kategorie auswählen (rotiert, Material-bewusst)
        _categories = self._available_categories(material, era, chain, defects)
        if not _categories:
            return None

        import random as _rnd

        _cat = _categories[hash(f"{phase_id}_{int(phase_duration_s // 30)}") % len(_categories)]

        # Intermezzo zusammenbauen: Einleitung + Inhalt + Rückleitung
        _intro = self._intermezzo_intro(phase_id)
        _content = self._intermezzo_content(_cat, material, era, chain, defects, _rnd)
        _outro = self._intermezzo_outro(phase_id)

        return f"{_intro}\n\n{_content}\n\n{_outro}"

    def _available_categories(self, material, era, chain, defects):
        """Ermittelt verfügbare Intermezzo-Kategorien basierend auf Song-Daten."""
        cats = ["general"]
        if era and era <= 1970:
            cats.append("era_vintage")
        if era and 1971 <= era <= 1999:
            cats.append("era_analog")
        if material in ("vinyl", "shellac"):
            cats.append("material_vinyl")
        if material in ("cassette_tape", "reel_tape", "cassette"):
            cats.append("material_tape")
        if material in ("mp3_low", "mp3_high", "aac"):
            cats.append("material_digital")
        if len(chain) >= 3:
            cats.append("chain_deep")
        if defects and len(defects) >= 3:
            cats.append("defects_rich")
        return cats

    def _intermezzo_intro(self, phase_id):
        _intros = [
            "Während Aurik weiterarbeitet, eine kleine Geschichte zu Deiner Musik:",
            "Solange die Algorithmen rechnen — ein interessanter Fakt am Rande:",
            "Die Analyse läuft. Zeit für einen kurzen Blick auf das, was Deine Musik so besonders macht:",
            "Aurik ist noch beschäftigt. Hier ein Gedanke, der Dir beim nächsten Hören auffallen wird:",
            "Im Hintergrund läuft die Arbeit. Wusstest Du eigentlich…",
        ]
        return _intros[hash(phase_id) % len(_intros)]

    def _intermezzo_outro(self, phase_id):
        _outros = [
            "… und jetzt zurück zur Arbeit — Aurik ist fast fertig mit diesem Schritt.",
            "… aber genug der Geschichten — die Algorithmen rufen. Gleich geht's weiter.",
            "… so, weiter im Programm: Aurik arbeitet noch ein bisschen an diesem Abschnitt.",
            "… und schon nähern wir uns dem nächsten Meilenstein. Aurik ist gleich durch.",
        ]
        return _outros[hash(phase_id + "outro") % len(_outros)]

    def _intermezzo_content(self, cat, material, era, chain, defects, rng):
        _content = {
            "general": [
                "Aurik analysiert Deine Musik mit über 60 verschiedenen Merkmalen — "
                "vom leisesten Rauschen bis zur höchsten Höhe. Das ist mehr, als die "
                "meisten menschlichen Toningenieure gleichzeitig im Ohr behalten können.",
                "Die Algorithmen, die hier laufen, wurden an Tausenden von Aufnahmen "
                "trainiert — aus allen Jahrzehnten, allen Genres, allen Tonträgern. "
                "Deine Musik profitiert von diesem geballten Wissen.",
            ],
            "era_vintage": [
                f"In den {era}ern wurde Musik noch mit Röhrenmikrofonen und "
                f"Bandmaschinen aufgenommen — jedes Gerät hatte seinen eigenen, "
                f"unverkennbaren Klang. Aurik kennt diese alten Schätze und "
                f"weiss genau, wie sie geklungen haben müssen.",
                f"Die {era}er — das war die Zeit von {self._era_factoid(era)}. "
                f"Deine Aufnahme trägt den Klang dieser Ära in sich.",
            ],
            "era_analog": [
                f"In den {era}ern war die Musikproduktion komplett analog — "
                f"keine Computer, keine digitalen Effekte. Nur Bandmaschinen, "
                f"Mischpulte und das geschulte Ohr des Toningenieurs. "
                f"Aurik stellt diesen warmen, analogen Klang wieder her.",
            ],
            "material_vinyl": [
                "Schallplatten sind faszinierend: Die Musik ist buchstäblich "
                "in die Rillen eingraviert. Eine Nadel fährt hindurch und "
                "übersetzt die Vibrationen zurück in Schall. Jede noch so "
                "kleine Beschädigung der Rille wird hörbar — und Aurik "
                "kann sie erkennen und reparieren.",
                "Vinyl hat einen ganz eigenen, warmen Klang — das liegt am "
                "RIAA-Entzerrungsstandard, der seit 1954 weltweit verwendet wird. "
                "Aurik weiss das und behandelt Vinyl-Aufnahmen entsprechend.",
            ],
            "material_tape": [
                "Kassetten wurden 1963 von Philips erfunden und eroberten die Welt "
                "im Sturm. Endlich konnte man Musik überallhin mitnehmen! Der Preis: "
                "ein höheres Rauschen und die berüchtigten Bandsalat-Momente. "
                "Aurik kennt alle typischen Kassettenschäden auswendig.",
                "Tonbänder speichern Musik als magnetische Muster. Mit der Zeit "
                "verblassen diese Muster — die hohen Frequenzen verschwinden zuerst. "
                "Aurik kann diese verlorenen Höhen aus dem verbleibenden Signal "
                "rekonstruieren.",
            ],
            "material_digital": [
                "MP3-Dateien sparen Platz, indem sie Töne weglassen, die das "
                "menschliche Ohr angeblich nicht hört. Das Problem: Das Ohr hört "
                "sie DOCH — zumindest indirekt. Aurik kann diese fehlenden "
                "Informationen aus dem Kontext erschliessen und wiederherstellen.",
                "Digitale Kompression ist wie ein Puzzle, bei dem 80% der Teile "
                "fehlen. Aurik errät die fehlenden Teile aus den verbleibenden — "
                "mit erstaunlicher Genauigkeit.",
            ],
            "chain_deep": [
                f"Deine Aufnahme hat {len(chain)} Stationen durchlaufen. Jede "
                f"davon hat Spuren hinterlassen — wie Schichten in einem "
                f"archäologischen Fund. Aurik arbeitet sich durch diese Schichten "
                f"hindurch, um zum ursprünglichen Klang vorzudringen.",
                "Mehrgenerationen-Aufnahmen sind selten und wertvoll — sie erzählen "
                "eine Geschichte. Von der ersten Aufnahme bis zur Datei, die Du "
                "geöffnet hast, ist Deine Musik durch viele Hände gegangen. "
                "Aurik ehrt diesen Weg.",
            ],
            "defects_rich": [
                "Jede Art von Schaden erzählt eine eigene Geschichte: Knackser "
                "von der Plattenspielernadel, Rauschen vom Band, dumpfer Klang "
                "von der MP3-Kompression. Aurik liest diese Geschichten wie "
                "ein Archäologe — und repariert sie Schicht für Schicht.",
            ],
        }
        opts = _content.get(cat, _content["general"])
        return opts[rng.randint(0, len(opts) - 1)]

    @staticmethod
    def _era_factoid(era):
        _facts = {
            1950: "Rock'n'Roll, Petticoats und der Beginn der Jugendkultur",
            1960: "der Beatles, der Mondlandung und des Transistorradios",
            1970: "von ABBA, Disco und dem Siegeszug der Musikkassette",
            1980: "der CD, der ersten Synthesizer und MTV",
            1990: "der ersten MP3s, des Internets und Napster",
        }
        decade = (era // 10) * 10
        return _facts.get(decade, "einer ganz besonderen musikalischen Ära")

    # ── Ende Entertainment-Intermezzo ───────────────────────────────────

    def recommendation(self, was_reverted: bool = False, quality_score: float = 0.0) -> str | None:
        """Liefert eine kontext-bewusste Handlungsempfehlung für den nächsten Lauf."""
        ctx = self._context
        # material and chain context (material not yet used in current logic)
        chain = ctx.get("transfer_chain") or []
        era = ctx.get("era_decade")

        if was_reverted:
            _recs = [
                "💡 Tipp fürs nächste Mal: Der Studio-2026-Modus erlaubt mehr "
                "Eingriffstiefe und könnte bei dieser Aufnahme bessere Ergebnisse "
                "erzielen. Er ist weniger konservativ als der Restoration-Modus.",
                "💡 Diese Aufnahme ist stark degradiert. Die 'Leichte Reinigung' "
                "könnte besser funktionieren als eine Vollrestauration — "
                "manchmal ist weniger mehr.",
            ]
            import random as _rnd

            return _rnd.choice(_recs)

        if quality_score < 55.0:
            return (
                "💡 Das Ergebnis ist okay, aber nicht überragend. Beim nächsten Mal "
                "könntest Du es mit dem Studio-2026-Modus versuchen — "
                "er ist mutiger und holt oft mehr aus schwierigem Material heraus."
            )

        if len(chain) >= 3:
            return (
                "💡 Deine Musik hat mehrere Generationen durchlaufen. Für "
                "zukünftige Restaurierungen: Wenn Du Zugang zu einer früheren "
                "Generation hast (z.B. die Original-Kassette statt der MP3-Kopie), "
                "wird das Ergebnis noch besser. Aurik arbeitet am liebsten "
                "mit dem frühestmöglichen Träger."
            )

        return None

    def _warum(self, phase_id: str) -> str:
        pid = str(phase_id or "").lower()
        ctx = self._context
        material = ctx.get("material", "")
        for schluessel, varianten in _WARUM_DIESE_PHASE.items():
            if schluessel in pid or pid.startswith(schluessel):
                if material and material in varianten:
                    texte = varianten[material]
                else:
                    texte = varianten.get("_default", [])
                if texte:
                    anzahl = len(ctx.get("defects", []) or [])
                    return texte[hash(f"{pid}_{self._session_key}") % len(texte)].format(
                        material=_TRÄGER_NAMEN.get(material, material or "diesem Träger"), defect_count=anzahl
                    )
        return ""

    def _aktivitaeten(self, phase_id: str) -> list[str]:
        pid = str(phase_id or "").lower()
        for schluessel in _AKTIVITAETEN:
            if schluessel in pid or pid.startswith(schluessel):
                return _AKTIVITAETEN[schluessel]
        return _ALLGEMEINE_AKTIVITAETEN

    @staticmethod
    def _symbol(phase_id: str) -> str:
        pid = str(phase_id or "").lower()
        if "rausch" in pid or "noise" in pid or "denoise" in pid:
            return "🔇"
        if "knack" in pid or "click" in pid or "crackle" in pid:
            return "🔍"
        if "klang" in pid or "eq" in pid or "frequenz" in pid:
            return "🎚️"
        if "harmonisch" in pid or "warm" in pid:
            return "🔥"
        if "stimme" in pid or "vocal" in pid or "gesang" in pid:
            return "🎤"
        if "stereo" in pid or "raum" in pid:
            return "🎧"
        if "master" in pid or "polish" in pid or "schliff" in pid:
            return "✨"
        if "export" in pid or "speichern" in pid:
            return "💾"
        if "rumble" in pid or "brumm" in pid or "hum" in pid:
            return "📉"
        if "reparatur" in pid or "repair" in pid:
            return "🔧"
        if "wow" in pid or "flutter" in pid or "gleichlauf" in pid:
            return "〰️"
        if "tempo" in pid or "speed" in pid or "pitch" in pid:
            return "⏱️"
        if "laut" in pid or "loudness" in pid:
            return "📊"
        if "hall" in pid or "reverb" in pid:
            return "🏠"
        if "scan" in pid or "analys" in pid or "untersuch" in pid:
            return "🔬"
        return "⚙️"


# Singleton
_erzaehler: PhaseProgressNarrator | None = None


def get_narrator() -> PhaseProgressNarrator:
    global _erzaehler
    if _erzaehler is None:
        _erzaehler = PhaseProgressNarrator()
    return _erzaehler
