"""
Phase DAG — §7.5a [RELEASE_MUST]
=================================

Formaler Abhängigkeitsgraph der Aurik-Pipeline-Phasen.
Definiert HARD_BEFORE-Constraints, INDEPENDENT-Gruppen und CONFLICT-Paare.

Spec: 06_phases_system.md §7.5a (v10.0.0)
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseConstraint:
    """A muss vor B ausgeführt worden sein (HARD_BEFORE)."""

    before: str  # Phase-ID die zuerst laufen muss
    after: str  # Phase-ID die danach läuft
    reason: str  # Normative Begründung


@dataclass(frozen=True)
class ConflictPair:
    """Zwei Phasen dürfen nicht gleichzeitig aktiv sein."""

    phase_a: str
    phase_b: str
    reason: str


# ---------------------------------------------------------------------------
# Normative HARD_BEFORE-Constraints (§7.5a)
# ---------------------------------------------------------------------------

HARD_BEFORE_CONSTRAINTS: list[PhaseConstraint] = [
    # phase_01 muss IMMER zuerst laufen (DC-Offset → alle anderen)
    PhaseConstraint("phase_01_click_removal", "phase_03_denoise", "DC/Click vor ML-NR"),
    PhaseConstraint("phase_01_click_removal", "phase_06_frequency_restoration", "DC-Offset vor BW-Extension"),
    PhaseConstraint("phase_01_click_removal", "phase_07_harmonic_restoration", "DC vor Harmonik"),
    PhaseConstraint("phase_01_click_removal", "phase_09_crackle_removal", "DC vor Crackle"),
    PhaseConstraint("phase_01_click_removal", "phase_12_wow_flutter_fix", "DC vor Wow/Flutter"),
    PhaseConstraint("phase_01_click_removal", "phase_18_noise_gate", "DC vor Noise-Gate"),
    PhaseConstraint("phase_01_click_removal", "phase_29_tape_hiss_reduction", "DC vor Band-NR"),
    # §v10.94 Non-Plus-Ultra: Hum-Entfernung VOR ML-Denoising.
    # P02 notched 50/60 Hz + Harmonische (bis 400-480 Hz). P03 (ML-NR) trainiert
    # auf dem Restsignal — ohne Hum-Entfernung lernt das ML-Modell Brumm-Harmonische
    # als "Signal" und entfernt Musikinhalt in den betroffenen Bändern.
    PhaseConstraint(
        "phase_02_hum_removal",
        "phase_03_denoise",
        "Hum-Notch-Filter vor ML-NR (§v10.94: verhindert Brumm-Lernen als Signal)",
    ),
    # NR vor Harmonik (Phase_03 → Phase_06/07)
    PhaseConstraint(
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "NR vor BW-Extension (kein Rauschen in Extended-Harmonics)",
    ),
    PhaseConstraint(
        "phase_03_denoise", "phase_07_harmonic_restoration", "NR vor Harmonik-Enhancement (§2.46 Stufe 4 vor 5)"
    ),
    # Carrier-Chain-Reihenfolge (§2.46 Stufe 4 → Stufe 5)
    PhaseConstraint(
        "phase_29_tape_hiss_reduction",
        "phase_07_harmonic_restoration",
        "Band-NR vor Harmonik (Rauschen nicht als Harmonik rekonstruieren)",
    ),
    # Pegeldrift-Korrektur (§2.46 Stufe 4.5): nach subtraktiver Carrier-NR,
    # vor additiver BW-/Harmonik-Restauration, damit weder Rauschen noch
    # synthetische Obertöne die Gain-Hüllkurve verfälschen.
    PhaseConstraint(
        "phase_03_denoise",
        "phase_40_loudness_normalization",
        "Carrier-NR vor Pegeldrift-Ausgleich (§2.46 Stufe 4 vor 4.5)",
    ),
    PhaseConstraint(
        "phase_29_tape_hiss_reduction",
        "phase_40_loudness_normalization",
        "Band-NR vor Pegeldrift-Ausgleich (§2.46 Stufe 4 vor 4.5)",
    ),
    PhaseConstraint(
        "phase_40_loudness_normalization",
        "phase_06_frequency_restoration",
        "Pegeldrift-Ausgleich vor BW-Extension (§2.46 Stufe 4.5 vor 5)",
    ),
    PhaseConstraint(
        "phase_40_loudness_normalization",
        "phase_07_harmonic_restoration",
        "Pegeldrift-Ausgleich vor Harmonik (§2.46 Stufe 4.5 vor 5)",
    ),
    PhaseConstraint(
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
        "BW-Extension vor Harmonik (Stufe 5 intern geordnet)",
    ),
    # Finale Export-Sicherheitskette: LUFS/Gain vor TruePeak, TruePeak vor Format/Dither.
    PhaseConstraint(
        "phase_17_mastering_polish",
        "phase_40_loudness_normalization",
        "Mastering-Polish vor LUFS-/Gain-Normierung (keine Pegeländerung nach Final-LUFS)",
    ),
    PhaseConstraint(
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
        "LUFS-/Gain-Normierung vor TruePeak-Limiter (Limiter darf nicht nachträglich skaliert werden)",
    ),
    PhaseConstraint(
        "phase_47_truepeak_limiter",
        "phase_41_output_format_optimization",
        "TruePeak-Limiter vor Ausgabeformat/Dither (finaler Peak-Schutz vor Exportformat)",
    ),
    # §III (copilot-instructions.md): Glue Stage als vorletzte Phase in ALLEN Modi —
    # nach TruePeak/LUFS, vor Dithering/Output-Format.
    PhaseConstraint(
        "phase_47_truepeak_limiter",
        "phase_glue_stage",
        "Glue Stage nach TruePeak-Limiter (keine Bus-Kompression nach finalem Peak-Schutz)",
    ),
    PhaseConstraint(
        "phase_40_loudness_normalization",
        "phase_glue_stage",
        "Glue Stage nach LUFS-/Gain-Normierung auch wenn TruePeak nicht aktiv ist",
    ),
    PhaseConstraint(
        "phase_17_mastering_polish",
        "phase_glue_stage",
        "Glue Stage nach Mastering-Polish (Studio-Modus)",
    ),
    PhaseConstraint(
        "phase_glue_stage",
        "phase_41_output_format_optimization",
        "Glue Stage vor Ausgabeformat/Dither (vorletzte Phase in ALLEN Modi)",
    ),
    PhaseConstraint(
        "phase_40_loudness_normalization",
        "phase_41_output_format_optimization",
        "LUFS-/Gain-Normierung vor Ausgabeformat auch wenn TruePeak nicht aktiv ist",
    ),
    # Crackle vor Noise-Gate (Phase_09 → Phase_18)
    PhaseConstraint(
        "phase_09_crackle_removal",
        "phase_18_noise_gate",
        "Crackle vor NR/Gate (Crackle-Residuen nicht als Rauschen gaten)",
    ),
    # Wow/Flutter vor Azimuth (Phase_12 → Phase_25)
    PhaseConstraint(
        "phase_12_wow_flutter_fix",
        "phase_25_azimuth_correction",
        "Wow/Flutter-Korrektur vor Azimuth (Zeit-Alignement zuerst)",
    ),
    # Dropout vor BW-Extension (Phase_24 → Phase_06)
    PhaseConstraint(
        "phase_24_dropout_repair",
        "phase_06_frequency_restoration",
        "Dropout-Reparatur vor BW-Extension (keine Lücken in erweitertem Spektrum)",
    ),
    # ADC-Artefakt-Reihenfolge: die frühere Quantisierungs-NR-Phase existiert in UV3 nicht mehr.
    # Daher hier kein phase_31-Constraint mehr — sonst entstehen False-Positives mit
    # phase_31_speed_pitch_correction, das semantisch ein anderer Verarbeitungsschritt ist.
]

# Kurzform-Map: "phase_XX" → Vollname (für validate_phase_order)
_SHORT_TO_FULL: dict[str, str] = {
    "phase_01": "phase_01_click_removal",
    "phase_03": "phase_03_denoise",
    "phase_06": "phase_06_frequency_restoration",
    "phase_07": "phase_07_harmonic_restoration",
    "phase_09": "phase_09_crackle_removal",
    "phase_12": "phase_12_wow_flutter_fix",
    "phase_18": "phase_18_noise_gate",
    "phase_24": "phase_24_dropout_repair",
    "phase_25": "phase_25_azimuth_correction",
    "phase_29": "phase_29_tape_hiss_reduction",
    "phase_30": "phase_30_dc_offset_removal",
    "phase_40": "phase_40_loudness_normalization",
    "phase_41": "phase_41_output_format_optimization",
    "phase_47": "phase_47_truepeak_limiter",
}


# ---------------------------------------------------------------------------
# CONFLICT-Paare (aus §2.29e CONFLICT_REGISTRY)
# ---------------------------------------------------------------------------

CONFLICT_PAIRS: list[ConflictPair] = [
    ConflictPair(
        "phase_06_bw_extension",
        "phase_23_audio_sr_upsampling",
        "Beide erweitern Bandbreite — nur eine aktiv (Phase_06 hat Vorrang bei Tape/Shellac)",
    ),
    ConflictPair(
        "phase_21_exciter",
        "phase_07_harmonic_restoration",
        "§0a: phase_21 (Exciter) ist in Restoration VERBOTEN — nie gleichzeitig mit phase_07",
    ),
]


# ---------------------------------------------------------------------------
# Parallelisierungs-Klassen (§7.5a)
# ---------------------------------------------------------------------------

INDEPENDENT_CLASS_A = frozenset(
    {
        "phase_14_stereo_width",
        "phase_15_stereo_field_repair",
        "phase_25_azimuth_correction",
    }
)
# Klasse A: Stereo/Phase — parallel ausführbar nach phase_12.

INDEPENDENT_CLASS_B = frozenset(
    {
        "phase_09_crackle_removal",
        "phase_24_dropout_repair",
    }
)
# Klasse B: Lokale Defekte — parallel nach phase_01; Defekttypen überlappen nicht.

INDEPENDENT_CLASS_C = frozenset(
    {
        "phase_05_hum_removal",
        "phase_11_spectral_repair",
    }
)
# Klasse C: Analyse/leichtgewichtig — kann parallel zu anderen laufen.


# ---------------------------------------------------------------------------
# Validierungs-API (§7.5a)
# ---------------------------------------------------------------------------


def _normalize_phase_id(phase_id: str) -> str:
    """Normiert Kurzform (phase_03) auf Langform (phase_03_denoise) wenn möglich."""
    phase_id = phase_id.strip().lower()
    # Bereits in Kurzform ohne Suffix?
    for short, full in _SHORT_TO_FULL.items():
        if phase_id == short:
            return full
        if phase_id.startswith(short + "_"):
            return phase_id  # bereits Langform
    return phase_id


_PHASE_ID_RE = re.compile(r"^phase_(\d+)(?:_|$)")


def _phase_num(pid: str) -> int:
    """Extrahiert die Phasennummer aus einer Phase-ID (Tiebreaker für Sortierung).

    §III (copilot-instructions.md): `phase_glue_stage` ist die vorletzte Phase —
    ihr Sortierschlüssel liegt hinter allen nummerierten Phasen (die exakte
    Position regeln die HARD_BEFORE-Constraints). Unbekannte IDs erhalten den
    neutralen Schlüssel 999. Kein Exception-Ersatzpfad (§V7 (copilot-instructions.md)):
    Die Glue-Stage ist ein definierter Fall, kein Parse-Fehler.
    """
    if pid == "phase_glue_stage":
        return 1000
    match = _PHASE_ID_RE.match(pid)
    if match:
        return int(match.group(1))
    return 999


def sort_phases_by_dag(phase_list: list[str]) -> list[str]:
    """Sortiert phase_list in topologisch korrekter Reihenfolge gemäß HARD_BEFORE-Constraints.

    Verwendet Kahn's Algorithmus. Phasen ohne aktive Constraints werden nach ihrer
    numerischen Phasen-ID (stabiler Tiebreaker) sortiert — entspricht der Designreihenfolge.

    Args:
        phase_list: Beliebige Reihenfolge von Phase-IDs (kurz- oder langform).

    Returns:
        Phasen in gültiger Ausführungsreihenfolge (alle aktiven HARD_BEFORE-Constraints erfüllt).
        Falls ein Zyklus erkannt wird (sollte nie passieren), wird numerischer Sort-Fallback genutzt.
    """
    if len(phase_list) <= 1:
        return list(phase_list)

    # Normierung: original → normalized, normalized → first-seen-original
    norm_map: dict[str, str] = {}  # orig → normalized
    norm_to_orig: dict[str, str] = {}  # normalized → orig (first occurrence)
    for orig in phase_list:
        norm = _normalize_phase_id(orig)
        norm_map[orig] = norm
        if norm not in norm_to_orig:
            norm_to_orig[norm] = orig

    active_norms: set[str] = set(norm_map.values())

    # Adjazenzliste und Eingangsgrad für aktive Phasen
    in_degree: dict[str, int] = dict.fromkeys(active_norms, 0)
    edges: dict[str, list[str]] = {n: [] for n in active_norms}

    for c in HARD_BEFORE_CONSTRAINTS:
        b = _normalize_phase_id(c.before)
        a = _normalize_phase_id(c.after)
        if b in active_norms and a in active_norms and a not in edges[b]:
            edges[b].append(a)
            in_degree[a] += 1

    # Kahn's BFS — Startknoten mit Eingangsgrad 0, numerisch sortiert (Tiebreaker)
    def _orig_num(norm: str) -> int:
        return _phase_num(norm_to_orig.get(norm, norm))

    ready: list[str] = sorted([n for n, d in in_degree.items() if d == 0], key=_orig_num)
    result: list[str] = []

    while ready:
        node = ready.pop(0)
        result.append(norm_to_orig.get(node, node))
        new_ready: list[str] = []
        for neighbor in edges[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                new_ready.append(neighbor)
        if new_ready:
            ready = sorted(ready + new_ready, key=_orig_num)

    if len(result) < len(phase_list):
        logger.warning(
            "sort_phases_by_dag: Zyklus in Verarbeitungsschritt-DAG erkannt — Ersatzpfad auf numerische Sortierung. Verbleibend: %s",
            [norm_to_orig.get(n, n) for n in active_norms if n not in {norm_map[r] for r in result}],
        )
        return sorted(phase_list, key=_phase_num)

    return result


def validate_phase_order(phase_list: list[str]) -> list[str]:
    """Prüft eine geordnete Phase-Liste gegen HARD_BEFORE-Constraints.

    Args:
        phase_list: Geordnete Liste von Phase-IDs (kurz- oder langform).

    Returns:
        Liste von Constraint-Verletzungen (leer = korrekt).
        Format: "phase_07_harmonic_restoration kommt vor phase_03_denoise (NR vor Harmonik)"
    """
    normalized = [_normalize_phase_id(p) for p in phase_list]
    violations = []

    for constraint in HARD_BEFORE_CONSTRAINTS:
        b = _normalize_phase_id(constraint.before)
        a = _normalize_phase_id(constraint.after)

        # Nur prüfen wenn beide in der Liste sind
        b_indices = [i for i, p in enumerate(normalized) if p == b]
        a_indices = [i for i, p in enumerate(normalized) if p == a]

        if not b_indices or not a_indices:
            continue  # eine der Phasen nicht aktiv → Constraint irrelevant

        b_idx = min(b_indices)
        a_idx = min(a_indices)
        if a_idx < b_idx:
            violations.append(f"{a} kommt vor {b} — Verletzung: {constraint.reason}")

    return violations


def check_conflict(phase_a: str, phase_b: str) -> str | None:
    """Prüft ob zwei Phasen im Konflikt stehen.

    Returns:
        Konflikt-Beschreibung wenn vorhanden, sonst None.
    """
    a = _normalize_phase_id(phase_a)
    b = _normalize_phase_id(phase_b)

    for pair in CONFLICT_PAIRS:
        pa = _normalize_phase_id(pair.phase_a)
        pb = _normalize_phase_id(pair.phase_b)
        if (a.startswith(pa[:8]) and b.startswith(pb[:8])) or (a.startswith(pb[:8]) and b.startswith(pa[:8])):
            return pair.reason

    return None


def get_parallel_class(phase_id: str) -> str | None:
    """Gibt die Parallelisierungs-Klasse zurück ('A', 'B', 'C') oder None."""
    p = _normalize_phase_id(phase_id)
    for cls_name, cls_set in [("A", INDEPENDENT_CLASS_A), ("B", INDEPENDENT_CLASS_B), ("C", INDEPENDENT_CLASS_C)]:
        for member in cls_set:
            if p.startswith(member[:11]):  # "phase_XX_" prefix
                return cls_name
    return None
