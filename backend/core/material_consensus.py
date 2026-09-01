"""Material-Konsens — Löst Widersprüche zwischen den 3 Detektoren auf.

Problem: MediumDetector, EraClassifier und DefectScanner laufen unabhängig
und können widersprüchliche Material-Typen liefern (z.B. mp3_high vs. vinyl vs. cassette).

Lösung: Gewichteter Konsens mit Konfidenz-basierter Auflösung.
- MediumDetector: höchstes Gewicht (physikalische Signalanalyse)
- EraClassifier: mittleres Gewicht (Ära → Material-Inferenz)
- DefectScanner: ergänzend (Defektmuster → Material)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MATERIAL_WEIGHTS = {
    "medium_detector": 0.50,  # Physikalische Trägermedium-Analyse (autoritativ)
    "era_classifier": 0.30,  # Ära → Material-Inferenz (korrelativ)
    "defect_scanner": 0.20,  # Defektmuster → Material (indirekt)
}


def resolve_material_consensus(
    medium_result: dict[str, Any] | None = None,
    era_result: dict[str, Any] | None = None,
    defect_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Löst Material-Widersprüche gewichtet auf.

    Args:
        medium_result: {"material": "vinyl", "confidence": 0.85, "chain": "vinyl_direct"}
        era_result:    {"material": "cassette", "decade": 1985, "confidence": 0.60}
        defect_result: {"material": "cassette", "score": 5.39}

    Returns:
        {"material": "vinyl", "confidence": 0.72, "source": "medium_detector",
         "all_votes": {...}, "conflict_detected": True/False}
    """
    votes: dict[str, float] = {}
    details: dict[str, Any] = {}

    # Sammle gewichtete Stimmen
    if medium_result and medium_result.get("material"):
        mat = medium_result["material"]
        conf = medium_result.get("confidence", 0.5)
        weight = MATERIAL_WEIGHTS["medium_detector"]
        votes[mat] = votes.get(mat, 0) + conf * weight
        details["medium_detector"] = {
            "material": mat,
            "confidence": conf,
            "chain": medium_result.get("chain", "unknown"),
        }

    if era_result and era_result.get("material"):
        mat = era_result["material"]
        conf = era_result.get("confidence", 0.5)
        weight = MATERIAL_WEIGHTS["era_classifier"]
        votes[mat] = votes.get(mat, 0) + conf * weight
        details["era_classifier"] = {"material": mat, "confidence": conf, "decade": era_result.get("decade", 0)}

    if defect_result and defect_result.get("material"):
        mat = defect_result["material"]
        conf = min(defect_result.get("score", 5.0) / 10.0, 1.0)
        weight = MATERIAL_WEIGHTS["defect_scanner"]
        votes[mat] = votes.get(mat, 0) + conf * weight
        details["defect_scanner"] = {"material": mat, "score": defect_result.get("score", 0)}

    # §v10.14: Defect-per-Material affinity scores (§v10.304.14).
    # Jeder Defekttyp hat eine bekannte Material-Affinität (z.B. crackle→vinyl).
    # Die pro-Material aggregierte Severity wird als zusätzliche Stimme eingewoben.
    # Dies gibt dem DefectScanner eine VOICE im Konsens, selbst wenn sein
    # primary material_type vom MediumDetector abweicht.
    if defect_result and defect_result.get("material_scores"):
        _mat_scores: dict[str, float] = defect_result["material_scores"]
        _total_sev = sum(_mat_scores.values())
        if _total_sev > 0.0:
            _weight = MATERIAL_WEIGHTS["defect_scanner"] * 0.6  # 60 % des defect-Gewichts für Affinitäten
            for _mat, _sev in _mat_scores.items():
                _norm_sev = _sev / _total_sev  # normalisiert auf [0, 1]
                votes[_mat] = votes.get(_mat, 0.0) + _norm_sev * _weight
            details["defect_affinities"] = _mat_scores

    # §v10.14.1 Era-Consistency Boost: Wenn der EraClassifier eine hohe
    # Konfidenz (>0.50) hat und sein material_prior mit EINEM der votierten
    # Materialien übereinstimmt, bekommt dieses Material einen Boost.
    # Logik: Der EraClassifier hat die Aufnahme-Ära aus physikalischen
    # Signalcharakteristika (BW, SNR, Stereo) abgeleitet. Wenn diese
    # Charakteristika konsistent mit dem Material sind, ist das ein
    # unabhängiger Validierungspunkt — kein Zirkelschluss.
    if era_result and era_result.get("material") and era_result.get("confidence", 0) >= 0.50:
        _era_mat = str(era_result["material"]).lower()
        _era_conf = float(era_result.get("confidence", 0.5))
        if _era_mat in votes:
            # Era-Material-Konsistenz-Boost: +0.10 bei conf≥0.60, +0.20 bei conf≥0.75
            _era_boost = 0.10 if _era_conf >= 0.60 else 0.05
            if _era_conf >= 0.75:
                _era_boost = 0.20
            _era_boost *= MATERIAL_WEIGHTS["era_classifier"]
            votes[_era_mat] += _era_boost
            details["era_consistency_boost"] = {
                "material": _era_mat,
                "boost": round(_era_boost, 3),
                "era_confidence": _era_conf,
            }
            logger.debug(
                "Material-Konsens: Era-Boost +%.3f für %s (era_conf=%.2f)",
                _era_boost,
                _era_mat,
                _era_conf,
            )

    # §v10.14.1 Chain-Consistency Penalty: Wenn die Tonträgerkette
    # (vom MediumDetector) bekannte Träger enthält, die ein votiertes
    # Material NICHT enthält, wird dieses Material penalisiert.
    # Beispiel: chain=['vinyl','mp3_high'] + material='cassette' →
    # Cassette ist NICHT in der Chain → Penalty.
    if medium_result and medium_result.get("chain"):
        _chain_materials: set[str] = set()
        for _cm in str(medium_result.get("chain", "")).replace(" → ", "→").split("→"):
            _cm = _cm.strip().lower().replace(" ", "_").replace("-", "_")
            if _cm and _cm != "unknown":
                _chain_materials.add(_cm)
        if len(_chain_materials) >= 1:
            for _mat in list(votes.keys()):
                _mat_key = _mat.lower().replace(" ", "_").replace("-", "_")
                if _mat_key not in _chain_materials and _mat_key != "unknown":
                    # Material nicht in der Chain → starke Penalty
                    # (aber nicht auf 0 — Defektmuster können override)
                    _penalty = 0.60  # 60% Reduktion
                    votes[_mat] *= 1.0 - _penalty
                    logger.debug(
                        "Material-Konsens: Chain-Penalty für %s (nicht in chain %s)",
                        _mat,
                        _chain_materials,
                    )

    if not votes:
        return {
            "material": "unknown",
            "confidence": 0.0,
            "source": "none",
            "all_votes": details,
            "conflict_detected": False,
        }

    # Gewinner mit höchstem gewichtetem Score
    best_material = max(votes.items(), key=lambda x: x[1])
    total_weight = sum(MATERIAL_WEIGHTS.values())
    normalized_confidence = best_material[1] / total_weight

    # Konflikt-Erkennung
    unique_materials = {d["material"] for d in details.values()}
    conflict_detected = len(unique_materials) > 1

    if conflict_detected:
        logger.debug(
            "Material-Konsens: KONFLIKT — %s (gewählt: %s, Konfidenz: %.2f)",
            {k: v["material"] for k, v in details.items()},
            best_material[0],
            normalized_confidence,
        )
    else:
        logger.info("Material-Konsens: EINSTIMMIG — %s (%.2f)", best_material[0], normalized_confidence)

    return {
        "material": best_material[0],
        "confidence": round(normalized_confidence, 2),
        "source": max(details.items(), key=lambda x: x[1].get("confidence", 0))[0],
        "all_votes": details,
        "conflict_detected": conflict_detected,
    }


def validate_material_era_consistency(material: str, decade: int, transfer_chain: list[str] | None = None) -> bool:
    """Prüft ob das erkannte Material zur Ära passt (§v10.14.1).

    ZWEI Validierungsregeln:
    1. Produktions-Ende-Regel: Ein physisches Trägermedium (Shellac, Wachswalze)
       wurde nach einem bestimmten Datum nicht mehr produziert. Eine Aufnahme
       NACH diesem Datum KANN nicht auf diesem Medium entstanden sein.
       Bsp: shellac + decade=2005 → UNMÖGLICH (Shellac-Produktion endete ~1958).

    2. Erfindungs-Floor-Regel: Ein Medium kann keine Aufnahme enthalten,
       die VOR seiner Erfindung entstanden ist — ABER NUR wenn das Medium
       das ORIGINAL-Aufnahmemedium ist, nicht wenn es ein späteres
       Digitalisierungs-Format ist.
       Bsp: vinyl + decade=1920 → möglich (wenn Vinyl NACH 1948 erkannt wurde,
       ist die Aufnahme älter und wurde später auf Vinyl gepresst).
       ABER: mp3_high + decade=1920 → möglich (alte Aufnahme, neue Digitalisierung).

    Args:
        material: Das erkannte Material (z.B. "shellac", "vinyl", "mp3_high").
        decade: Die vom EraClassifier geschätzte Dekade (z.B. 1970).
        transfer_chain: Optionale Tonträgerkette für Kontext.

    Returns:
        True wenn konsistent, False wenn physikalisch unmöglich.
    """
    # ── Produktions-Ende (Hard Ceilings) ───────────────────────────────
    # Diese Medien wurden nach diesem Jahr NICHT MEHR als ORIGINAL-
    # Aufnahmemedium verwendet. Eine Aufnahme DANACH kann nicht auf
    # diesem Medium entstanden sein.
    _PRODUCTION_END: dict[str, int] = {
        "wax_cylinder": 1929,  # Edison stellte Wachswalzen 1929 ein
        "wire_recording": 1945,  # Drahtton nach WWII obsolet
        "shellac": 1958,  # Letzte kommerzielle Shellac-Pressungen
        "lacquer_disc": 1960,  # Transcription discs
        "8track": 1982,  # 8-Track Ende der Produktion
        "dat": 2005,  # DAT-Produktion eingestellt
        "minidisc": 2013,  # Letzte MiniDisc-Player
        "dcc": 1996,  # DCC eingestellt
    }

    _mat_lower = material.lower().replace(" ", "_").replace("-", "_")
    _end = _PRODUCTION_END.get(_mat_lower)
    if _end is not None and decade > _end:
        # Das Medium wurde NACH dem geschätzten Aufnahmejahr noch produziert?
        # Nein — wenn decade > _end, wurde es VOR der Aufnahme eingestellt.
        # Das ist physikalisch unmöglich.
        logger.warning(
            "validate_material_era_consistency: %s production ended %d, but era=%d → IMPOSSIBLE",
            material,
            _end,
            decade,
        )
        return False

    # ── Erfindungs-Floor (Hard Floor) ───────────────────────────────────
    # Digitale Verteilformate (MP3, CD, Streaming) sind Container —
    # sie können Aufnahmen aus JEDER früheren Ära enthalten.
    # ABER: Wenn decade VOR der Erfindung des Formats liegt UND das
    # Format das EINZIGE erkannte Medium ist, ist die Kombination
    # extrem unwahrscheinlich (das Format existierte noch nicht).
    _INVENTION_FLOOR: dict[str, int] = {
        "vinyl": 1948,
        "cassette": 1963,
        "reel_tape": 1935,
        "cd": 1982,
        "cd_digital": 1982,
        "dat": 1987,
        "minidisc": 1992,
        "mp3_low": 1995,
        "mp3_high": 1995,
        "mp3_high_vbr": 1998,
        "aac": 1997,
        "streaming": 2005,
        "dcc": 1992,
        "bluray_audio": 2006,
        "sacd": 1999,
        "pcm_digital": 1982,
    }

    _invented = _INVENTION_FLOOR.get(_mat_lower)
    if _invented is not None and decade < _invented:
        # Die Aufnahme wurde VOR der Erfindung des Mediums gemacht.
        # Wenn das Medium ein VERTEIL-FORMAT ist (CD, MP3, Streaming),
        # kann es eine ältere Aufnahme enthalten → akzeptieren.
        # Wenn das Medium ein ORIGINAL-AUFNAHMEMEDIUM ist (Shellac, Vinyl,
        # Cassette) und das EINZIGE erkannte Medium → IMPOSSIBLE.
        _distribution_formats = {
            "cd",
            "cd_digital",
            "mp3_low",
            "mp3_high",
            "mp3_high_vbr",
            "aac",
            "streaming",
            "bluray_audio",
            "sacd",
            "pcm_digital",
            "dat",
            "minidisc",
            "dcc",
        }
        if _mat_lower not in _distribution_formats:
            # Analoges Original-Medium — kann nicht vor seiner Erfindung
            # als ORIGINAL-Aufnahmemedium verwendet worden sein.
            # ABER: Wenn die Kette mehrere Medien enthält, könnte das
            # physische Medium ein späteres Remaster sein.
            # Bei Einzel-Medium (keine Chain) → IMPOSSIBLE.
            if not transfer_chain or len(transfer_chain) <= 1:
                logger.warning(
                    "validate_material_era_consistency: %s invented %d, but era=%d and no transfer chain → IMPOSSIBLE",
                    material,
                    _invented,
                    decade,
                )
                return False

    return True


def build_chain(materials: list[str], era_decade: int | None = None) -> list[str]:
    """Baut die Tonträgerkette AUSSCHLIESSLICH aus erkannten Medien.

    GRUNDSATZ: Kein Medium wird erfunden oder angenommen.
    Die Kette enthält NUR das, was die Detektoren tatsächlich erkannt haben.
    - Erkennt der MediumDetector vinyl → vinyl kommt in die Kette.
    - Erkennt der DefectScanner cassette → cassette kommt dazu.
    - NICHTS wird implizit ergänzt. Kein "reel_tape" wenn keines erkannt wurde.

    Args:
        materials: Liste der TATSÄCHLICH erkannten Materialien
        era_decade: Geschätzte Aufnahme-Dekade (nicht verwendet)

    Returns:
        Chronologisch sortierte, deduplizierte Kette der ERKANNTEN Medien.
    """
    _era_order = [
        "wax_cylinder",
        "shellac",
        "reel_tape",
        "lacquer_disc",
        "vinyl",
        "cassette",
        "dat",
        "cd",
        "minidisc",
        "mp3",
        "mp3_low",
        "mp3_high",
        "streaming",
    ]

    # Nur deduplizieren + chronologisch sortieren. NICHTS hinzufügen.
    seen: set[str] = set()
    chain: list[str] = []
    for m in materials:
        if m and m != "unknown" and m not in seen:
            seen.add(m)
            chain.append(m)

    chain.sort(key=lambda m: _era_order.index(m) if m in _era_order else 99)
    return chain
