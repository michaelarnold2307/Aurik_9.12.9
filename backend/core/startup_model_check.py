"""Startup-Modell-Integritätsprüfung (§2 Spec 08 — Out-of-the-Box-Pflicht).

Prüft beim App-Start, ob alle lokal gebündelten ML-Modelle vorhanden sind.
Erzeugt einen deutschsprachigen Bericht für den Laienanwender.

Entwurfs-Prinzipien:
    - Kein ML-Import — reine Python-stdlib (json, pathlib, logging)
    - Nicht-blockierend: DSP-Fallbacks existieren für jedes Modell
    - Drei Stufen: OK / DEGRADED (fehlende SOTA-Modelle) / MINIMAL (kein ML)
    - Alle Nutzer-Meldungen auf Deutsch

Referenz: §13 Aurik-Spec — Out-of-the-Box, §8.1 PluginLifecycleManager
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

_MANIFEST_PATH: Path = Path(__file__).parent.parent.parent / "models" / "manifest.json"

# Primär-Modelle: direkt im Kern-Pipeline-Pfad, ihr Fehlen > deutlicher Qualitätsverlust
_PRIMARY_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "deepfilternet_v3",
        "deepfilternet_dec",
        "deepfilternet_erb_dec",
        "rmvpe",
        "silero_vad",
        "beats_iter3",
        "panns",
        "versa_mos",
        "sgmse_plus",
        "htdemucs_ft",
        "apollo",
        "resemble_enhance",
        "vocos_mel_44khz",
    }
)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class ModelCheckResult:
    """Ergebnis der Startup-Modell-Integritätsprüfung.

    Attributes:
        all_ok:              True wenn alle bundled-Modelle vorhanden sind.
        mode:                "FULL_ML" | "DEGRADED_ML" | "DSP_ONLY"
        missing_primary:     Primär-Modelle mit fehlender Datei (spürbare Qualitätseinbußen).
        missing_optional:    Optionale Modelle mit fehlender Datei (kein Qualitätsverlust).
        found_count:         Anzahl vorhandener Modell-Dateien.
        total_bundled:       Gesamtzahl der im Manifest als bundled markierten Einträge.
        user_message_de:     Deutschsprachige Meldung für den Nutzer (leer = alles OK).
        user_title_de:       Titel für den deutschen Dialog-Text.
        is_critical:         True wenn im DSP_ONLY-Modus (kein primäres ML-Modell).
    """

    all_ok: bool = True
    mode: str = "FULL_ML"
    missing_primary: list[dict] = field(default_factory=list)
    missing_optional: list[dict] = field(default_factory=list)
    found_count: int = 0
    total_bundled: int = 0
    user_message_de: str = ""
    user_title_de: str = ""
    is_critical: bool = False


# ---------------------------------------------------------------------------
# Kern-Prüfung
# ---------------------------------------------------------------------------


def check_models(app_root: Path | None = None) -> ModelCheckResult:
    """Prüft alle bundled-Modelle aus models/manifest.json auf Existenz.

    Kategorisiert fehlende Modelle nach Auswirkung auf die Qualität:
        - missing_primary:  Spürbare Qualitätseinbußen im Vergleich zu Vollbetrieb.
        - missing_optional: Kein merklicher Unterschied (DSP-Fallbacks gleichwertig).

    Args:
        app_root: Basis-Pfad der Aurik-Installation. Standard: automatisch ermittelt.

    Returns:
        ModelCheckResult mit deutschsprachiger Nutzer-Meldung.
    """
    # §Spec 24 Root-Fix: librosa VOR jedem Worker-Thread vollständig auflösen.
    # check_models() läuft im Hauptthread, bevor der bridge-Warmup-Daemon startet
    # (Befund 2026-08-16: parallele Erst-Importe korrumpierten numba/lazy_loader).
    # Idempotent — kostet bei späteren Aufrufen nichts.
    from backend.core.librosa_bootstrap import ensure_librosa_ready

    ensure_librosa_ready()

    root = app_root or _MANIFEST_PATH.parent.parent

    # Manifest laden
    manifest_path = root / "models" / "manifest.json"
    if not manifest_path.exists():
        logger.warning("Start_Pruefung: manifest.json nicht gefunden in %s", manifest_path)
        return ModelCheckResult(
            all_ok=False,
            mode="DSP_ONLY",
            is_critical=True,
            user_title_de="⚠ Modell-Konfiguration nicht gefunden",
            user_message_de=(
                "Die Modell-Konfigurationsdatei (models/manifest.json) wurde nicht gefunden.\n\n"
                "Aurik startet mit eingeschränkter Klangqualität — KI-basierte Verbesserungen sind deaktiviert.\n\n"
                "Lösung: Stellen Sie sicher, dass der Aurik-Installationsordner vollständig "
                "vorhanden ist und models/manifest.json existiert."
            ),
        )

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:
        logger.error("Start_Pruefung: manifest.json konnte nicht gelesen werden: %s", exc)
        return ModelCheckResult(
            all_ok=False,
            mode="DSP_ONLY",
            is_critical=True,
            user_title_de="⚠ Modell-Konfiguration fehlerhaft",
            user_message_de=(
                f"Die Modell-Konfigurationsdatei konnte nicht geladen werden:\n{exc}\n\n"
                "Aurik startet mit eingeschränkter Klangqualität. Bitte prüfen Sie die Installation."
            ),
        )

    models = manifest.get("models", [])
    bundled_models = [m for m in models if m.get("bundled", False)]

    missing_primary: list[dict] = []
    missing_optional: list[dict] = []
    found_count = 0

    for entry in bundled_models:
        name = entry.get("name", "?")
        bundled_path = entry.get("bundled_path", "")
        if not bundled_path:
            continue

        file_path = root / bundled_path
        if file_path.exists():
            found_count += 1
            logger.debug("Start_Pruefung: OK — %s (%s)", name, bundled_path)
        else:
            logger.warning("Start_Pruefung: FEHLT — %s (%s)", name, bundled_path)
            info = {
                "name": name,
                "path": bundled_path,
                "size_gb": entry.get("size_gb", 0.0),
                "description": entry.get("description", ""),
                "fallback": entry.get(
                    "fallback", "DSP"
                ),  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
            if name in _PRIMARY_MODEL_NAMES:
                missing_primary.append(info)
            else:
                missing_optional.append(info)

    total_bundled = len(bundled_models)

    # Modus bestimmen
    if not missing_primary and not missing_optional:
        mode = "FULL_ML"
        all_ok = True
    elif not missing_primary:
        mode = "DEGRADED_ML"  # Optionale Modelle fehlen — kein Qualitätsverlust
        all_ok = False
    elif found_count == 0:
        mode = "DSP_ONLY"
        all_ok = False
    else:
        mode = "DEGRADED_ML"
        all_ok = False

    is_critical = mode == "DSP_ONLY"

    # Deutschsprachige Nutzer-Meldung
    user_title_de = ""
    user_message_de = ""

    if missing_primary or missing_optional:
        n_total_missing = len(missing_primary) + len(missing_optional)

        if mode == "DSP_ONLY":
            user_title_de = "⚠ KI-Modelle fehlen — Eingeschränkter Betrieb"
            user_message_de = (
                "Es wurden keine KI-Modell-Dateien gefunden "
                f"({n_total_missing} von {total_bundled} fehlen).\n\n"
                "Die Restaurierungsqualität ist deutlich eingeschränkt — "
                "KI-gestützte Methoden sind nicht verfügbar.\n\n"
                "Lösung: Stellen Sie sicher, dass der Ordner 'models/' vollständig "
                "im Programmverzeichnis vorhanden ist.\n\n"
                "Aurik kann trotzdem gestartet werden — die Verarbeitung erfolgt "
                "mit klassischen Verfahren."
            )
        elif missing_primary:
            names = ", ".join(m["name"] for m in missing_primary[:4])
            if len(missing_primary) > 4:
                names += f" und {len(missing_primary) - 4} weitere"
            user_title_de = "ℹ Einige KI-Modelle nicht gefunden"
            user_message_de = (
                f"{len(missing_primary)} KI-Modell(e) fehlen: {names}.\n\n"
                "Aurik startet mit leicht eingeschränkter Qualität. Für die betroffenen "
                "Verarbeitungsschritte werden klassische Verfahren verwendet.\n\n"
                "Die Grundfunktionalität ist vollständig erhalten."
            )
        elif missing_optional:
            # Nur optionale fehlen → kein Dialog nötig, nur Log
            logger.info(
                "Start_Pruefung: %d optionale Modelle nicht gefunden — DSP-Fallbacks aktiv.",
                len(missing_optional),
            )

    result = ModelCheckResult(
        all_ok=all_ok,
        mode=mode,
        missing_primary=missing_primary,
        missing_optional=missing_optional,
        found_count=found_count,
        total_bundled=total_bundled,
        user_message_de=user_message_de,
        user_title_de=user_title_de,
        is_critical=is_critical,
    )

    _log_summary(result)
    # §v10.950: Model Zoo Registry — alle Modelle sichtbar machen
    try:
        from backend.core.model_zoo_registry import report as _zoo_report

        logger.info("Start_Pruefung: %s", _zoo_report())
    except Exception:
        pass

    return result


def _log_summary(result: ModelCheckResult) -> None:
    """Loggt eine kompakte englische Zusammenfassung des Check-Ergebnisses."""
    if result.all_ok:
        logger.info(
            "Start_Pruefung: ALL OK — %d/%d bundled models present (FULL_ML Betriebsart)",
            result.found_count,
            result.total_bundled,
        )
    else:
        logger.warning(
            "Start_Pruefung: Betriebsart=%s found=%d/%d missing_primary=%d missing_optional=%d",
            result.mode,
            result.found_count,
            result.total_bundled,
            len(result.missing_primary),
            len(result.missing_optional),
        )
        if result.missing_primary:
            for m in result.missing_primary:
                logger.warning(
                    "  MISSING PRIMARY: %s → %s (Ersatzpfad: %s)",
                    m["name"],
                    m["path"],
                    m.get("fallback", "DSP"),  # §V6 (copilot-instructions.md): logger.warning handled at call site
                )


# ---------------------------------------------------------------------------
# Singleton-Cache (wird beim ersten Aufruf befüllt)
# ---------------------------------------------------------------------------

_cached_result: ModelCheckResult | None = None
_check_lock = threading.Lock()


def get_startup_check_result(app_root: Path | None = None) -> ModelCheckResult:
    """Gibt gecachtes Ergebnis zurück (Thread-sicher, wird nur einmal ausgeführt).

    Args:
        app_root: Wird nur beim ersten Aufruf berücksichtigt.

    Returns:
        ModelCheckResult (gecacht nach erstem Aufruf).
    """
    global _cached_result
    if _cached_result is None:
        with _check_lock:
            if _cached_result is None:
                _cached_result = check_models(app_root)
    return _cached_result
