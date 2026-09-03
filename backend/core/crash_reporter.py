"""backend/core/crash_reporter.py — §v10.700 I4: Opt-in Crash-Reporting.

from typing import Any
Fängt unbehandelte Exceptions, sammelt Stack-Trace + Aurik-Version + OS + RAM,
und speichert Crash-Reports lokal. KEIN automatischer Upload — DSGVO-konform.

Nutzung:
    from backend.core.crash_reporter import install_crash_handler
    install_crash_handler()  # Einmal beim App-Start

Crash-Reports werden gespeichert unter: ~/.aurik/crash_reports/
Export via CLI: aurik report --export → verschlüsseltes ZIP
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path.home() / ".aurik" / "crash_reports"
_MAX_REPORTS = 50  # Maximale Anzahl gespeicherter Reports


def _get_system_info() -> dict[str, Any]:  # type: ignore[name-defined]
    """Sammelt System-Informationen für Crash-Report."""
    info: dict[str, Any] = {  # type: ignore[name-defined]
        "platform": sys.platform,
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Aurik-Version
    try:
        from backend.core.version import __version__

        info["aurik_version"] = __version__
    except ImportError:
        info["aurik_version"] = "unknown"

    # RAM
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
        info["ram_percent"] = mem.percent
    except ImportError:
        info["ram_total_gb"] = -1

    # GPU
    try:
        from backend.core.ml_device_manager import get_ml_device_manager

        mgr = get_ml_device_manager()
        info["gpu_backend"] = str(getattr(mgr, "_backend", "unknown"))
        info["gpu_name"] = str(getattr(mgr, "_gpu_name", "unknown"))
    except ImportError:
        info["gpu_backend"] = "unknown"

    return info


def _save_report(exc_type: type, exc_value: BaseException, exc_tb: Any) -> str | None:  # type: ignore[name-defined]
    """Speichert einen Crash-Report als JSON."""
    try:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Alte Reports aufräumen
        existing = sorted(_REPORTS_DIR.glob("crash_*.json"))
        for old in existing[: -_MAX_REPORTS + 1]:
            try:
                old.unlink()
            except OSError:
                logger.debug("Crash-Reporter: Alte Crash-Datei konnte nicht geloescht werden", exc_info=True)

        # Report zusammenbauen
        report: dict[str, Any] = {  # type: ignore[name-defined]
            "system": _get_system_info(),
            "exception": {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb),
            },
        }

        # Speichern
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = _REPORTS_DIR / f"crash_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.critical("Crash-Report gespeichert: %s", report_path)
        return str(report_path)
    except Exception:
        logger.critical("Konnte Crash-Report nicht speichern", exc_info=True)
        return None


def _crash_handler(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:  # type: ignore[name-defined]
    """Globaler Exception-Hook — wird bei unbehandelten Exceptions aufgerufen."""
    # KeyboardInterrupt nicht als Crash behandeln
    if exc_type is KeyboardInterrupt:
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    report_path = _save_report(exc_type, exc_value, exc_tb)

    # Standard-Handler aufrufen (print traceback)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

    # Hinweis für Nutzer
    if report_path:
        logger.critical("Aurik ist abgestürzt. Crash-Report gespeichert unter: %s", report_path)
        logger.critical("Export mit: aurik report --export")


def install_crash_handler() -> None:
    """Installiert den globalen Exception-Hook. Einmal beim App-Start aufrufen."""
    sys.excepthook = _crash_handler
    logger.info("Crash-Reporter installiert — Reports unter %s", _REPORTS_DIR)


# §v10.993: GUI-Sichtbarkeit — "Neue Fehler seit letzter Sitzung"
_LAST_SEEN_FILE = _REPORTS_DIR / ".last_seen"


def get_last_seen_ts() -> float:
    """Zeitstempel der letzten Sitzung, in der Reports als gesehen markiert wurden."""
    try:
        return float(_LAST_SEEN_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        logger.warning("§V6 ML→DSP-Fallback: get_last_seen_ts fehlgeschlagen → neutraler Return (0.0)")
        return 0.0


def mark_reports_seen() -> None:
    """Markiert alle aktuellen Reports als gesehen (Basislinie für die nächste Sitzung)."""
    try:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_SEEN_FILE.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        logger.debug("Crash-Reporter: .last_seen konnte nicht geschrieben werden", exc_info=True)


def get_new_reports() -> list[dict[str, Any]]:  # type: ignore[name-defined]
    """Reports, die seit der letzten als gesehen markierten Sitzung entstanden sind.

    Returns:
        [{_file, type, message, timestamp}, …] — leichte Zusammenfassung fürs Frontend.
    """
    since = get_last_seen_ts()
    reports: list[dict[str, Any]] = []  # type: ignore[name-defined]
    if not _REPORTS_DIR.exists():
        return reports
    for report_file in sorted(_REPORTS_DIR.glob("crash_*.json")):
        try:
            if report_file.stat().st_mtime <= since:
                continue
            with open(report_file, encoding="utf-8") as f:
                data = json.load(f)
            exc = data.get("exception", {}) or {}
            reports.append(
                {
                    "_file": str(report_file),
                    "type": str(exc.get("type", "unknown")),
                    "message": str(exc.get("message", ""))[:200],
                    "timestamp": report_file.stat().st_mtime,
                }
            )
        except Exception:
            logger.debug("Konnte Report nicht lesen: %s", report_file)
    return reports


def get_reports() -> list[dict[str, Any]]:  # type: ignore[name-defined]
    """Listet alle gespeicherten Crash-Reports auf."""
    reports = []
    if _REPORTS_DIR.exists():
        for report_file in sorted(_REPORTS_DIR.glob("crash_*.json"), reverse=True):
            try:
                with open(report_file, encoding="utf-8") as f:
                    data = json.load(f)
                data["_file"] = str(report_file)
                reports.append(data)
            except Exception:
                logger.debug("Konnte Report nicht lesen: %s", report_file)
    return reports


def export_reports(output_path: str | Path | None = None) -> str | None:
    """Exportiert alle Crash-Reports als ZIP-Datei (für Support)."""
    import zipfile

    reports = get_reports()
    if not reports:
        logger.info("Keine Crash-Reports vorhanden")
        return None

    output_path = Path(output_path or Path.home() / "aurik_crash_reports.zip")
    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for report in reports:
                report_file = report.get("_file", "")
                if report_file and Path(report_file).exists():
                    zf.write(report_file, Path(report_file).name)
        logger.info("Crash-Reports exportiert: %s (%d Reports)", output_path, len(reports))
        return str(output_path)
    except Exception:
        logger.error("Ausgabe fehlgeschlagen", exc_info=True)
        return None
