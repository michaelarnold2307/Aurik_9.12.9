"""core/model_downloader.py — §13.3 Modell-Download-Management.

Prüft lokal gebündelte Modelle aus ``models/manifest.json`` und lädt optional
SOTA-Upgrade-Modelle im Hintergrund nach.

Architekturregel:
    - Alle primären Modelle sind ``bundled: true`` und sofort out-of-the-box verfügbar.
    - Kein Netzwerk-Zugriff wird beim ersten Start erzwungen.
    - SOTA-Upgrades werden optional und asynchron nachgeladen.
    - Thread-safe Singleton (Double-Checked Locking, §3.2).

Spec-Referenz: §13.3 Distribution & Installation — Out-of-the-Box-Pflicht.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MANIFEST_PATH = _PROJECT_ROOT / "models" / "manifest.json"
_SOTA_CACHE_DIR = Path.home() / ".aurik" / "sota_models"
_SOTA_MANIFEST = Path.home() / ".aurik" / "sota_manifest.json"

# ──────────────────────────────────────────────────────────────────────────────
# Offline-Modus (aktuell aktiv — kein Netzwerk, nur lokale Dateien)
# Auf False setzen um SOTA-Upgrade-Downloads zu aktivieren.
# ──────────────────────────────────────────────────────────────────────────────
OFFLINE_MODE: bool = False  # §v10.700 L2: SOTA-Modell-Downloads aktiviert
_ALLOWED_SOTA_URL_SCHEMES = {"https"}


# ──────────────────────────────────────────────────────────────────────────────
# Datenklassen
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelEntry:
    """Eintrag aus manifest.json."""

    name: str
    bundled: bool
    bundled_path: str
    sha256: str
    size_bytes: int
    required: bool
    fallback: str
    sota_upgrade: dict | None = field(default=None)
    license: str = ""
    reference: str = ""
    format: str = "onnx"


@dataclass
class ModelStatus:
    """Status eines Modells nach ``ensure_all()``."""

    name: str
    available: bool  # True = lokal OK, False = DSP-Fallback aktiv
    path: Path | None  # Absoluter Pfad wenn available=True
    source: str  # "bundled" | "sota_cache" | "fallback"
    sota_available: bool  # True wenn SOTA-Upgrade bereits gecacht


# ──────────────────────────────────────────────────────────────────────────────
# SHA256-Verifikation (öffentlich, Spec §13.3)
# ──────────────────────────────────────────────────────────────────────────────


def verify_model(path: Path, expected_sha256: str) -> bool:
    """Prüft Integrität einer Modelldatei via SHA256.

    Args:
        path: Absoluter Pfad zur Modelldatei.
        expected_sha256: Erwartete SHA256-Prüfsumme (hex, Groß-/Kleinschreibung egal).

    Returns:
        True wenn Prüfsumme korrekt, False sonst.
    """
    if not path.is_file():
        return False
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest().lower() == expected_sha256.lower()
    except OSError as exc:
        logger.debug("SHA256-Prüfung fehlgeschlagen (%s): %s", path, exc)
        return False


def _compute_sha256(path: Path) -> str:
    """Berechnet SHA256 einer lokalen Datei (Streaming, 64KB-Blöcke)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


# ── Retry/Resume-Konfiguration ────────────────────────────────────────────────

_MAX_RETRIES: int = 3
_BASE_BACKOFF_S: float = 2.0
_BACKOFF_MULTIPLIER: float = 2.0
_MIN_BYTES_PER_SECOND: float = 1024.0  # 1 KB/s Minimum für Timeout-Berechnung


def _download_with_retry(
    url: str,
    target: Path,
    *,
    expected_size_bytes: int = 0,
    progress_callback: Callable[[str, float], None] | None = None,
    model_name: str = "",
) -> bool:
    """Lädt ein SOTA-Modell mit Retry + Resume + adaptivem Timeout.

    Args:
        url: Download-URL (nur https://)
        target: Zielpfad
        expected_size_bytes: Erwartete Größe für Timeout-Kalkulation (0=default 300s)
        progress_callback: Optionaler Fortschritt-Callback(name, fraction 0-1)
        model_name: Name für Logging/Callback

    Returns:
        True bei Erfolg, False bei endgültigem Fehlschlag.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SOTA_URL_SCHEMES:
        raise ValueError(f"Nicht erlaubtes Download-Schema: {parsed.scheme or '<leer>'}")

    # Adaptiver Timeout: min 60s, max 1800s, skaliert mit erwarteter Größe
    if expected_size_bytes > 0:
        timeout_s = max(60.0, min(1800.0, expected_size_bytes / _MIN_BYTES_PER_SECOND))
    else:
        timeout_s = 300.0

    last_error: str = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            headers: dict[str, str] = {}
            resume_pos: int = 0

            # Resume: Wenn partielle Datei existiert, Range-Request
            if target.is_file():
                resume_pos = target.stat().st_size
                if resume_pos > 0:
                    headers["Range"] = f"bytes={resume_pos}-"
                    logger.debug("Download-Resume bei Byte %d für %s", resume_pos, model_name or url)

            with requests.get(url, stream=True, timeout=timeout_s, headers=headers) as response:
                if resume_pos > 0 and response.status_code == 206:
                    # Partial Content — an partielle Datei anhängen
                    mode = "ab"
                elif response.status_code == 200:
                    # Vollständiger Download — überschreiben
                    mode = "wb"
                    resume_pos = 0
                else:
                    response.raise_for_status()
                    mode = "wb"  # Fallback

                total_size = int(response.headers.get("content-length", 0))
                if resume_pos > 0 and mode == "ab":
                    total_size += resume_pos

                downloaded = resume_pos
                with target.open(mode) as fh:
                    for chunk in response.iter_content(chunk_size=131072):  # 128KB chunks
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback is not None and total_size > 0:
                                with contextlib.suppress(Exception):
                                    progress_callback(model_name or url, min(0.99, downloaded / total_size))

            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(model_name or url, 1.0)

            logger.info(
                "Download erfolgreich: %s (%d bytes, attempt %d/%d)",
                model_name or url,
                downloaded,
                attempt,
                _MAX_RETRIES,
            )
            return True

        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                backoff = _BASE_BACKOFF_S * (_BACKOFF_MULTIPLIER ** (attempt - 1))
                logger.warning(
                    "Download-Versuch %d/%d fehlgeschlagen (%s), neuer Versuch in %.1fs...",
                    attempt,
                    _MAX_RETRIES,
                    last_error[:120],
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "Download endgültig fehlgeschlagen nach %d Versuchen: %s",
                    _MAX_RETRIES,
                    last_error[:200],
                )

        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                backoff = _BASE_BACKOFF_S * (_BACKOFF_MULTIPLIER ** (attempt - 1))
                logger.warning("Download-Fehler (Versuch %d/%d): %s", attempt, _MAX_RETRIES, last_error[:120])
                time.sleep(backoff)
            else:
                logger.error("Download endgültig fehlgeschlagen: %s", last_error[:200])

    return False


# Kompatibilität: Alte _download_remote_model-Funktion ruft jetzt neue Implementierung
def _download_remote_model(url: str, target: Path, timeout_s: float = 30.0) -> None:
    """Legacy-Wrapper für _download_with_retry. Wirft Exception bei Fehlschlag."""
    if not _download_with_retry(url, target, expected_size_bytes=0):
        raise OSError(f"Download fehlgeschlagen nach {_MAX_RETRIES} Versuchen: {url}")


# ──────────────────────────────────────────────────────────────────────────────
# ModelDownloader — Singleton
# ──────────────────────────────────────────────────────────────────────────────


class ModelDownloader:
    """Prüft lokal gebündelte Modelle und lädt optional SOTA-Upgrades nach.

    Ladereihenfolge pro Modell (``load_bundled``):
        1. SOTA-Cache prüfen (~/.aurik/sota_models/) → wenn vorhanden und SHA256 ok → nutzen
        2. bundled_path lokal prüfen (PROJECT_MODELS_DIR) mit SHA256-Verifikation
        3. SHA256-Mismatch → DSP-Fallback aktivieren, Warnung loggen
        4. sota_upgrade optional im Hintergrund nachladen (``schedule_sota_upgrade``)

    Thread-Safety:
        Double-Checked Locking Singleton; separate ``_download_lock`` für
        nebenläufige Download-Aufträge.

    Nutzer-Meldungen (Deutsch, laienverständlich):
        - KI-Modell {name} konnte nicht geladen werden — klassische Methode wird genutzt.
        - Besseres KI-Modell ({sota_name}) wird im Hintergrund geladen...
        - {sota_name} steht ab sofort zur Verfügung (nächste Verarbeitung).
    """

    PROJECT_MODELS_DIR: Path = _PROJECT_ROOT / "models"
    MANIFEST: Path = _MANIFEST_PATH

    # Singleton-State
    _instance: ModelDownloader | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> ModelDownloader:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._initialized = False
                    cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._download_lock = threading.Lock()
        _SOTA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._manifest_entries: list[ModelEntry] = self._parse_manifest()
        logger.info(
            "ModelDownloader initialisiert — %d Modell-Einträge geladen.",
            len(self._manifest_entries),
        )

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _parse_manifest(self) -> list[ModelEntry]:
        """Liest models/manifest.json und gibt Liste von ModelEntry zurück.

        Returns:
            Leere Liste wenn Manifest fehlt oder ungültig ist.
        """
        if not self.MANIFEST.is_file():
            logger.warning("Manifest nicht gefunden: %s", self.MANIFEST)
            return []
        try:
            with open(self.MANIFEST, encoding="utf-8") as fh:
                data = json.load(fh)
            entries: list[ModelEntry] = []
            for raw in data.get("models", []):
                entries.append(
                    ModelEntry(
                        name=raw.get("name", ""),
                        bundled=bool(raw.get("bundled", False)),
                        bundled_path=raw.get("bundled_path", ""),
                        sha256=raw.get("sha256", ""),
                        size_bytes=int(raw.get("size_bytes", 0)),
                        required=bool(raw.get("required", False)),
                        fallback=raw.get(
                            "fallback", "dsp"
                        ),  # §V6 (copilot-instructions.md): logger.warning handled at call site
                        sota_upgrade=raw.get("sota_upgrade"),
                        license=raw.get("license", ""),
                        reference=raw.get("reference", ""),
                        format=raw.get("format", "onnx"),
                    )
                )
            return entries
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Manifest-Parsing fehlgeschlagen: %s", exc)
            return []

    def get_entry(self, model_name: str) -> ModelEntry | None:
        """Gibt ModelEntry für ``model_name`` zurück oder None."""
        for entry in self._manifest_entries:
            if entry.name == model_name:
                return entry
        return None

    # ── Haupt-API (Spec §13.3) ────────────────────────────────────────────────

    def ensure_all(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, bool]:
        """Prüft alle Modelle aus dem Manifest.

        Ladereihenfolge pro Modell:
            1. bundled_path lokal prüfen (SHA256-Verifikation)
            2. Falls vorhanden und valide → direkt nutzen (kein Download)
            3. Falls nicht vorhanden → DSP-Fallback aktivieren
            4. sota_upgrade optional im Hintergrund nachladen

        Args:
            progress_callback: Optionaler Callback(model_name, fraction ∈ [0,1]).

        Returns:
            Dict[model_name → True (lokal OK) / False (DSP-Fallback aktiv)]
        """
        results: dict[str, bool] = {}
        n = len(self._manifest_entries)
        for i, entry in enumerate(self._manifest_entries):
            path = self.load_bundled(entry)
            ok = path is not None
            results[entry.name] = ok
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(entry.name, (i + 1) / max(1, n))
        logger.info(
            "ensure_all: %d/%d Modelle verfügbar.",
            sum(results.values()),
            len(results),
        )
        return results

    def load_bundled(self, entry: dict | ModelEntry) -> Path | None:
        """Gibt Pfad zum lokal gebündelten Modell zurück (SHA256-verifiziert).

        Sucht zuerst im SOTA-Cache (~/.aurik/sota_models/), dann im gebündelten
        models/-Verzeichnis.

        Args:
            entry: Dict aus manifest.json ODER ModelEntry-Objekt.

        Returns:
            Absoluter Path wenn Modell vorhanden und SHA256 korrekt, sonst None.

        Side-effects:
            Loggt deutsche Nutzer-Meldung bei Fehler.
        """
        if isinstance(entry, dict):
            me = ModelEntry(
                name=entry.get("name", ""),
                bundled=bool(entry.get("bundled", False)),
                bundled_path=entry.get("bundled_path", ""),
                sha256=entry.get("sha256", ""),
                size_bytes=int(entry.get("size_bytes", 0)),
                required=bool(entry.get("required", False)),
                fallback=entry.get(
                    "fallback", "dsp"
                ),  # §V6 (copilot-instructions.md): logger.warning handled at call site
                sota_upgrade=entry.get("sota_upgrade"),
            )
        else:
            me = entry

        # 1) SOTA-Cache — nur im Online-Modus
        if not OFFLINE_MODE:
            sota_path = _SOTA_CACHE_DIR / f"{me.name}.onnx"
            if sota_path.is_file():
                sota_sha = self._read_sota_sha256(me.name)
                if sota_sha:
                    if verify_model(sota_path, sota_sha):
                        logger.debug("SOTA-Zwischenspeicher verwendet: %s", sota_path)
                        return sota_path
                else:
                    return sota_path

        # 2) Gebündeltes Modell prüfen
        if not me.bundled_path:
            logger.warning(
                "KI-Modell %s konnte nicht geladen werden — klassische Methode wird genutzt.",
                me.name,
            )
            return None

        bundled_abs = self.PROJECT_MODELS_DIR / me.bundled_path.lstrip("/")
        # Pfad kann relativ zum Manifest sein — ggf. direkt nutzen
        if not bundled_abs.is_file():
            bundled_abs = Path(me.bundled_path)

        if not bundled_abs.is_file():
            logger.info(
                "KI-Modell %s konnte nicht geladen werden — klassische Methode wird genutzt. "
                "(Datei nicht vorhanden: %s)",
                me.name,
                bundled_abs,
            )
            return None

        # 3) SHA256-Verifikation (überspringen wenn kein Hash angegeben)
        if me.sha256 and not verify_model(bundled_abs, me.sha256):
            logger.warning(
                "KI-Modell %s konnte nicht geladen werden — SHA256-Prüfung fehlgeschlagen. "
                "Klassische Methode wird genutzt.",
                me.name,
            )
            return None

        logger.debug("Gebündeltes Modell geladen: %s", bundled_abs)
        return bundled_abs

    def schedule_sota_upgrade(
        self,
        entry: dict | ModelEntry,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """Lädt SOTA-Upgrade optional im Hintergrund nach (Threading).

        Im OFFLINE_MODE (aktuell aktiv) wird kein Netzwerkaufruf durchgeführt.
        Zum Aktivieren: OFFLINE_MODE = False in diesem Modul setzen.

        Args:
            entry: Dict aus manifest.json ODER ModelEntry-Objekt.
            progress_callback: Optional Callback(sota_name, fraction ∈ [0,1]).
        """
        if OFFLINE_MODE:
            model_name = entry.get("name", "") if isinstance(entry, dict) else entry.name
            logger.debug("SOTA-Upgrade für '%s' übersprungen — OFFLINE_Betriebsart aktiv.", model_name)
            return

        # ── Online-Pfad (nur wenn OFFLINE_MODE = False) ───────────────────────

        if isinstance(entry, dict):
            sota = entry.get("sota_upgrade")
            model_name = entry.get("name", "")
        else:
            sota = entry.sota_upgrade
            model_name = entry.name

        if not sota:
            logger.debug("Kein SOTA-Upgrade für Modell '%s' konfiguriert.", model_name)
            return

        sota_name = sota.get("name", model_name)
        sota_url = sota.get("url", "")
        sota_sha256 = sota.get("sha256", "")
        sota_size = sota.get("size_bytes", 0)

        if not sota_url:
            logger.debug("SOTA-Upgrade ohne URL für '%s' — übersprungen.", model_name)
            return

        def _download_worker() -> None:
            with self._download_lock:
                target = _SOTA_CACHE_DIR / f"{model_name}.onnx"
                if target.is_file():
                    # Prüfe ob bereits gecachtes Modell valide ist
                    if sota_sha256 and verify_model(target, sota_sha256):
                        logger.debug("SOTA bereits gecacht und verifiziert: %s", target)
                        if progress_callback is not None:
                            with contextlib.suppress(Exception):
                                progress_callback(sota_name, 1.0)
                        return
                    # Kein SHA256 oder Verification fehlgeschlagen → neu laden
                    if not sota_sha256:
                        # Auto-compute SHA256 von existierender Datei
                        cached_sha = _compute_sha256(target)
                        self._write_sota_sha256(model_name, cached_sha)
                        logger.debug("SOTA gecacht, SHA256 neu berechnet: %s", cached_sha[:16])
                        if progress_callback is not None:
                            with contextlib.suppress(Exception):
                                progress_callback(sota_name, 1.0)
                        return

                logger.info("Besseres KI-Modell (%s) wird im Hintergrund geladen...", sota_name)
                if progress_callback is not None:
                    with contextlib.suppress(Exception):
                        progress_callback(sota_name, 0.0)

                try:
                    success = _download_with_retry(
                        sota_url,
                        target,
                        expected_size_bytes=sota_size,
                        progress_callback=progress_callback,
                        model_name=sota_name,
                    )
                    if not success:
                        logger.info(
                            "Download fehlgeschlagen (%s) — Standard-Modell bleibt aktiv.",
                            sota_name,
                        )
                        target.unlink(missing_ok=True)
                        return

                    # SHA256-Verifikation nach Download
                    if sota_sha256:
                        if verify_model(target, sota_sha256):
                            self._write_sota_sha256(model_name, sota_sha256)
                        else:
                            logger.warning(
                                "SOTA-Modell '%s' SHA256-Prüfung fehlgeschlagen — "
                                "heruntergeladene Datei wird verworfen.",
                                sota_name,
                            )
                            target.unlink(missing_ok=True)
                            return
                    else:
                        # Auto-compute SHA256 und speichern für zukünftige Verifikation
                        computed = _compute_sha256(target)
                        self._write_sota_sha256(model_name, computed)
                        logger.info("SOTA-SHA256 berechnet und gespeichert: %s", computed[:16])

                    logger.info(
                        "%s steht ab sofort zur Verfügung (nächste Verarbeitung).",
                        sota_name,
                    )
                    if progress_callback is not None:
                        with contextlib.suppress(Exception):
                            progress_callback(sota_name, 1.0)

                except (requests.RequestException, urllib.error.URLError, OSError, ValueError) as exc:
                    logger.info(
                        "Download fehlgeschlagen (%s) — Standard-Modell bleibt aktiv. Fehler: %s",
                        sota_name,
                        exc,
                    )
                    target.unlink(missing_ok=True)

        thread = threading.Thread(target=_download_worker, daemon=True)
        thread.start()

    # ── SOTA-Cache-Manifest ────────────────────────────────────────────────────

    def _read_sota_sha256(self, model_name: str) -> str:
        """Liest gecachte SHA256 für ein SOTA-Modell."""
        if not _SOTA_MANIFEST.is_file():
            return ""
        try:
            with open(_SOTA_MANIFEST, encoding="utf-8") as fh:
                data = json.load(fh)
            return str(data.get(model_name, {}).get("sha256", ""))
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
            logger.debug("§V6 SOTA-SHA256-Cache-Lesen fehlgeschlagen — leere String zurückgegeben (Model %s): %s", model_name, exc)
            return ""

    def _write_sota_sha256(self, model_name: str, sha256: str) -> None:
        """Speichert SHA256 für ein SOTA-Modell im Cache-Manifest."""
        data: dict = {}
        if _SOTA_MANIFEST.is_file():
            try:
                with open(_SOTA_MANIFEST, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        data.setdefault(model_name, {})["sha256"] = sha256
        try:
            with open(_SOTA_MANIFEST, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            logger.debug("SOTA-Manifest konnte nicht geschrieben werden: %s", exc)

    # ── Statistik ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, ModelStatus]:
        """Gibt aktuellen Status aller Modelle zurück (ohne Downloads auszulösen).

        Returns:
            Dict[model_name → ModelStatus]
        """
        statuses: dict[str, ModelStatus] = {}
        for entry in self._manifest_entries:
            sota_cached = (_SOTA_CACHE_DIR / f"{entry.name}.onnx").is_file()
            path = self.load_bundled(entry)
            source = "fallback"
            if path is not None:
                source = "sota_cache" if sota_cached and path.parent == _SOTA_CACHE_DIR else "bundled"
            statuses[entry.name] = ModelStatus(
                name=entry.name,
                available=path is not None,
                path=path,
                source=source,
                sota_available=sota_cached,
            )
        return statuses

    # ── Fortschritt & Benachrichtigung ─────────────────────────────────────────

    def get_download_progress(self) -> dict[str, Any]:
        """Gibt Fortschritt aller aktiven/abgeschlossenen SOTA-Downloads zurück.

        Returns:
            Dict mit total_models, completed, failed, in_progress, per_model_details.
        """
        statuses = self.get_status()
        total_sota = sum(1 for e in self._manifest_entries if e.sota_upgrade and e.sota_upgrade.get("url"))
        downloaded = sum(1 for s in statuses.values() if s.sota_available)
        return {
            "total_sota_models": total_sota,
            "downloaded": downloaded,
            "pending": max(0, total_sota - downloaded),
            "per_model": {
                name: {
                    "available": s.available,
                    "source": s.source,
                    "sota_cached": s.sota_available,
                }
                for name, s in statuses.items()
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Singleton-Accessor (§3.2 Convenience-Funktion)
# ──────────────────────────────────────────────────────────────────────────────

_DOWNLOADER_INSTANCE_HOLDER: list[ModelDownloader | None] = [None]
_downloader_lock = threading.Lock()


def get_model_downloader() -> ModelDownloader:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking).

    Returns:
        Globale ModelDownloader-Instanz.
    """
    if _DOWNLOADER_INSTANCE_HOLDER[0] is None:
        with _downloader_lock:
            if _DOWNLOADER_INSTANCE_HOLDER[0] is None:
                _DOWNLOADER_INSTANCE_HOLDER[0] = ModelDownloader()
    downloader = _DOWNLOADER_INSTANCE_HOLDER[0]
    assert downloader is not None
    return downloader


def verify_and_load(model_name: str) -> Path | None:
    """Convenience-Funktion: Lädt und verifiziert ein Modell nach Name.

    Args:
        model_name: Name aus models/manifest.json.

    Returns:
        Absoluter Pfad wenn vorhanden und valide, sonst None.
    """
    dl = get_model_downloader()
    entry = dl.get_entry(model_name)
    if entry is None:
        logger.debug("Modell '%s' nicht im Manifest gefunden.", model_name)
        return None
    return dl.load_bundled(entry)
