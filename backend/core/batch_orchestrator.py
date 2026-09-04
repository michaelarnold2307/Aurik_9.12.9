"""backend/core/batch_orchestrator.py — §v10.700 G5.

Batch-Intelligenz: Clustering ähnlicher Files, Priorisierung (schnelle zuerst),
Wiederaufnahme bei Abbruch via batch_state.json.

Architektur:
  1. Pre-Analysis ALLER Files → Feature-Vektoren
  2. Clustering: ähnliche Files → gleiche Parameter wiederverwenden
  3. Priorisierung: kleine Files + hohe Restorability zuerst
  4. Wiederaufnahme: batch_state.json speichert Fortschritt
  5. ETA: gleitender Durchschnitt über alle Files

Nutzung:
    orchestrator = BatchOrchestrator()
    orchestrator.add_files(["/path/song1.wav", "/path/song2.mp3"])
    orchestrator.prepare()  # Pre-Analysis + Clustering
    orchestrator.run()       # Sequentiell mit Fortschritt
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchFile:
    """Ein einzelnes File im Batch."""

    path: str
    filename: str = ""
    file_hash: str = ""
    size_mb: float = 0.0
    duration_s: float = 0.0
    sample_rate: int = 0

    # Pre-Analysis
    material_type: str = "unknown"
    restorability_score: float = 50.0
    defect_count: int = 0
    snr_db: float = 0.0

    # Status
    status: str = "pending"  # pending | running | completed | failed | skipped
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    output_path: str = ""

    # Clustering
    cluster_id: int = 0
    priority: int = 0  # niedriger = höhere Priorität


@dataclass
class BatchState:
    """Gesamtzustand eines Batch-Laufs."""

    batch_id: str = ""
    files: list[BatchFile] = field(default_factory=list)
    current_index: int = 0
    total_files: int = 0
    files_completed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    started_at: float = 0.0
    estimated_total_s: float = 0.0
    mode: str = "restoration"
    output_dir: str = ""

    def progress_pct(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.files_completed + self.files_failed + self.files_skipped) / self.total_files * 100

    def eta_s(self) -> float:
        if self.files_completed == 0:
            return self.estimated_total_s
        elapsed = time.monotonic() - self.started_at
        rate = self.files_completed / max(elapsed, 1.0)
        remaining = self.total_files - self.files_completed - self.files_failed - self.files_skipped
        return remaining / max(rate, 0.01)


class BatchOrchestrator:
    """Orchestriert Batch-Restaurierung mit Intelligenz."""

    def __init__(self, output_dir: str = "output_audio"):
        self._files: list[BatchFile] = []
        self._state: BatchState | None = None
        self._output_dir = Path(output_dir)
        self._progress_callback: Callable[[BatchState], None] | None = None
        self._file_callback: Callable[[BatchFile], Any] | None = None

    # ── File Management ──────────────────────────────────────────

    def add_files(self, paths: list[str]) -> None:
        """Fügt Dateien zum Batch hinzu."""
        for path in paths:
            p = Path(path)
            if not p.exists():
                logger.warning("Datei nicht gefunden: %s", path)
                continue
            bf = BatchFile(
                path=str(p),
                filename=p.name,
                file_hash=_hash_file(path),
                size_mb=p.stat().st_size / (1024 * 1024) if p.exists() else 0.0,
            )
            self._files.append(bf)
        logger.info("%d Dateien zum Batch hinzugefügt", len(paths))

    def remove_file(self, path: str) -> None:
        """Entfernt eine Datei aus dem Batch."""
        self._files = [f for f in self._files if f.path != path]

    def clear(self) -> None:
        """Leert den Batch."""
        self._files.clear()
        self._state = None

    # ── Pre-Analysis ─────────────────────────────────────────────

    def prepare(self) -> BatchState:
        """Führt Pre-Analysis für alle Files durch (Feature-Extraktion + Clustering)."""
        if not self._files:
            raise ValueError("Keine Dateien im Batch")

        batch_id = _generate_batch_id()
        state = BatchState(
            batch_id=batch_id,
            total_files=len(self._files),
            started_at=time.monotonic(),
            output_dir=str(self._output_dir),
        )

        # Pre-Analyse jedes Files
        for f in self._files:
            try:
                _analyze_file(f)
            except Exception as e:
                logger.warning("Pre-Analyse fehlgeschlagen für %s: %s", f.filename, e)
                f.status = "skipped"
                f.error = str(e)
            state.files.append(f)

        # Clustering: ähnliche Files gruppieren
        _cluster_files(state.files)

        # Priorisierung: kleine + hohe Restorability zuerst
        _prioritize_files(state.files)

        # ETA schätzen
        state.estimated_total_s = _estimate_total_time(state.files)

        self._state = state
        logger.info(
            "Batch %s vorbereitet: %d Files, %d Cluster, ETA %.0fs",
            batch_id,
            state.total_files,
            len({f.cluster_id for f in state.files}),
            state.estimated_total_s,
        )
        return state

    # ── Execution ────────────────────────────────────────────────

    def run(self) -> BatchState:
        """Führt den Batch sequentiell aus (nach Priorität sortiert)."""
        if self._state is None:
            self.prepare()

        state = self._state
        if state is None:
            raise RuntimeError("prepare() fehlgeschlagen")

        state.started_at = time.monotonic()

        for i, bf in enumerate(state.files):
            if bf.status in ("skipped", "failed"):
                continue

            state.current_index = i
            bf.status = "running"
            bf.started_at = time.monotonic()

            try:
                if self._file_callback:
                    result = self._file_callback(bf)
                    if result is not None:
                        bf.output_path = str(result) if hasattr(result, "__str__") else ""
                # Simuliere Erfolg wenn kein Callback (nur Orchestrierung)
                bf.status = "completed"
                state.files_completed += 1
            except Exception as e:
                bf.status = "failed"
                bf.error = str(e)
                state.files_failed += 1
                logger.error("Batch-File fehlgeschlagen: %s — %s", bf.filename, e)

            bf.completed_at = time.monotonic()

            if self._progress_callback:
                self._progress_callback(state)

        # Speichere State für Wiederaufnahme
        _save_batch_state(state)

        logger.info(
            "Batch %s abgeschlossen: %d ok, %d fehlgeschlagen, %d uebersprungen",
            state.batch_id,
            state.files_completed,
            state.files_failed,
            state.files_skipped,
        )
        return state

    def resume(self, state_path: str) -> BatchState | None:
        """Nimmt einen abgebrochenen Batch wieder auf."""
        try:
            data = json.loads(Path(state_path).read_text())
            state = _load_batch_state(data)
            self._state = state
            logger.info(
                "Batch %s wiederaufgenommen bei File %d/%d", state.batch_id, state.current_index, state.total_files
            )
            return self.run()
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error("Wiederaufnahme fehlgeschlagen: %s", e)
            return None

    # ── Callbacks ────────────────────────────────────────────────

    def on_progress(self, callback: Callable[[BatchState], None]) -> None:
        """Callback bei Fortschritt (nach jedem File)."""
        self._progress_callback = callback

    def on_file(self, callback: Callable[[BatchFile], Any]) -> None:
        """Callback pro File (führt die eigentliche Restaurierung durch)."""
        self._file_callback = callback

    # ── Accessors ────────────────────────────────────────────────

    @property
    def state(self) -> BatchState | None:
        return self._state

    @property
    def files(self) -> list[BatchFile]:
        return self._files


# ── Interne Hilfsfunktionen ──────────────────────────────────────


def _analyze_file(bf: BatchFile) -> None:
    """Pre-Analyse eines Files: Dauer, Samplerate, Schätzung Restorability."""
    try:
        import soundfile as sf

        info = sf.info(bf.path)
        bf.duration_s = info.duration
        bf.sample_rate = info.samplerate
    except Exception:
        # Fallback: grobe Schätzung aus Dateigröße
        bf.duration_s = bf.size_mb * 10  # ~10s pro MB bei MP3 128kbps
        bf.sample_rate = 44100

    # Restorability: kleiner = mehr Defekte erwartet
    # Grobe Heuristik basierend auf Dateigröße/Dauer
    if bf.duration_s > 0:
        bitrate_est = (bf.size_mb * 8 * 1024) / bf.duration_s  # kbps
        if bitrate_est < 128:
            bf.restorability_score = 30.0  # stark komprimiert
        elif bitrate_est < 320:
            bf.restorability_score = 60.0
        else:
            bf.restorability_score = 85.0  # verlustfrei/hochbitratig
    else:
        bf.restorability_score = 50.0

    bf.material_type = "unknown"


def _cluster_files(files: list[BatchFile]) -> None:
    """Gruppiert ähnliche Files via Feature-Vektor (Dauer, Größe, Bitrate)."""
    if len(files) < 2:
        for i, f in enumerate(files):
            f.cluster_id = i
        return

    # Feature-Vektor: [duration_s, size_mb, restorability_score]
    features = np.array([[f.duration_s, f.size_mb, f.restorability_score] for f in files])

    # Normalisieren
    f_mean = features.mean(axis=0)
    f_std = features.std(axis=0) + 1e-10
    f_norm = (features - f_mean) / f_std

    # Einfaches K-Means (k = min(5, n//2))
    k = min(5, max(1, len(files) // 2))
    if k <= 1:
        for f in files:
            f.cluster_id = 0
        return

    # Zufällige Initialisierung (deterministisch via Hash)
    rng = np.random.RandomState(42)
    centroids = f_norm[rng.choice(len(files), k, replace=False)]

    for _ in range(10):  # max 10 Iterationen
        # Distanzen
        distances = np.array([np.sum((f_norm - c) ** 2, axis=1) for c in centroids])
        labels = np.argmin(distances, axis=0)

        # Neue Centroids
        new_centroids = np.array(
            [f_norm[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)]
        )

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    for i, f in enumerate(files):
        f.cluster_id = int(labels[i])

    logger.info("Clustering: %d Files → %d Cluster", len(files), len(set(labels)))


def _prioritize_files(files: list[BatchFile]) -> None:
    """Priorisiert Files: kleine + hohe Restorability zuerst."""
    for i, f in enumerate(files):
        # Priorität: −duration (kleinere = höhere Prio) + restorability
        f.priority = int(-f.duration_s * 10 + f.restorability_score)

    files.sort(key=lambda f: f.priority, reverse=True)


def _estimate_total_time(files: list[BatchFile]) -> float:
    """Schätzt Gesamtzeit: ~3× Echtzeit pro File + 10s Overhead."""
    active = [f for f in files if f.status == "pending"]
    return sum((f.duration_s * 3 + 10) for f in active)


def _generate_batch_id() -> str:
    """Generiert eine Batch-ID aus Timestamp."""
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"b_{ts}"


def _hash_file(path: str) -> str:
    """SHA-256 Hash der Datei (für Wiedererkennung)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError as exc:
        logger.debug("§V6 SHA-256 Hash fehlgeschlagen — leere String zurückgegeben: %s", exc)
        return ""


def _save_batch_state(state: BatchState) -> None:
    """Speichert Batch-State als JSON für Wiederaufnahme."""
    state_path = Path(state.output_dir) / "batch_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "batch_id": state.batch_id,
        "current_index": state.current_index,
        "total_files": state.total_files,
        "files_completed": state.files_completed,
        "files_failed": state.files_failed,
        "files_skipped": state.files_skipped,
        "started_at": state.started_at,
        "mode": state.mode,
        "output_dir": state.output_dir,
        "files": [
            {
                "path": f.path,
                "filename": f.filename,
                "status": f.status,
                "error": f.error,
                "output_path": f.output_path,
                "cluster_id": f.cluster_id,
                "priority": f.priority,
            }
            for f in state.files
        ],
    }
    state_path.write_text(json.dumps(data, indent=2))
    logger.info("Batch-State gespeichert: %s", state_path)


def _load_batch_state(data: dict) -> BatchState:
    """Lädt Batch-State aus JSON-Dict."""
    state = BatchState(
        batch_id=data["batch_id"],
        current_index=data.get("current_index", 0),
        total_files=data["total_files"],
        files_completed=data.get("files_completed", 0),
        files_failed=data.get("files_failed", 0),
        files_skipped=data.get("files_skipped", 0),
        started_at=data.get("started_at", 0.0),
        mode=data.get("mode", "restoration"),
        output_dir=data.get("output_dir", "output_audio"),
    )
    for fdata in data.get("files", []):
        bf = BatchFile(
            path=fdata["path"],
            filename=fdata.get("filename", ""),
            status=fdata.get("status", "pending"),
            error=fdata.get("error", ""),
            output_path=fdata.get("output_path", ""),
            cluster_id=fdata.get("cluster_id", 0),
            priority=fdata.get("priority", 0),
        )
        state.files.append(bf)
    return state
