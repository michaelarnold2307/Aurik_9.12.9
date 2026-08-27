"""
LEGACY_NON_RELEASE: FastAPI Batch Processing Endpunkte mit Multi-Batch-Support.

Dieser Serverpfad ist nicht Teil des Desktop-/AppImage-Releasepfads. Release-fähige
GUI/CLI/Batch-Flows müssen den Canonical Contract über backend.api.bridge nutzen.
Migriert von Flask (Port 5000) zu FastAPI (Port 8000) mit Batch-IDs.
"""

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as _np
import soundfile as _sf
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from backend.api.bridge import export_guard as _export_guard
from backend.file_import import load_audio_file
from denker.aurik_denker import get_aurik_denker

# Setup Router
router = APIRouter(prefix="/batch", tags=["batch"])

# Batch-Status-Verwaltung (Multi-Batch-Support)
batch_jobs: dict[str, dict[str, Any]] = {}
batch_lock = threading.Lock()

# Konfiguration
AUDIO_IN_DIR = Path("input_audio")
AUDIO_OUT_DIR = Path("output_audio")
AUDIO_OUT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


def batch_worker(batch_id: str, input_files: list[str]):
    """
    Worker-Thread für Batch-Verarbeitung via AurikDenker (UV3).

    Ersetzt die veraltete AdaptiveProcessingPipeline (v8.x) die HIPS-Pre-checks
    und nicht-installiertes openai-whisper benötigte.
    """
    try:
        denker = get_aurik_denker()

        with batch_lock:
            batch_jobs[batch_id]["total"] = len(input_files)
            batch_jobs[batch_id]["status"] = "processing"

        for idx, fname in enumerate(input_files):
            in_path = AUDIO_IN_DIR / fname
            out_path = AUDIO_OUT_DIR / fname

            with batch_lock:
                batch_jobs[batch_id]["last_file"] = fname
                batch_jobs[batch_id]["current_file"] = fname

            try:
                _loaded = load_audio_file(str(in_path), do_carrier_analysis=False)
                if _loaded is None or _loaded.get("error"):
                    raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {in_path}")
                audio = _np.asarray(_loaded["audio"], dtype=_np.float32)
                sr = int(_loaded["sr"])

                denker_result: Any = denker.denke(audio, sr, mode="restoration", input_path=str(in_path))
                result_sr = int(getattr(denker_result, "sample_rate", sr) or sr)
                result_audio = _np.asarray(denker_result.audio, dtype=_np.float32)
                result_quality = float(getattr(denker_result, "quality_estimate", 0.0) or 0.0)
                result_material = str(
                    getattr(denker_result, "material", "") or getattr(denker_result, "material_type", "")
                )
                result_deferred = list(getattr(denker_result, "deferred_phases", []) or [])

                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                try:
                    _sf.write(str(tmp_path), _export_guard(result_audio), result_sr)
                    tmp_path.replace(out_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)

                audit_path = out_path.with_suffix(".json")
                with open(str(audit_path), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "filename": fname,
                            "quality_estimate": result_quality,
                            "material": result_material,
                            "deferred_phases": result_deferred,
                        },
                        f,
                        indent=2,
                    )

                logger.info("[Batch %s] verarbeitet: %s", batch_id, fname)

            except Exception as e:
                logger.exception("[Batch %s] Error processing %s: %s", batch_id, fname, e)
                with batch_lock:
                    if "errors" not in batch_jobs[batch_id]:
                        batch_jobs[batch_id]["errors"] = []
                    batch_jobs[batch_id]["errors"].append({"file": fname, "error": str(e)})

            with batch_lock:
                batch_jobs[batch_id]["progress"] = idx + 1

        # Batch abgeschlossen
        with batch_lock:
            batch_jobs[batch_id]["status"] = "completed"
            batch_jobs[batch_id]["current_file"] = None
            logger.info("[Batch %s] abgeschlossen", batch_id)

    except Exception as e:
        logger.exception("[Batch %s] Fatal error: %s", batch_id, e)
        with batch_lock:
            batch_jobs[batch_id]["status"] = "failed"
            batch_jobs[batch_id]["error"] = str(e)


@router.post("/start")
async def start_batch(background_tasks: BackgroundTasks, files: list[UploadFile] | None = None):
    """
    Startet einen neuen Batch-Job

    Args:
        files: Optional - Liste von Upload-Dateien. Falls None, werden Dateien aus input_audio/ verwendet

    Returns:
        dict: {"batch_id": str, "status": str, "total_files": int}
    """
    try:
        input_files = []

        # Option 1: Files wurden hochgeladen
        if files and len(files) > 0:
            # Speichere hochgeladene Dateien temporär
            for upload_file in files:
                if upload_file.filename and upload_file.filename.strip():
                    # §Security: Path-Traversal verhindern — nur Dateiname, kein Pfad
                    _safe_name = Path(upload_file.filename).name
                    if not _safe_name:
                        continue
                    # Validate safe name to prevent path traversal
                    if "/" in _safe_name or "\\" in _safe_name or ".." in _safe_name:
                        logger.warning("Skipping unsafe file name: %s", _safe_name)
                        continue
                    file_path = AUDIO_IN_DIR / _safe_name
                    with open(str(file_path), "wb") as f:
                        content = await upload_file.read()
                        f.write(content)
                    input_files.append(_safe_name)
                    logger.info("Uploaded file: %s", _safe_name)

        # Option 2: Verwende existierende Dateien aus input_audio/
        if not input_files:
            if not AUDIO_IN_DIR.exists():
                raise HTTPException(status_code=400, detail=f"Input directory nicht gefunden: {AUDIO_IN_DIR}")

            input_files = [
                f.name
                for f in AUDIO_IN_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in [".wav", ".flac", ".mp3", ".ogg"]
            ]

        if not input_files:
            raise HTTPException(
                status_code=400,
                detail="Keine Audio-Dateien zum Verarbeiten (weder hochgeladen noch im input_audio/ Verzeichnis)",
            )

        # Generiere eindeutige Batch-ID
        batch_id = str(uuid.uuid4())

        # Initialisiere Batch-Status
        with batch_lock:
            batch_jobs[batch_id] = {
                "batch_id": batch_id,
                "status": "queued",
                "progress": 0,
                "total": len(input_files),
                "last_file": None,
                "current_file": None,
                "files": input_files,
                "errors": [],
            }

        # Starte Batch-Worker in Background-Task
        background_tasks.add_task(batch_worker, batch_id, input_files)

        logger.info("[Batch %s] gestartet with %s files", batch_id, len(input_files))

        return {"batch_id": batch_id, "status": "started", "total_files": len(input_files)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error starting batch")
        raise HTTPException(status_code=500, detail=f"Batch-Start fehlgeschlagen: {e}") from e


@router.get("/status/{batch_id}")
async def batch_status(batch_id: str):
    """
    Gibt Status eines Batch-Jobs zurück

    Args:
        batch_id: Eindeutige Batch-ID

    Returns:
        dict: Batch-Status mit progress, total, status, etc.
    """
    with batch_lock:
        if batch_id not in batch_jobs:
            raise HTTPException(status_code=404, detail="Batch-ID nicht gefunden")

        status_data = batch_jobs[batch_id].copy()

    return status_data


@router.get("/result/{batch_id}")
async def batch_result(batch_id: str):
    """
    Gibt Ergebnis-Dateien eines Batch-Jobs zurück

    Args:
        batch_id: Eindeutige Batch-ID

    Returns:
        dict: {"batch_id": str, "output_files": list, "errors": list}
    """
    with batch_lock:
        if batch_id not in batch_jobs:
            raise HTTPException(status_code=404, detail="Batch-ID nicht gefunden")

        job = batch_jobs[batch_id]

    # Sammle Output-Dateien
    output_files = []
    if AUDIO_OUT_DIR.exists():
        for f in AUDIO_OUT_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in [".wav", ".flac", ".mp3", ".ogg"]:
                output_files.append({"filename": f.name, "size": f.stat().st_size, "path": str(f)})

    return {
        "batch_id": batch_id,
        "status": job["status"],
        "output_files": output_files,
        "errors": job.get("errors", []),
    }


@router.get("/audit/{batch_id}")
async def batch_audit(batch_id: str):
    """
    Gibt Audit-Reports für Batch-Job zurück

    Args:
        batch_id: Eindeutige Batch-ID

    Returns:
        dict: {"batch_id": str, "audits": list}
    """
    with batch_lock:
        if batch_id not in batch_jobs:
            raise HTTPException(status_code=404, detail="Batch-ID nicht gefunden")

    # Sammle alle Audit-JSON-Dateien
    audits = []
    if AUDIO_OUT_DIR.exists():
        for f in AUDIO_OUT_DIR.iterdir():
            if f.is_file() and f.suffix == ".json" and "_audit" not in f.name:
                try:
                    with open(str(f), encoding="utf-8") as fp:
                        audit_data = json.load(fp)
                        audits.append({"filename": f.name, "data": audit_data})
                except Exception as e:
                    logger.warning("Could not read audit file %s: %s", f, e)

    return {"batch_id": batch_id, "audits": audits}


@router.get("/list")
async def list_batches():
    """
    Gibt alle Batch-Jobs zurück

    Returns:
        dict: {"batches": list}
    """
    with batch_lock:
        batches = [
            {"batch_id": bid, "status": job["status"], "progress": job["progress"], "total": job["total"]}
            for bid, job in batch_jobs.items()
        ]

    return {"batches": batches}


@router.delete("/cancel/{batch_id}")
async def cancel_batch(batch_id: str):
    """
    Bricht einen laufenden Batch-Job ab (Status wird auf 'cancelled' gesetzt)

    Args:
        batch_id: Eindeutige Batch-ID

    Returns:
        dict: {"batch_id": str, "status": "cancelled"}
    """
    with batch_lock:
        if batch_id not in batch_jobs:
            raise HTTPException(status_code=404, detail="Batch-ID nicht gefunden")

        if batch_jobs[batch_id]["status"] in ["completed", "failed", "cancelled"]:
            raise HTTPException(status_code=400, detail=f"Batch bereits {batch_jobs[batch_id]['status']}")

        batch_jobs[batch_id]["status"] = "cancelled"

    logger.info("[Batch %s] Cancelled by user", batch_id)

    return {"batch_id": batch_id, "status": "cancelled"}
