"""
LEGACY_NON_RELEASE: Minimal-API für historische SOTA-Batch-Workflows.

Dieser Serverpfad ist nicht Teil des Desktop-/AppImage-Releasepfads. Release-fähige
GUI/CLI/Batch-Flows müssen den Canonical Contract über backend.api.bridge nutzen.
Bietet REST-Endpunkte für Start, Status, Ergebnis und Audit-Report.
"""

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from flask import Flask, jsonify

from backend.api.bridge import export_guard
from backend.file_import import load_audio_file

try:
    from dsp.dsp_decision_logic import DSPDecisionLogic  # type: ignore[import]

    _DSP_DECISION_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
    _DSP_DECISION_AVAILABLE = False

    class DSPDecisionLogic:  # type: ignore[no-redef]
        """Stub for DSPDecisionLogic when module is not available."""

        def __init__(self, config_path: str) -> None:
            raise RuntimeError("DSPDecisionLogic not available. Install backend.core.dsp_decision_logic.")

        def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
            """Gibt Audio unverändert zurück, wenn DSPDecisionLogic nicht verfügbar ist."""
            _ = sr
            return audio


logger = logging.getLogger(__name__)

app = Flask(__name__)

AUDIO_IN_DIR = Path("input_audio")
AUDIO_OUT_DIR = Path("output_audio")
CONFIG = "config_dsp_chain_example.yaml"
BATCH_STATUS: dict[str, Any] = {
    "running": False,
    "progress": 0,
    "total": 0,
    "last_file": None,
}


def batch_worker() -> None:
    """Verarbeitet historische Batch-Dateien im Legacy-REST-Pfad."""
    logic = DSPDecisionLogic(config_path=CONFIG)
    files = [p.name for p in AUDIO_IN_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    BATCH_STATUS["total"] = len(files)
    BATCH_STATUS["running"] = True
    for idx, fname in enumerate(files):
        in_path = AUDIO_IN_DIR / fname
        out_path = AUDIO_OUT_DIR / fname
        BATCH_STATUS["last_file"] = fname
        try:
            _loaded = load_audio_file(str(in_path), do_carrier_analysis=False)
            if _loaded is None or _loaded.get("error"):
                raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {in_path}")
            audio, sr = _loaded["audio"], int(_loaded["sr"])
            object.__setattr__(logic, "output_path_hint", out_path)
            result = logic.process(audio, sr)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            try:
                sf.write(tmp_path, export_guard(result), sr)
                tmp_path.replace(out_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error("[BatchAPI] Fehler bei %s: %s", in_path, e)
        BATCH_STATUS["progress"] = idx + 1
    BATCH_STATUS["running"] = False


@app.route("/batch/start", methods=["POST"])
def start_batch():
    """Startet den historischen REST-Batch-Worker."""
    if BATCH_STATUS["running"]:
        return jsonify({"status": "already running"}), 409
    threading.Thread(target=batch_worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/batch/status", methods=["GET"])
def batch_status():
    """Gibt den aktuellen Legacy-Batch-Status zurück."""
    return jsonify(BATCH_STATUS)


@app.route("/batch/result", methods=["GET"])
def batch_result():
    """Listet erzeugte Audiodateien des Legacy-Batch-Pfads."""
    files = [p.name for p in AUDIO_OUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    return jsonify({"files": files})


@app.route("/batch/audit", methods=["GET"])
def batch_audit():
    """Listet erzeugte Audit-Dateien des Legacy-Batch-Pfads."""
    audits = [p.name for p in AUDIO_OUT_DIR.iterdir() if p.is_file() and p.name.endswith("_audit.json")]
    return jsonify({"audits": audits})


if __name__ == "__main__":
    AUDIO_OUT_DIR.mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=5000)  # nosec B104 - LEGACY_NON_RELEASE local debug server
