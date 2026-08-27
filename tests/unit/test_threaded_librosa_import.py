"""Threaded-Import-Smoke: librosa-Erstzugriffe aus parallelen Threads (Spec 24).

Root-Fix-Regression 2026-08-16: Im GUI-Prozess crashten parallele
librosa-Erstimporte (bridge-Warmup-Daemon vs. Hauptthread) mit
- AttributeError: 'function' object has no attribute 'get_call_template'
- KeyError: 'scipy.sparse._construct'
Unit-Tests laufen single-threaded mit warmem Cache und sehen diese Klasse
nie — deshalb testen wir sie im SUBPROZESS mit frischem numba-Cache.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

_SCRIPT_WITH_BOOTSTRAP = """
import threading
import numpy as np
from backend.core.librosa_bootstrap import ensure_librosa_ready

assert ensure_librosa_ready() is True
errors = []

def load(sub, fn, args):
    try:
        import importlib
        import librosa
        importlib.import_module(sub)
        getattr(getattr(librosa, fn[0]), fn[1])(**args)
    except Exception as exc:  # noqa: BLE001
        errors.append((fn[1], type(exc).__name__, str(exc)[:120]))

jobs = [
    ("librosa.core.constantq", ("feature", "chroma_cqt"),
     {"y": np.zeros(11025, dtype=np.float32) + 0.1, "sr": 22050}),
    ("librosa.onset", ("onset", "onset_strength"),
     {"y": np.zeros(22050, dtype=np.float32), "sr": 22050}),
    ("librosa.beat", ("feature", "mfcc"),
     {"y": np.zeros(4096, dtype=np.float32), "sr": 8000, "n_mfcc": 13}),
    ("librosa.core.pitch", ("feature", "zero_crossing_rate"),
     {"y": np.zeros(4096, dtype=np.float32)}),
]
threads = [threading.Thread(target=load, args=j) for j in jobs]
for t in threads:
    t.start()
for t in threads:
    t.join()
if errors:
    print("THREADED_IMPORT_FAILED", errors)
    raise SystemExit(1)
print("THREADED_IMPORT_OK")
"""

_SCRIPT_NO_BOOTSTRAP_DOC = """
# Dokumentations-Lauf: OHNE Bootstrap KANN das Race zuschlagen (KeyError/
# get_call_template). Der Lauf ist probabilistisch und wird daher nicht
# als harter Test geführt — die Garantie ist der WITH-Bootstrap-Test.
"""


def _run_script(script: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as cache_dir:
        env = {"NUMBA_CACHE_DIR": cache_dir, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env={k: v for k, v in __import__("os").environ.items() if k not in env} | env,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_bootstrap_prevents_parallel_first_import_race() -> None:
    """Mit ensure_librosa_ready() im Hauptthread laufen 4 parallele
    Erst-Importe + Feature-Aufrufe fehlerfrei (frischer numba-Cache)."""
    code, out = _run_script(_SCRIPT_WITH_BOOTSTRAP)
    assert code == 0, f"Subprozess scheiterte (exit={code}): {out[-2000:]}"
    assert "THREADED_IMPORT_OK" in out


def test_bootstrap_idempotent_and_threadsafe_in_process() -> None:
    """ensure_librosa_ready() ist idempotent und aus N Threads parallel sicher."""
    import threading

    from backend.core.librosa_bootstrap import ensure_librosa_ready

    results: list[bool] = []

    def call() -> None:
        results.append(ensure_librosa_ready())

    first = ensure_librosa_ready()
    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert first is True
    assert all(results)
    # Zweiter Durchlauf = No-Op-Pfad (Flag)
    assert ensure_librosa_ready() is True


def test_bootstrap_makes_cqt_and_onset_usable() -> None:
    """Nach dem Bootstrap sind chroma_cqt/onset_strength im selben Prozess
    nutzbar (kein lazy_loader-Nachladen im Worker)."""
    script = """
import numpy as np
from backend.core.librosa_bootstrap import ensure_librosa_ready
ensure_librosa_ready()
import librosa
librosa.feature.chroma_cqt(y=np.zeros(11025, dtype=np.float32) + 0.1, sr=22050)
librosa.onset.onset_strength(y=np.zeros(22050, dtype=np.float32), sr=22050)
print("FEATURES_OK")
"""
    code, out = _run_script(script)
    assert code == 0, f"Subprozess scheiterte (exit={code}): {out[-2000:]}"
    assert "FEATURES_OK" in out
