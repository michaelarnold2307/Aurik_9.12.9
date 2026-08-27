"""Aurik-Venv: deterministische Prozess-Kodierung (UTF-8) erzwingen.

Befund 2026-08-16: In diesem Venv schrieb Python stdout/stderr mit einer
Nicht-UTF-8-Kodierung (UTF-16LE), was Tooling-/Log-Diagnosen verfälschte
(Mojibake in Pipes/Dateien). Diese sitecustomize stellt UTF-8 für alle
Prozesse dieses Interpreters sicher — unabhängig von PYTHONIOENCODING/LANG.

Repo-Quelle: scripts/venv_sitecustomize.py. Bei Venv-Neuaufbau kopieren:
    cp scripts/venv_sitecustomize.py <venv>/lib/pythonX.Y/site-packages/sitecustomize.py
"""

import sys


def _force_utf8(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8")
    except Exception:
        pass


_force_utf8(sys.stdout)
_force_utf8(sys.stderr)
