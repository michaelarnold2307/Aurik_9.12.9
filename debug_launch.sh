#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Aurik 9 — Debug-Launcher (außerhalb VS Code Ownership)
#
# Startet das Frontend in einer eigenen Prozess-Session (setsid) so dass
# VS Code den GUI-Prozess nicht verwaltet und beim Schließen des Terminals
# der Prozess NICHT beendet wird.
#
# Features:
#   - AURIK_DEBUG=1 → alle Logger auf DEBUG, dediziertes Timestamp-Log
#   - Qt-Nachrichten (qInstallMessageHandler) landen ebenfalls im Log
#   - faulthandler aktiv (SIGSEGV/SIGABRT → python_faulthandler.log)
#   - setsid: neue Session, kein VS Code als Prozess-Owner
#   - TERM_PROGRAM="" gecleart: verhindert nohup-Bypass in run_aurik.sh
#   - QT_LOGGING_RULES: Qt-Framework-Debug deaktiviert (Spam verm.)
#   - Volle stderr-Ausgabe landet in aurik_debug_latest_console.log
#
# Verwendung:
#   ./debug_launch.sh              # startet detached, kehrt sofort zurück
#   ./debug_launch.sh --foreground # blockiert Terminal (für tmux/screen)
#   ./debug_launch.sh --tail       # startet + öffnet sofort live-tail
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv_aurik/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
CONSOLE_LOG="$LOG_DIR/aurik_debug_latest_console.log"
PID_FILE="$SCRIPT_DIR/temp_repro/aurik_debug.pid"

# ── Voraussetzungen ───────────────────────────────────────────────────────────
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "FEHLER: venv-Python nicht gefunden: $VENV_PYTHON" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$SCRIPT_DIR/temp_repro"

# ── Doppelstart verhindern ────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    _old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$_old_pid" ]] && kill -0 "$_old_pid" 2>/dev/null; then
        echo "⚠  Aurik Debug-Session läuft bereits (PID $_old_pid)."
        echo "   Log: $(cat "$LOG_DIR/aurik_debug_latest.log" 2>/dev/null || echo "$CONSOLE_LOG")"
        echo "   Beenden via: kill $_old_pid"
        exit 0
    fi
fi

# ── Umgebung setzen ───────────────────────────────────────────────────────────
export AURIK_DEBUG=1
# OpenBLAS/OMP: thread-safe, identisch zu main.py §§ (müssen VOR numpy gesetzt sein)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# VS Code-Kennung entfernen → run_aurik.sh verwendet exec statt nohup
unset TERM_PROGRAM
# Qt: keine nativen Dialoge (verhindert portal-Hänger auf Linux, identisch main.py)
export QT_QPA_PLATFORMTHEME=
# Qt-Framework-Debug-Spam deaktivieren (nur Aurik-eigene Logs sind interessant)
export QT_LOGGING_RULES="*.debug=false;qt.qpa=false"
# Python: kein .pyc in debug sessions (Traceback-Zeilennummern immer korrekt)
export PYTHONDONTWRITEBYTECODE=1
# faulthandler: Stack-Trace bei harten Crashes (SIGSEGV etc.)
export PYTHONFAULTHANDLER=1

cd "$SCRIPT_DIR"

# ── Start ─────────────────────────────────────────────────────────────────────
_MODE="${1:-}"

if [[ "$_MODE" == "--foreground" ]]; then
    # Blockierend im Terminal — für tmux/screen
    echo "Aurik Debug-Frontend startet im Vordergrund …"
    echo "Log-Datei (nach Start): $(cat "$LOG_DIR/aurik_debug_latest.log" 2>/dev/null || echo 'wird angelegt')"
    exec "$VENV_PYTHON" -B -W ignore::FutureWarning -W ignore::DeprecationWarning Aurik10/main.py
fi

# Detached in eigener Session (setsid), VS Code ist NICHT Owner
echo -n "Starte Aurik Debug-Frontend (detached, setsid) … "
setsid "$VENV_PYTHON" -B -W ignore::FutureWarning -W ignore::DeprecationWarning Aurik10/main.py \
    >>"$CONSOLE_LOG" 2>&1 &
_pid="$!"
echo "$_pid" >"$PID_FILE"
echo "PID $_pid"
echo ""
echo "  Console-Log (stderr/stdout):  $CONSOLE_LOG"
echo "  Timestamp-Debug-Log:          \$(cat $LOG_DIR/aurik_debug_latest.log)"
echo "  Allgemeines Backend-Log:      $LOG_DIR/aurik_backend.log"
echo "  Crash-Log (faulthandler):     $LOG_DIR/python_faulthandler.log"
echo ""
echo "  Live-Monitoring:"
echo "    tail -f \"\$(cat $LOG_DIR/aurik_debug_latest.log)\""
echo "    tail -f $CONSOLE_LOG"
echo ""
echo "  Prozess beenden:  kill $_pid  oder  kill \$(cat $PID_FILE)"

if [[ "$_MODE" == "--tail" ]]; then
    echo ""
    echo "─── Live-Log-Stream (Strg+C zum Beenden des tails — Frontend läuft weiter) ───"
    # Kurz warten bis main.py den Debug-Log angelegt hat
    for _i in $(seq 1 20); do
        _debug_log="$(cat "$LOG_DIR/aurik_debug_latest.log" 2>/dev/null || true)"
        if [[ -n "$_debug_log" && -f "$_debug_log" ]]; then
            break
        fi
        sleep 0.3
    done
    _debug_log="$(cat "$LOG_DIR/aurik_debug_latest.log" 2>/dev/null || echo "$CONSOLE_LOG")"
    tail -f "$_debug_log" &
    tail -f "$CONSOLE_LOG" &
    wait
fi
