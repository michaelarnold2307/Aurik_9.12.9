"""§v10.17 FallbackAuditor — verhindert stille Degradation.

Jeder Fallback wird registriert und am Pipeline-Ende als konsolidierter
Report ausgegeben. Der Nutzer sieht sofort: „Aurik lief im DEGRADED-Modus".

Prinzip: Gold-Standard zuerst. Fallback nur wenn nötig. Aber: NIEMALS still.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FallbackEvent:
    component: str  # "SingMOS", "UV3", "HPE", etc.
    gold_standard: str  # "versa_singmos_pro", "unified_restorer_v3", etc.
    fallback_used: (
        str  # "pqs_dsp", "passthrough", etc.  # §V6 (copilot-instructions.md): logger.warning handled at call site
    )
    reason: str  # "device_mismatch", "syntax_error", "timeout", etc.
    severity: str = "warning"  # "info", "warning", "error"


class FallbackAuditor:
    """Zentraler Registrar für alle Fallback-Ereignisse."""

    _auto_detect_installed: bool = False

    def __init__(self):
        self._events: list[FallbackEvent] = []
        self._lock = threading.Lock()

    @property
    def degraded(self) -> bool:
        return len(self._events) > 0

    @property
    def should_block_pipeline(self) -> bool:
        """True wenn die Pipeline wegen zu vieler Fallbacks blockiert werden sollte."""
        return self.cascade_exceeded or any(
            e.severity == "error" for e in self._events if "fatal" in str(e.reason).lower()
        )

    @property
    def has_critical_degradation(self) -> bool:
        return any(e.severity == "error" for e in self._events)

    # ── Cascade Management (§v10.17) ────────────────────────────────

    _MAX_CASCADE_DEPTH: int = 5  # Max 5 Fallbacks pro Komponente
    _ESCALATE_AFTER: int = 3  # Ab 3. Fallback → severity="error"
    _BLOCK_AFTER: int = 8  # Ab 8 Fallbacks gesamt → Pipeline blockieren

    @property
    def cascade_exceeded(self) -> bool:
        """True wenn die Fallback-Tiefe überschritten wurde."""
        return len(self._events) >= self._BLOCK_AFTER

    def record(self, component: str, gold: str, fallback: str, reason: str, severity: str = "warning"):
        with self._lock:
            # Cascade depth check per component
            same_component = sum(1 for e in self._events if e.component == component)
            if same_component >= self._MAX_CASCADE_DEPTH:
                logger.error("FallbackAuditor CASCADE-EXCEEDED: %s hat %d Fallbacks", component, same_component)
                severity = "error"
            elif same_component >= self._ESCALATE_AFTER:
                severity = "error"

            self._events.append(
                FallbackEvent(
                    component=component,
                    gold_standard=gold,
                    fallback_used=fallback,
                    reason=reason,
                    severity=severity,
                )
            )
            if self.cascade_exceeded:
                logger.critical(
                    "FallbackAuditor BLOCK: %d Fallbacks gesamt — Pipeline-Qualität gefährdet", len(self._events)
                )
        # Handler-based auto-detect catches this — don't double-log
        pass

    def reset(self) -> None:
        """Leert alle Events — Song-Scope-Reset (§V8/§G1, copilot-instructions.md).

        Ohne Reset akkumulieren Degradations-Events session-global und der
        PreExportValidator blockiert ab _BLOCK_AFTER Events fälschlich Exporte
        späterer Songs (Rev. 2026-08-16).
        """
        with self._lock:
            self._events.clear()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "degraded": self.degraded,
                "critical": self.has_critical_degradation,
                "total_fallbacks": len(self._events),
                "events": [
                    {"component": e.component, "gold": e.gold_standard, "fallback": e.fallback_used, "reason": e.reason}
                    for e in self._events
                ],
            }

    def report(self) -> str:
        if not self.degraded:
            return "Aurik: GOLD-STANDARD — alle Komponenten in voller Qualität."
        lines = [f"Aurik: DEGRADED — {len(self._events)} Fallback(s) aktiv:"]
        for e in self._events:
            lines.append(f"  {e.component}: {e.gold_standard} → {e.fallback_used} ({e.reason})")
        return "\n".join(lines)

    # ── Auto-Detect Middleware ───────────────────────────────────────

    _fallback_patterns: list[tuple[str, str, str, str]] = [
        # (keyword_in_message, component, gold_standard, fallback_name)
        (
            "DSP-Fallback",
            "ML-Model",
            "GPU",
            "DSP",
        ),  # §V6 (copilot-instructions.md): logger.warning handled at call site
        (
            "DSP fallback",
            "ML-Model",
            "GPU",
            "DSP",
        ),  # §V6 (copilot-instructions.md): logger.warning handled at call site
        ("pYIN-Fallback", "PitchDetection", "FCPE/CREPE", "pYIN"),
        ("passthrough", "Pipeline", "FullRestoration", "Passthrough"),
        ("nicht verfügbar", "Plugin", "FullML", "DSP"),
        ("not available", "Plugin", "FullML", "DSP"),
        ("CPU-Only", "GPU", "HIP/CUDA", "CPU"),
        ("CPU only", "GPU", "HIP/CUDA", "CPU"),
        ("übersprungen", "Component", "FullProcessing", "Skipped"),
        ("skipped", "Component", "FullProcessing", "Skipped"),
        ("Timeout", "Component", "Full", "Timeout"),
        ("Fallback auf", "Export", "Primary", "Fallback"),
        ("Emergency-Eviction", "PluginLifecycle", "FullPlugin", "Evicted"),
        ("eviction", "PluginLifecycle", "FullPlugin", "Evicted"),
    ]

    @classmethod
    def enable_auto_detect(cls):
        if cls._auto_detect_installed:
            return  # Already installed — no double-registration
        """Installiert einen Logging-Filter der Fallback-Muster automatisch erkennt.

        Nach dem Aufruf werden ALLE Logger automatisch auf Fallback-Patterns
        geprüft — kein manuelles Instrumentieren mehr nötig.
        """
        import logging

        _auditor = get_fallback_auditor()
        _patterns = cls._fallback_patterns

        class FallbackDetectHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = record.getMessage()
                    for keyword, comp, gold, fallb in _patterns:
                        if keyword.lower() in msg.lower():
                            _auditor.record(comp, gold, fallb, keyword.lower().replace(" ", "_")[:50])
                            break  # one match per message
                except Exception as _fa_emit_exc:
                    logger.debug(
                        "Ersatzpfad_auditor: log aufzeichnen processing fehlgeschlagen (unkritisch): %s", _fa_emit_exc
                    )

        # An Root-Logger hängen — fängt ALLE Logs ab
        root = logging.getLogger()
        handler = FallbackDetectHandler()
        handler.setLevel(logging.WARNING)
        root.addHandler(handler)
        FallbackAuditor._auto_detect_installed = True


_instance: FallbackAuditor | None = None
_lock = threading.Lock()


def get_fallback_auditor() -> FallbackAuditor:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FallbackAuditor()
    return _instance
