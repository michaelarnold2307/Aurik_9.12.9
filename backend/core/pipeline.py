"""
Aurik 10.0.0 – Normative Pipeline
Abbild der normativen Verarbeitungskette: Epistemic Gate → Zonen → Conduct → Regulator
"""

from backend.core.audit_log.audit_log import AuditLog
from backend.core.conduct_enforcer.conduct_enforcer import ConductEnforcer
from backend.core.epistemic_gate.epistemic_gate import EpistemicGate
from backend.core.regulator.regulator import Regulator
from backend.core.zone_engine.zone_engine import ZoneEngine


class AurikPipeline:
    def __init__(self):
        self.epistemic_gate = EpistemicGate()
        self.zone_engine = ZoneEngine()
        self.conduct_enforcer = ConductEnforcer()
        self.regulator = Regulator()
        self.audit_log = AuditLog()

    def process(self, audio_data, tontraegerkette_info):
        # 1. Epistemic Gate
        if not self.epistemic_gate.check_responsibility(audio_data):
            self.audit_log.log_run({"status": "abgebrochen", "grund": "Nicht zuständig"})
            return "Nicht zuständig"
        # 2. Zonen-Engine
        zones = self.zone_engine.segment(audio_data)  # type: ignore[attr-defined]
        # 3. Conduct Enforcer
        conduct_results = self.conduct_enforcer.enforce(zones)  # type: ignore[attr-defined]
        # 4. Regulator
        regulation = self.regulator.regulate(zones, tontraegerkette_info)
        # 5. Audit-Log
        self.audit_log.log_run(
            {"status": "erfolgreich", "zones": zones, "conduct": conduct_results, "regulation": regulation}
        )
        return regulation


# ── Alias-Klassen (für Dead-Import-Reparatur) ───────────────────────

class RestorationPipeline(AurikPipeline):
    """Alias für AurikPipeline — für Dead-Import-Kompatibilität."""

    def __init__(self) -> None:
        super().__init__()
