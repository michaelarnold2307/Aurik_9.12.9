"""
backend/core/phase_fingerprint.py — PhaseFingerprint (§v10.10)
===============================================================

Hash aller Phase-Parameter für inkrementelles Re-Processing.
Bei Re-Run werden nur Phasen mit geändertem Fingerprint + abhängige neu berechnet.

Synergie Preset × Selbstkalibrierung:
    Preset-Änderung = Fingerprint-Änderung → Phase wird neu berechnet
    Selbstkalibrierung = Strength-Änderung → Fingerprint-Änderung → neu berechnet

Usage:
    from backend.core.phase_fingerprint import PhaseFingerprinter
    fp = PhaseFingerprinter()
    fp.record("phase_03_denoise", strength=0.5, wet_dry=0.8, ...)
    if fp.has_changed("phase_03_denoise", strength=0.6):
        # Phase muss neu berechnet werden
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FINGERPRINT_DIR = Path.home() / ".aurik" / "fingerprints"


class PhaseFingerprinter:
    """Erstellt und vergleicht Fingerprints für inkrementelles Re-Processing."""

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id or ""
        self._fingerprints: dict[str, str] = {}
        if self._run_id:
            self._load()

    def compute(self, phase_id: str, **params: Any) -> str:
        """Berechnet Fingerprint-Hash für eine Phase aus ihren Parametern.

        Args:
            phase_id: Eindeutige Phase-ID.
            **params: Alle relevanten Phase-Parameter.

        Returns:
            SHA256-Hash (erste 16 Zeichen).
        """
        _sorted = json.dumps(params, sort_keys=True, default=str)
        _hash = hashlib.sha256(f"{phase_id}:{_sorted}".encode()).hexdigest()[:16]
        return _hash

    def record(self, phase_id: str, **params: Any) -> None:
        """Speichert Fingerprint für eine ausgeführte Phase."""
        _fp = self.compute(phase_id, **params)
        self._fingerprints[phase_id] = _fp
        if self._run_id:
            self._save()

    def has_changed(self, phase_id: str, **params: Any) -> bool:
        """Prüft ob sich Parameter seit letztem Run geändert haben.

        Returns:
            True wenn Fingerprint anders oder kein vorheriger Eintrag existiert.
        """
        _new_fp = self.compute(phase_id, **params)
        _old_fp = self._fingerprints.get(phase_id)
        return _old_fp is None or _old_fp != _new_fp

    def get_dependent_phases(
        self,
        changed_phase: str,
        all_phases: list[str],
    ) -> list[str]:
        """Gibt alle Phasen zurück die nach changed_phase kommen (= abhängig).

        Args:
            changed_phase: Die geänderte Phase-ID.
            all_phases: Vollständige geordnete Phasenliste.

        Returns:
            Liste von Phasen die neu berechnet werden müssen.
        """
        try:
            _idx = all_phases.index(changed_phase)
            return all_phases[_idx:]
        except ValueError as exc:
            logger.debug("§V6 changed_phase nicht in all_phases gefunden — alle Phasen zurückgegeben (Phase %s): %s", changed_phase, exc)
            return list(all_phases)

    def needs_reprocessing(
        self,
        phase_id: str,
        all_phases: list[str],
        **params: Any,
    ) -> bool:
        """Vollständige Prüfung: Phase geändert ODER abhängig von geänderter Phase.

        Returns:
            True wenn diese Phase (oder eine Vorgänger-Phase) neu berechnet werden muss.
        """
        if self.has_changed(phase_id, **params):
            return True
        # Prüfe ob eine Vorgänger-Phase geändert wurde
        _phase_idx = all_phases.index(phase_id) if phase_id in all_phases else -1
        for _i, _p in enumerate(all_phases):
            if _i >= _phase_idx:
                break
            if self.has_changed(_p):
                return True
        return False

    def _save(self) -> None:
        """Persistiert Fingerprints auf Disk."""
        try:
            _dir = _FINGERPRINT_DIR / self._run_id
            _dir.mkdir(parents=True, exist_ok=True)
            _path = _dir / "fingerprints.json"
            _path.write_text(json.dumps(self._fingerprints, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Fingerprint speichern fehlgeschlagen: %s", exc)

    def _load(self) -> None:
        """Lädt Fingerprints von Disk."""
        try:
            _path = _FINGERPRINT_DIR / self._run_id / "fingerprints.json"
            if _path.exists():
                self._fingerprints = json.loads(_path.read_text(encoding="utf-8"))
        except Exception:
            self._fingerprints = {}
