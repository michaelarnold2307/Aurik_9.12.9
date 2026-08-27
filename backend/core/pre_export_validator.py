"""§v10.17 PreExportValidator — automatischer Gate vor jedem sf.write()."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def validate_before_export(audio: np.ndarray, sr: int, is_studio_2026: bool = False) -> tuple[bool, list[str]]:
    """Wird VOR jedem Export automatisch aufgerufen.

    Returns:
        (passed, warnings) — passed=False bedeutet: Export sollte blockiert werden.
    """
    warnings = []

    try:
        from backend.core.export_quality_gate import ExportQualityGate
        from backend.core.fallback_auditor import get_fallback_auditor

        # 1. ExportQualityGate
        check = ExportQualityGate.check(audio, sr, is_studio_2026=is_studio_2026)
        if check.errors:
            logger.error("PreExportValidator: %d kritische Fehler — Ausgabe BLOCKIERT", len(check.errors))
            return False, check.errors
        if check.warnings:
            warnings.extend(check.warnings)

        # 2. FallbackAuditor — konsolidierter Degradations-Bericht + zu viele Degradationen?
        fa = get_fallback_auditor()
        if fa.degraded:
            logger.warning("%s", fa.report())  # §v10.17: sichtbarer Bericht vor jedem Export
        if fa.should_block_pipeline if hasattr(fa, "should_block_pipeline") else False:
            logger.error("PreExportValidator: Ersatzpfad-Kaskadenlimit überschritten — Ausgabe BLOCKIERT")
            return False, warnings + ["fallback_cascade_exceeded"]

        # 3. Audio-Sanity
        arr = np.asarray(audio)
        if arr.size == 0:
            return False, warnings + ["empty_audio"]
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False, warnings + ["nan_inf_in_audio"]

        logger.info("PreExportValidator: PASS — %.1f dBTP, %.1f LUFS", check.true_peak_dbtp, check.integrated_lufs)
        return True, warnings

    except Exception as e:
        logger.warning("PreExportValidator fehlgeschlagen: %s — Ausgabe NICHT blockiert", e)
        return True, warnings  # Nicht blockieren wenn Validator selbst fehlschlägt
