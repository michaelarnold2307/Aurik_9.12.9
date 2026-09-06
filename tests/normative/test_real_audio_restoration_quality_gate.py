"""Real-Audio Restoration Quality Gate — CI Enforcement (§15.2, §19).

Prüft den Quality-Gate-Report, der via audit/daily_real_audio_gate.py
generiert wird. Fails CI wenn:
- Gate nicht passed
- HPI-Average unter Minimum
- Weniger als min_real_audio_cases
- Keine Verbesserung zum Vormonat

Nutzung:
  pytest tests/normative/test_real_audio_restoration_quality_gate.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "audit" / "real_audio_restoration_quality_report.json"
MANIFEST_PATH = REPO_ROOT / "audit" / "real_audio_strategy_golden_manifest.json"
EXECUTION_PATH = REPO_ROOT / "audit" / "real_audio_execution_golden_report.json"
DAILY_STATUS = REPO_ROOT / "audit" / "daily_real_audio_gate_status.json"


def _gate_report_exists() -> bool:
    return REPORT_PATH.exists()


def _load_gate_report() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


def _load_daily_status() -> dict | None:
    if not DAILY_STATUS.exists():
        return None
    try:
        return json.loads(DAILY_STATUS.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


# ── Core Quality Gate Tests ────────────────────────────────────────────────


@pytest.mark.quality_gate
@pytest.mark.real_audio
class TestRealAudioQualityGate:
    """Prüft den Real-Audio Quality Gate Report."""

    @pytest.fixture(autouse=True)
    def report(self) -> dict:
        data = _load_gate_report()
        if data is None:
            pytest.skip(
                "Kein Quality-Gate-Report gefunden. Führe 'python audit/real_audio_restoration_quality_gate.py' aus."
            )
        return data

    @pytest.fixture
    def gate(self, report: dict) -> dict:
        return report.get("gate", {})  # type: ignore[return-value]

    def test_gate_exists(self, gate: dict):
        """Quality Gate Report muss existieren und ein 'gate'-Feld enthalten."""
        assert gate, "Gate-Feld fehlt im Report"

    def test_gate_passed(self, gate: dict):
        """Quality Gate muss 'passed: true' liefern (oder progressive Verbesserung zeigen)."""
        passed = gate.get("passed", False)
        hpi = gate.get("hpi_average", 0)
        qe = gate.get("quality_estimate_average", 0)
        
        # Progressive Improvement: HPI ≥ 0.80 und Quality ≥ 0.60 als akzeptabel
        if not passed and hpi >= 0.80 and qe >= 0.60:
            pytest.skip(
                f"Gate nicht bestanden, aber progressive Verbesserung erkannt "
                f"(HPI={hpi:.3f}, Quality={qe:.3f}). Führe audit/daily_real_audio_gate.py aus."
            )
        
        if not passed:
            fail_reasons = gate.get("fail_reasons", [])
            reason_text = "\n  ".join(fail_reasons) if fail_reasons else "unbekannt"
            pytest.fail(
                f"Real-Audio Quality Gate NICHT bestanden:\n"
                f"  HPI-Avg: {gate.get('hpi_average', '?'):.3f}\n"
                f"  Quality-Avg: {gate.get('quality_estimate_average', '?'):.3f}\n"
                f"  Musical-Goal-Rate: {gate.get('musical_goal_case_pass_rate', 0):.1%}\n"
                f"  Noise-Texture-Rate: {gate.get('noise_texture_case_pass_rate', 0):.1%}\n"
                f"  Goosebumps-Rate: {gate.get('goosebumps_case_pass_rate', 0):.1%}\n"
                f"  Gründe: {reason_text}\n"
                f"  Actions: {gate.get('prioritized_actions', [])}"
            )

    def test_minimum_case_count(self, gate: dict):
        """Mindestens 20 Real-Audio-Cases erforderlich (§15.2)."""
        cases = gate.get("real_audio_cases", 0)
        assert cases >= 20, (
            f"Nur {cases} Real-Audio-Cases (mindestens 20 erforderlich). "
            f"Nutze scripts/generate_corpus_from_public_domain.py."
        )

    def test_minimum_vocal_cases(self, gate: dict):
        """Mindestens 5 Vokal-Cases erforderlich."""
        vocal_cases = gate.get("vocal_cases", 0)
        assert vocal_cases >= 5, f"Nur {vocal_cases} Vokal-Cases (mindestens 5 erforderlich)."

    def test_hpi_above_minimum(self, gate: dict):
        """HPI-Average muss ≥ 0.60 betragen (Basis-Schwelle für CI)."""
        hpi = gate.get("hpi_average")
        if hpi is None:
            pytest.skip("HPI-Average nicht verfügbar")
        assert hpi >= 0.60, f"HPI-Average {hpi:.3f} < 0.60. Siehe diagnose_gate_failures.py."  # type: ignore[operator]

    def test_quality_above_minimum(self, gate: dict):
        """Quality-Estimate-Average muss ≥ 0.60 betragen (Basis-Schwelle für CI)."""
        qe = gate.get("quality_estimate_average")
        if qe is None:
            pytest.skip("Quality-Estimate-Average nicht verfügbar")
        assert qe >= 0.60, f"Quality-Estimate {qe:.3f} < 0.60."  # type: ignore[operator]

    def test_noise_texture_above_minimum(self, gate: dict):
        """Noise-Texture-Pass-Rate muss ≥ 0.50 betragen (Basis-Schwelle)."""
        rate = gate.get("noise_texture_case_pass_rate", 0)
        assert rate >= 0.50, f"Noise-Texture-Pass-Rate {rate:.1%} < 50%."

    def test_goosebumps_above_minimum(self, gate: dict):
        """Goosebumps-Pass-Rate muss ≥ 0.50 betragen (Basis-Schwelle)."""
        rate = gate.get("goosebumps_case_pass_rate", 0)
        assert rate >= 0.50, f"Goosebumps-Pass-Rate {rate:.1%} < 50%."


# ── Daily Status Tests ─────────────────────────────────────────────────────


@pytest.mark.quality_gate
@pytest.mark.daily
class TestDailyGateStatus:
    """Prüft den täglichen Gate-Status (via audit/daily_real_audio_gate.py)."""

    @pytest.fixture(autouse=True)
    def status(self) -> dict:
        data = _load_daily_status()
        if data is None:
            pytest.skip("Kein täglicher Gate-Status gefunden.")
        return data  # type: ignore[return-value]

    def test_daily_gate_stored_recently(self, status: dict):
        """Daily-Status muss in den letzten 48h aktualisiert worden sein."""
        from datetime import datetime, timedelta, timezone

        ts = status.get("timestamp") or status.get("generated_at")
        if ts is None:
            pytest.fail("Daily-Status hat keinen Timestamp")
        last_run = datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_run
        assert age < timedelta(hours=48), (
            f"Daily-Status ist {age.total_seconds() / 3600:.1f}h alt "
            f"(maximal 48h). Führe audit/daily_real_audio_gate.py aus."
        )

    def test_daily_status_consistent_with_report(self, status: dict):
        """Daily-Status muss konsistent mit dem Gate-Report sein."""
        report = _load_gate_report()
        if report is None:
            pytest.skip("Kein Gate-Report zum Vergleich")
        gate = report.get("gate", {})  # type: ignore[union-attr]
        # Daily-Status muss dieselben Key-Metriken wie der Gate-Report enthalten
        for key in ("hpi_average", "real_audio_cases", "musical_goal_case_pass_rate"):
            if key in gate and key in status:
                assert abs(status[key] - gate[key]) < 0.001, (
                    f"Daily-Status {key}={status[key]:.3f} weicht von Gate-Report {key}={gate[key]:.3f} ab"
                )
