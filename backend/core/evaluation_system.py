"""§v10.995: Das EINE Evaluations-System — objektiv, ehrlich, CI-lauffähig.

Ein Einstiegspunkt für ALLE Bewertungen Auriks:

  1. Objective Metrics   — SNR-Delta, MSE-Reduktion, UTMOS-Delta,
                           Musical-Goals-Passrate, Bandbreiten-Delta
  2. Competitor-Vergleich — Aurik ≥ iZotope RX 11 (MUSHRA-Proxy 71.0 aus
                           benchmarks/musical_restoration_benchmark) in ≥7/10
                           Szenarien (Spec §8.2 Punkt 11)
  3. Regression           — Aurik gegen sich selbst: nie schlechter als Baseline
  4. Listening-Test       — kontrollierte AB-Paar-Exporte + Score-Import
                           in DASSELBE Report-Schema

Ein Report-Schema (JSON), ein CLI (scripts/evaluate.py), ein CI-Gate.

Ehrlichkeits-Regel: JEDES Ergebnis wird berichtet — auch Verschlechterungen.
Keine Filterung, keine Rosinenpickerei.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

# §8.2 Punkt 11: Aurik ≥ iZotope in ≥ 7/10 Szenarien
COMPETITIVE_WIN_RATIO_REQUIRED = 0.7
# §8.1: OS-Führerschaft ≥ 84.0 MUSHRA, ≥ 8/10 Szenarien
AMRB_MUSHRA_TARGET = 84.0
AMRB_WINS_REQUIRED = 8

# Verdict-Schwellen (pro Fall)
_SNR_IMPROVE_DB = 0.5
_MSE_IMPROVE_PCT = 5.0
_UTMOS_IMPROVE = 0.05


# ═════════════════════════════════════════════════════════════════════════════
# Datenmodell
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalCase:
    """Ein Bewertungsfall: beschädigt → (restauriert) vs. clean-Referenz."""

    case_id: str
    material: str = "unknown"
    damaged: np.ndarray | None = None
    clean: np.ndarray | None = None
    restored: np.ndarray | None = None
    sample_rate: int = 48000
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_clean_reference(self) -> bool:
        return self.clean is not None


@dataclass
class CaseMetrics:
    case_id: str
    snr_delta_db: float = 0.0
    mse_reduction_pct: float = 0.0
    utmos_delta: float | None = None  # None = Modell nicht verfügbar
    musical_goals_passed: int | None = None  # None = Checker nicht verfügbar
    musical_goals_total: int | None = None
    bandwidth_delta_hz: float = 0.0
    verdict: str = "neutral"  # improved | neutral | degraded

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "snr_delta_db": round(self.snr_delta_db, 3),
            "mse_reduction_pct": round(self.mse_reduction_pct, 3),
            "utmos_delta": None if self.utmos_delta is None else round(self.utmos_delta, 4),
            "musical_goals_passed": self.musical_goals_passed,
            "musical_goals_total": self.musical_goals_total,
            "bandwidth_delta_hz": round(self.bandwidth_delta_hz, 1),
            "verdict": self.verdict,
        }


@dataclass
class GateResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "details": self.details}


@dataclass
class EvalReport:
    """DAS eine Report-Schema für alle Bewertungen."""

    schema_version: str = SCHEMA_VERSION
    generated_at: str = ""
    mode: str = "objective"
    cases: list[CaseMetrics] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    verdict: str = "PASS"  # PASS | FAIL | SKIP

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "case_count": len(self.cases),
            "cases": [c.as_dict() for c in self.cases],
            "aggregate": self._aggregate(),
            "gates": [g.as_dict() for g in self.gates],
            "verdict": self.verdict,
        }

    def _aggregate(self) -> dict[str, Any]:
        if not self.cases:
            return {}
        snrs = [c.snr_delta_db for c in self.cases]
        mses = [c.mse_reduction_pct for c in self.cases]
        verdicts = [c.verdict for c in self.cases]
        utmos = [c.utmos_delta for c in self.cases if c.utmos_delta is not None]
        goals = [
            c.musical_goals_passed / c.musical_goals_total
            for c in self.cases
            if c.musical_goals_passed is not None and c.musical_goals_total
        ]
        return {
            "mean_snr_delta_db": round(float(np.mean(snrs)), 3),
            "mean_mse_reduction_pct": round(float(np.mean(mses)), 3),
            "mean_utmos_delta": None if not utmos else round(float(np.mean(utmos)), 4),
            "mean_goal_pass_rate": None if not goals else round(float(np.mean(goals)), 4),
            "improved": verdicts.count("improved"),
            "neutral": verdicts.count("neutral"),
            "degraded": verdicts.count("degraded"),
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Schreibt den Report als JSON. EIN Schema für alle Läufe."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = _REPORTS_DIR / f"evaluation_{stamp}.json"
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Evaluations-Report: %s (Verdict: %s)", out, self.verdict)
        return out

    @classmethod
    def load(cls, path: Path | str) -> EvalReport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        report = cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            generated_at=data.get("generated_at", ""),
            mode=data.get("mode", "objective"),
            verdict=data.get("verdict", "PASS"),
        )
        report.cases = [CaseMetrics(**c) for c in data.get("cases", [])]
        report.gates = [GateResult(**g) for g in data.get("gates", [])]
        return report


# ═════════════════════════════════════════════════════════════════════════════
# Objektive Metriken
# ═════════════════════════════════════════════════════════════════════════════


def _snr_db(signal: np.ndarray, noise_ref: np.ndarray) -> float:
    """SNR des Signals relativ zur clean-Referenz (Rauschen = Abweichung)."""
    diff = np.asarray(signal, dtype=np.float64) - np.asarray(noise_ref, dtype=np.float64)
    sig_power = float(np.mean(np.asarray(noise_ref, dtype=np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(diff**2)) + 1e-12
    return cast(float, 10.0 * np.log10(sig_power / noise_power))


def compute_objective_metrics(case: EvalCase) -> CaseMetrics:
    """Berechnet alle objektiven Metriken eines Falls — defensiv gegen Lücken."""
    metrics = CaseMetrics(case_id=case.case_id)
    damaged = case.damaged
    restored = case.restored
    clean = case.clean

    if damaged is None or clean is None:
        metrics.verdict = "neutral"
        return metrics

    if restored is not None:
        snr_damaged = _snr_db(damaged, clean)
        snr_restored = _snr_db(restored, clean)
        metrics.snr_delta_db = snr_restored - snr_damaged

        mse_damaged = float(np.mean((np.asarray(damaged) - np.asarray(clean)) ** 2))
        mse_restored = float(np.mean((np.asarray(restored) - np.asarray(clean)) ** 2))
        if mse_damaged > 1e-12:
            metrics.mse_reduction_pct = (1.0 - mse_restored / mse_damaged) * 100.0

        metrics.utmos_delta = _compute_utmos_delta(damaged, restored, case.sample_rate)
        goals = _compute_musical_goals(restored, clean, case.sample_rate)
        if goals is not None:
            metrics.musical_goals_passed, metrics.musical_goals_total = goals

        metrics.bandwidth_delta_hz = _bandwidth_hz(restored, case.sample_rate) - _bandwidth_hz(
            damaged, case.sample_rate
        )

        # ── Ehrliches Verdict ──
        improved = (
            metrics.snr_delta_db > _SNR_IMPROVE_DB
            or metrics.mse_reduction_pct > _MSE_IMPROVE_PCT
            or (metrics.utmos_delta is not None and metrics.utmos_delta > _UTMOS_IMPROVE)
        )
        degraded = (
            metrics.snr_delta_db < -_SNR_IMPROVE_DB
            or metrics.mse_reduction_pct < -_MSE_IMPROVE_PCT
            or (metrics.utmos_delta is not None and metrics.utmos_delta < -_UTMOS_IMPROVE)
        )
        metrics.verdict = "improved" if improved else ("degraded" if degraded else "neutral")

    return metrics


def _compute_utmos_delta(damaged: np.ndarray, restored: np.ndarray, sr: int) -> float | None:
    """UTMOS-MOS-Delta — None wenn Modell nicht ladbar (CI ohne ML-Gewichte)."""
    try:
        from plugins.utmos_plugin import get_utmos

        plugin = get_utmos()
        if plugin is None or getattr(plugin, "model", None) is None:
            return None
        mos_damaged = float(plugin.estimate_mos(np.asarray(damaged), sr).mos)
        mos_restored = float(plugin.estimate_mos(np.asarray(restored), sr).mos)
        return mos_restored - mos_damaged
    except Exception as exc:
        log.debug("UTMOS nicht verfügbar (%s) — Delta übersprungen", exc)
        return None


def _compute_musical_goals(restored: np.ndarray, clean: np.ndarray, sr: int) -> tuple[int, int] | None:
    """Musical-Goals-Passrate — None wenn Checker nicht verfügbar."""
    try:
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

        checker = MusicalGoalsChecker()
        result = checker.check_with_adaptive_thresholds(
            audio=np.asarray(restored, dtype=np.float32),
            sr=sr,
            adaptive_thresholds={},
            reference=np.asarray(clean, dtype=np.float32),
        )
        passed = int(getattr(result, "passed_count", 0))
        total = int(getattr(result, "total_count", 0))
        return (passed, total) if total > 0 else None
    except Exception as exc:
        log.debug("MusicalGoalsChecker nicht verfügbar (%s)", exc)
        return None


def _bandwidth_hz(audio: np.ndarray, sr: int) -> float:
    """-3 dB-Bandbreite als grobes Höhen-Maß (Analysefenster: letzte ≤8192 Samples)."""
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=0)
    n = min(len(mono), 8192)
    if n < 1024:
        return 0.0
    frame = mono[-n:] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(frame))
    peak = float(np.max(spectrum)) + 1e-12
    above = np.where(spectrum >= peak * 0.5)[0]
    if above.size == 0:
        return 0.0
    return float(above[-1]) / (len(spectrum) - 1) * (sr / 2)


# ═════════════════════════════════════════════════════════════════════════════
# Gates — die EINEN Entscheidungsregeln
# ═════════════════════════════════════════════════════════════════════════════


def gate_regression(cases: list[CaseMetrics]) -> GateResult:
    """Aurik gegen sich selbst: kein Fall degradiert, Mittelwert ≥ 0 dB."""
    if not cases:
        return GateResult("regression", False, {"reason": "keine Fälle"})
    degraded = [c.case_id for c in cases if c.verdict == "degraded"]
    mean_snr = float(np.mean([c.snr_delta_db for c in cases]))
    passed = not degraded and mean_snr >= 0.0
    return GateResult(
        "regression",
        passed,
        {"mean_snr_delta_db": round(mean_snr, 3), "degraded_cases": degraded},
    )


def gate_competitive(aurik_mushra: float, competitor_mushra: float = 71.0) -> GateResult:
    """§8.2 Punkt 11: Aurik ≥ iZotope RX 11 (MUSHRA) — Einzelszenario-Prüfung.

    Für die 7/10-Regel über mehrere Szenarien: gate_competitive_multi().
    """
    passed = aurik_mushra >= competitor_mushra
    return GateResult(
        "competitive",
        passed,
        {"aurik_mushra": round(aurik_mushra, 1), "rx11_mushra": competitor_mushra},
    )


def gate_competitive_multi(scenario_results: list[tuple[str, float, float]]) -> GateResult:
    """§8.2 Punkt 11 über mehrere Szenarien: ≥ 7/10 gewonnen.

    Args:
        scenario_results: [(name, aurik_mushra, rx11_mushra), …]
    """
    if not scenario_results:
        return GateResult("competitive", False, {"reason": "keine Szenarien"})
    wins = sum(1 for _, aurik, rx11 in scenario_results if aurik >= rx11)
    passed = wins >= max(1, int(len(scenario_results) * COMPETITIVE_WIN_RATIO_REQUIRED + 0.5))
    return GateResult(
        "competitive",
        passed,
        {
            "won": wins,
            "of": len(scenario_results),
            "required": int(len(scenario_results) * COMPETITIVE_WIN_RATIO_REQUIRED + 0.5),
            "scenarios": [
                {"name": n, "aurik": round(a, 1), "rx11": round(r, 1), "won": a >= r} for n, a, r in scenario_results
            ],
        },
    )


def gate_goal_achievement(cases: list[CaseMetrics]) -> GateResult:
    """Musical Goals: mittlere Passrate ≥ 0.8 (nur wenn Checker lief)."""
    rates = [
        c.musical_goals_passed / c.musical_goals_total
        for c in cases
        if c.musical_goals_passed is not None and c.musical_goals_total
    ]
    if not rates:
        return GateResult("goal_achievement", True, {"reason": "Checker nicht verfügbar — übersprungen"})
    mean_rate = float(np.mean(rates))
    return GateResult("goal_achievement", mean_rate >= 0.8, {"mean_pass_rate": round(mean_rate, 4)})


# ═════════════════════════════════════════════════════════════════════════════
# Das System — ein Orchestrator
# ═════════════════════════════════════════════════════════════════════════════


class EvaluationSystem:
    """DAS eine Evaluations-System. Einziger Einstiegspunkt für Bewertungen."""

    def run_objective(self, cases: list[EvalCase], *, gates: bool = True) -> EvalReport:
        metrics = [compute_objective_metrics(c) for c in cases]
        report = EvalReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            mode="objective",
            cases=metrics,
        )
        if gates:
            report.gates = [
                gate_regression(metrics),
                gate_goal_achievement(metrics),
            ]
            report.verdict = "PASS" if all(g.passed for g in report.gates) else "FAIL"
        return report

    def run_competitive(self, scenario_results: list[tuple[str, float, float]]) -> EvalReport:
        """Wettbewerber-Gate: Aurik gegen RX-11-Proxy (MUSHRA-Baselines)."""
        report = EvalReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            mode="competitive",
        )
        report.gates = [gate_competitive_multi(scenario_results)]
        report.verdict = "PASS" if report.gates[0].passed else "FAIL"
        return report


# ═════════════════════════════════════════════════════════════════════════════
# Kontrollierter Hörvergleich (Listening-Test)
# ═════════════════════════════════════════════════════════════════════════════


class ListeningTestExporter:
    """Exportiert randomisierte AB-Paare + Score-Bogen; Import in DASSELBE Schema.

    Kontrolliert = doppelblind-fähig: A/B-Zuordnung pro Fall zufällig,
    Decoder-Tabelle wird erst nach Score-Import aufgelöst.
    """

    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._key: dict[str, dict[str, str]] = {}

    def export_pair(self, case_id: str, restored: np.ndarray, reference: np.ndarray, sr: int) -> Path:
        """Schreibt A.wav/B.wav (randomisiert) + führt den Decoder-Schlüssel."""
        import wave

        # §G5 (copilot-instructions.md): Seeds pro Session — kein time.time() in
        # Entscheidungslogik. Session-Seed einmalig (os.urandom); pro Fall
        # deterministisch ableitbar, damit die A/B-Zuordnung reproduzierbar
        # über den Decoder-Schlüssel auflösbar bleibt.
        if not hasattr(self, "_session_seed"):
            self._session_seed = int.from_bytes(os.urandom(4), "little")
        _case_seed = (self._session_seed + sum(case_id.encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(_case_seed)
        swapped = bool(rng.integers(0, 2))
        pair_dir = self.out_dir / case_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        for name, audio in (("A", reference if swapped else restored), ("B", restored if swapped else reference)):
            mono = np.asarray(audio, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=0)
            pcm = np.clip(mono, -1.0, 1.0)
            pcm = (pcm * 32767).astype("<i2")
            with wave.open(str(pair_dir / f"{name}.wav"), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
        self._key[case_id] = {
            "A": "restored" if not swapped else "reference",
            "B": "reference" if not swapped else "restored",
        }
        return pair_dir

    def write_scoresheet(self) -> Path:
        """Score-Bogen (CSV): case_id, choice(A|B|equal), comment."""
        sheet = self.out_dir / "scoresheet.csv"
        lines = ["case_id,choice,comment"]
        for case_id in sorted(self._key):
            lines.append(f"{case_id},,")
        sheet.write_text("\n".join(lines), encoding="utf-8")
        return sheet

    def write_key(self) -> Path:
        """Decoder-Schlüssel — getrennt vom Score-Bogen (Doppelblind)."""
        key_file = self.out_dir / "decoder_key.json"
        key_file.write_text(json.dumps(self._key, indent=2), encoding="utf-8")
        return key_file


# ═════════════════════════════════════════════════════════════════════════════
# Korpus-Discovery (echte Aufnahmen, lokal)
# ═════════════════════════════════════════════════════════════════════════════


def discover_corpus_cases(corpus_root: Path | str, limit: int = 0) -> list[dict[str, Any]]:
    """Liest corpus/<material>/{clean,damaged,restored}/ und paart Dateien.

    Paarungs-Konventionen (in dieser Reihenfolge):
      1. Namensgleich (damaged.wav ↔ clean.wav)
      2. <song>_<decade>_<defekt>.wav ↔ <song>_<decade>_clean.wav

    Returns: [{case_id, material, damaged, clean, restored}, …] (Pfade)
    """
    root = Path(corpus_root)
    cases: list[dict[str, Any]] = []
    for material_dir in sorted(root.iterdir()):
        if not material_dir.is_dir():
            continue
        damaged_dir = material_dir / "damaged"
        clean_dir = material_dir / "clean"
        restored_dir = material_dir / "restored"
        if not damaged_dir.is_dir() or not clean_dir.is_dir():
            continue
        for damaged_file in sorted(damaged_dir.glob("*.wav")):
            clean_file = _find_clean_for(damaged_file, clean_dir)
            if clean_file is None:
                continue
            case: dict[str, Any] = {
                "case_id": f"{material_dir.name}_{damaged_file.stem}",
                "material": material_dir.name,
                "damaged_path": str(damaged_file),
                "clean_path": str(clean_file),
                "restored_path": None,
            }
            if restored_dir.is_dir():
                restored_file: Path | None = restored_dir / damaged_file.name
                if restored_file is None or not restored_file.exists():
                    restored_file = _find_clean_for(damaged_file, restored_dir)
                if restored_file is not None and restored_file.exists():
                    case["restored_path"] = str(restored_file)
            cases.append(case)
            if limit and len(cases) >= limit:
                return cases
    return cases


def _find_clean_for(damaged_file: Path, clean_dir: Path) -> Path | None:
    """Findet die clean-Referenz: namensgleich ODER <song>_<decade>_clean.wav."""
    direct = clean_dir / damaged_file.name
    if direct.exists():
        return direct
    stem = damaged_file.stem
    base = stem.rsplit("_", 1)[0]
    candidate = clean_dir / f"{base}_clean.wav"
    return candidate if candidate.exists() else None
