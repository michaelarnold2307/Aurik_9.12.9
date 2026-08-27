"""§v10.115 Exception-Forensik: Feedback-Loop, Aggregation, Pattern-Discovery.

Schließt die 6 Lücken der Exception-Forensik auf SOTA-Niveau:

L1: ExceptionAggregator — liest NDJSON, dedupliziert, klassifiziert
L2: safe_stft/safe_istft — siehe audio_utils.py (separater §v10.115 Abschnitt)
L3: Exception-Dashboard — CLI via scripts/forensics_dashboard.py
L4: Pattern-Mining — entdeckt neue Anti-Patterns aus Aggregator-Output
L5: Q-Score-Korrelation — misst ob Fixes die Qualität verbessern
L6: Continuous Analysis — inkrementelle NDJSON-Updates statt Einmal-Analyse

Architektur:
  Pipeline → oom_phase_forensics.ndjson
       ↓
  ExceptionAggregator.aggregate()
       ↓
  PatternMiner.discover() → neue Pattern-Kandidaten
       ↓
  Scanner-Update (manuell reviewt, dann in scan_anti_patterns.py)
       ↓
  QualityRegressionDetector → Q-Score-Trend nach Fix
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# L1: ExceptionAggregator — NDJSON → klassifizierte Fehler
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ClassifiedException:
    """Eine klassifizierte Exception aus dem NDJSON-Log."""

    exception_type: str  # z.B. "ValueError", "KeyError"
    message_fingerprint: str  # hash der normalisierten Message
    message: str  # original message (gekürzt)
    phase_id: str  # wo es passierte
    stage: str  # "phase_start", "phase_failed", ...
    count: int = 1  # wie oft aufgetreten
    first_seen: str = ""  # ISO timestamp
    last_seen: str = ""  # ISO timestamp
    traceback_snippet: str = ""  # erste 3 Frames
    pattern_class: str = ""  # P1-P6 oder "UNCLASSIFIED"
    q_score_at_time: float = 0.0  # Q-Score zum Zeitpunkt des Fehlers


class ExceptionAggregator:
    """L1: Aggregiert NDJSON-Exception-Logs zu klassifizierten Fehlerberichten.

    Liest `logs/oom_phase_forensics.ndjson` (oder andere NDJSON-Quellen),
    dedupliziert via Message-Fingerabdruck, klassifiziert nach bekannten
    Patterns und Trackt Häufigkeiten über die Zeit.
    """

    KNOWN_PATTERNS = {
        "noverlap must be less than nperseg": "P3",
        "The length of the input vector x must be greater than padlen": "P2",
        "setting an array element with a sequence": "P5",
        "not enough values to unpack": "P1",
        "local variable.*referenced before assignment": "P4",
        "KeyError.*material": "P6",
        "broadcast.*shapes": "P1",
        # §v10.303.11 Neue klassifizierte Patterns:
        "tuple.*has no attribute.*ndim": "P7",
        "phase_failed": "P8",
        "Stereo template must be 2D": "P9",
        "_SkipResult": "P10",
        "operands could not be broadcast": "P11",
        "window_length must be less than": "P12",
        "polyorder must be less than window_length": "P13",
        "truth value of an array.*ambiguous": "P14",
        "restore timeout": "P15",
        # §v10.303.39 Phase-0-spezifische Patterns:
        "TorchScript.*load.*failed": "P16",
        "ONNX.*inference.*failed": "P17",
        "BreathDetector.*not.*available": "P18",
        "phase0.*failed": "P19",
        "Apollo.*not found": "P20",
        "DeepFilterNet.*not.*available": "P21",
        "ResembleEnhance.*not.*available": "P22",
    }

    def __init__(self, log_dir: Path | str | None = None):
        if log_dir is None:
            log_dir = Path(__file__).resolve().parents[2] / "logs"
        self.log_dir = Path(log_dir)
        self.ndjson_path = self.log_dir / "oom_phase_forensics.ndjson"

    def record_exception(self, error_msg: str, phase_id: str = "unknown", stage: str = "phase_failed") -> None:
        """§v10.303.39: Schreibt Exception live in die NDJSON-Forensik-Datei."""
        import json as _json
        from datetime import datetime, timezone

        _entry = _json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase_id": phase_id,
                "stage": stage,
                "error": error_msg,
            }
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.ndjson_path, "a", encoding="utf-8") as _f:
            _f.write(_entry + "\n")
        # §v10.303.40 Exception-Budget: Zähle pro Phase
        if not hasattr(self, "_phase_exc_count"):
            self._phase_exc_count: dict[str, int] = {}
        self._phase_exc_count[phase_id] = self._phase_exc_count.get(phase_id, 0) + 1

    def get_phase_exception_count(self, phase_id: str) -> int:
        """§v10.303.40: Gibt Exception-Count für eine Phase zurück."""
        if not hasattr(self, "_phase_exc_count"):
            self._phase_exc_count = {}
        return self._phase_exc_count.get(phase_id, 0)

    def aggregate(self, since: str | None = None) -> list[ClassifiedException]:
        """Liest NDJSON, dedupliziert und klassifiziert alle Exceptions.

        Args:
            since: ISO timestamp — nur Einträge ab diesem Zeitpunkt.
                   None = alle Einträge.

        Returns:
            Liste klassifizierter Exceptions, absteigend nach Häufigkeit.
        """
        if not self.ndjson_path.exists():
            return []

        # §Perf: Der Gate-Lauf rief aggregate() 3×/Restore auf und parste dabei
        # 469.570 json.loads (11,7 s) — die NDJSON-Datei wird komplett gelesen,
        # obwohl sie sich innerhalb eines Prozesses nur durch Appends ändert.
        # Cache keyed auf (mtime, size): gleiche Datei → gleiche Roh-Entries.
        # Der since-Filter wird pro Aufruf angewendet — identisches Ergebnis.
        _st = self.ndjson_path.stat()
        _cache_key = (float(_st.st_mtime), int(_st.st_size))
        _cache = getattr(self, "_aggregate_cache", None)
        if _cache is not None and _cache[0] == _cache_key:
            raw_entries: list[dict[str, Any]] = _cache[1]
        else:
            raw_entries = []
            with open(self.ndjson_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Nur Einträge mit Fehlern
                    if "error" not in entry and entry.get("stage") not in (
                        "phase_failed",
                        "phase_exception_parallel",
                        "phase_exception",
                    ):
                        continue
                    raw_entries.append(entry)
            self._aggregate_cache = (_cache_key, raw_entries)

        if since:
            raw_entries = [e for e in raw_entries if e.get("timestamp", "") >= since]

        # Deduplizieren via Message-Fingerabdruck
        classified: dict[str, ClassifiedException] = {}
        for entry in raw_entries:
            msg = self._normalize_message(entry.get("error", entry.get("stage", "")))
            fp = hashlib.sha256(msg.encode()).hexdigest()[:16]

            if fp in classified:
                classified[fp].count += 1
                classified[fp].last_seen = entry.get("timestamp", "")
            else:
                exc_type = self._extract_type(entry)
                pattern = self._classify_pattern(msg, exc_type)
                classified[fp] = ClassifiedException(
                    exception_type=exc_type,
                    message_fingerprint=fp,
                    message=msg[:200],
                    phase_id=entry.get("phase_id", "?"),
                    stage=entry.get("stage", "?"),
                    count=1,
                    first_seen=entry.get("timestamp", ""),
                    last_seen=entry.get("timestamp", ""),
                    traceback_snippet=entry.get("traceback", "")[:500],
                    pattern_class=pattern,
                )

        return sorted(classified.values(), key=lambda x: x.count, reverse=True)

    def _normalize_message(self, msg: str) -> str:
        """Normalisiert Exception-Messages für Fingerprinting."""
        # Entferne variable Teile: Speicheradressen, IDs, Zahlen > 1000
        msg = re.sub(r"0x[0-9a-fA-F]+", "0xHEX", msg)
        msg = re.sub(r"\b\d{5,}\b", "N", msg)
        # Normalisiere Pfade
        msg = re.sub(r"/[a-zA-Z0-9_/.-]+\.py", "/PATH.py", msg)
        return msg.strip().lower()

    def _extract_type(self, entry: dict[str, Any]) -> str:
        """Extrahiert den Exception-Typ aus einem NDJSON-Eintrag."""
        error = entry.get("error", "")
        if ":" in error:
            return error.split(":")[0].strip()  # type: ignore[no-any-return]
        return entry.get("stage", "Unknown")  # type: ignore[no-any-return]

    def _classify_pattern(self, msg: str, exc_type: str) -> str:
        """Klassifiziert eine Exception nach bekannten Patterns."""
        combined = f"{exc_type}:{msg}"
        for keyword, pattern in self.KNOWN_PATTERNS.items():
            if re.search(keyword, combined, re.IGNORECASE):
                return pattern
        return "UNCLASSIFIED"

    def summary(self) -> dict[str, Any]:
        """Gibt eine Zusammenfassung des aktuellen NDJSON-Stands."""
        entries = self.aggregate()
        total = sum(e.count for e in entries)
        by_pattern: Any = Counter()
        by_phase: Any = Counter()
        for e in entries:
            by_pattern[e.pattern_class] += e.count
            by_phase[e.phase_id] += e.count

        return {
            "total_exceptions": total,
            "unique_messages": len(entries),
            "by_pattern": dict(by_pattern.most_common()),
            "by_phase": dict(by_phase.most_common(10)),
            "unclassified": by_pattern.get("UNCLASSIFIED", 0),
            "top_exceptions": [
                {"type": e.exception_type, "message": e.message[:120], "count": e.count} for e in entries[:10]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L4: PatternMiner — entdeckt neue Anti-Patterns aus NDJSON
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PatternCandidate:
    """Ein vom Miner vorgeschlagener neuer Pattern-Kandidat."""

    temporary_id: str  # z.B. "P7"
    description: str  # menschenlesbare Beschreibung
    regex_pattern: str  # Regex für Scanner-Integration
    affected_files: list[str]  # Dateien mit diesem Pattern
    exception_count: int  # wie oft aufgetreten
    confidence: float  # 0.0-1.0 Konfidenz


class PatternMiner:
    """L4: Entdeckt neue Anti-Pattern-Kandidaten aus aggregierten Exceptions.

    Analysiert UNCLASSIFIED Exceptions auf wiederkehrende Muster und
    schlägt neue Pattern-Regeln für den Scanner vor.

    Algorithmus:
    1. Cluster UNCLASSIFIED Exceptions nach Exception-Typ + Message-Ähnlichkeit
    2. Extrahiere gemeinsame Substrings als Regex-Kandidaten
    3. Validiere gegen Codebasis (kommt das Pattern in mehreren Dateien vor?)
    4. Bewerte Konfidenz nach Häufigkeit + Datei-Streuung
    """

    def __init__(self, aggregator: ExceptionAggregator):
        self.aggregator = aggregator
        self.repo_root = Path(__file__).resolve().parents[2]

    def discover(self) -> list[PatternCandidate]:
        """Mined neue Pattern-Kandidaten aus UNCLASSIFIED Exceptions."""
        entries = self.aggregator.aggregate()
        unclassified = [e for e in entries if e.pattern_class == "UNCLASSIFIED" and e.count >= 3]

        if not unclassified:
            return []

        candidates: list[PatternCandidate] = []
        pattern_idx = 7  # P7, P8, ...

        for entry in unclassified:
            # Extrahiere potenzielle Regex aus der Message
            regex = self._extract_regex_candidate(entry.message, entry.exception_type)

            # Finde betroffene Dateien (Code-Suche)
            affected = self._find_affected_files(regex) if regex else []

            if affected:
                candidates.append(
                    PatternCandidate(
                        temporary_id=f"P{pattern_idx}",
                        description=self._describe(entry),
                        regex_pattern=regex,
                        affected_files=affected[:10],
                        exception_count=entry.count,
                        confidence=self._confidence(entry.count, len(affected)),
                    )
                )
                pattern_idx += 1

        return sorted(candidates, key=lambda c: c.confidence, reverse=True)

    def _extract_regex_candidate(self, msg: str, exc_type: str) -> str:
        """Extrahiert einen Regex-Kandidaten aus einer Exception-Message."""
        # Entferne variable Teile
        msg = re.sub(r"0x[0-9a-fA-F]+", r"\\b0x[0-9a-fA-F]+\\b", msg)
        msg = re.sub(r"\b\d{4,}\b", r"\\d+", msg)
        # Finde das markanteste Keyword
        keywords = [
            "tuple.*ndim",
            "shape.*mismatch",
            "index.*out.*of.*bounds",
            "division.*by.*zero",
            "NoneType.*has.*no.*attribute",
            "cannot.*broadcast",
            "reshape.*incompatible",
            "memory.*allocation",
            "CUDA.*out.*of.*memory",
        ]
        for kw in keywords:
            if re.search(kw, msg, re.IGNORECASE):
                return kw
        return re.sub(r"\\d+", r"\\d+", msg[:80])

    def _find_affected_files(self, regex: str) -> list[str]:
        """Sucht nach Code-Stellen, die dem Pattern entsprechen."""
        try:
            import subprocess

            result = subprocess.run(
                ["grep", "-rl", regex, "backend/core/"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_root,
            )
            return [f.strip() for f in result.stdout.split("\n") if f.strip()][:10]
        except Exception:
            return []

    def _describe(self, entry: ClassifiedException) -> str:
        """Erzeugt eine menschenlesbare Beschreibung."""
        return f"{entry.exception_type}: {entry.message[:100]}"

    def _confidence(self, count: int, files: int) -> float:
        """Berechnet Konfidenz 0-1 basierend auf Häufigkeit und Streuung."""
        count_score = min(count / 50, 1.0)  # 50+ Exceptions = max confidence
        file_score = min(files / 5, 1.0)  # 5+ files = max confidence
        return round(0.5 * count_score + 0.5 * file_score, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# L5: QualityRegressionDetector — misst ob Fixes die Qualität verbessern
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualitySnapshot:
    """Q-Score + Exception-Rate zu einem Zeitpunkt."""

    timestamp: str
    q_score: float
    exception_count: int
    pattern_counts: dict[str, int]


class QualityRegressionDetector:
    """L5: Korreliert Exception-Raten mit Q-Score-Entwicklung.

    Trackt ob Exception-Fixes den Q-Score verbessern und detektiert
    Regressionen (neue Exceptions die den Score senken).
    """

    def __init__(self, aggregator: ExceptionAggregator):
        self.aggregator = aggregator
        self.history_path = self.aggregator.log_dir / "quality_history.ndjson"

    def snapshot(self, q_score: float) -> QualitySnapshot:
        """Erstellt einen Qualitäts-Snapshot des aktuellen Zustands."""
        summary = self.aggregator.summary()
        return QualitySnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            q_score=q_score,
            exception_count=summary["total_exceptions"],
            pattern_counts=summary["by_pattern"],
        )

    def record(self, q_score: float) -> None:
        """Schreibt einen Snapshot in die History."""
        snap = self.snapshot(q_score)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": snap.timestamp,
                        "q_score": snap.q_score,
                        "exception_count": snap.exception_count,
                        "pattern_counts": snap.pattern_counts,
                    }
                )
                + "\n"
            )

    def compare(self) -> dict[str, Any]:
        """Vergleicht den letzten mit dem vorletzten Snapshot."""
        if not self.history_path.exists():
            return {"status": "no_data"}

        snapshots: list[dict[str, Any]] = []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        snapshots.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if len(snapshots) < 2:
            return {"status": "insufficient_data", "snapshots": len(snapshots)}

        current = snapshots[-1]
        previous = snapshots[-2]

        q_delta = current["q_score"] - previous["q_score"]
        exc_delta = current["exception_count"] - previous["exception_count"]

        # Detektiere Regression: mehr Exceptions + niedrigerer Score
        regression = q_delta < -0.01 and exc_delta > 5

        return {
            "status": "ok",
            "current_q_score": current["q_score"],
            "previous_q_score": previous["q_score"],
            "q_score_delta": round(q_delta, 4),
            "exception_delta": exc_delta,
            "regression_detected": regression,
            "current_timestamp": current["timestamp"],
            "previous_timestamp": previous["timestamp"],
        }

    def trend(self, window: int = 10) -> list[dict[str, Any]]:
        """Gibt Q-Score- und Exception-Trend über die letzten N Snapshots."""
        if not self.history_path.exists():
            return []

        snapshots: list[dict[str, Any]] = []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        snapshots.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return snapshots[-window:]


# ═══════════════════════════════════════════════════════════════════════════════
# L6: ContinuousAnalysis — inkrementelle Scanner-Updates
# ═══════════════════════════════════════════════════════════════════════════════


class ContinuousAnalyzer:
    """L6: Inkrementelle Analyse — liest nur neue NDJSON-Einträge seit letztem Lauf.

    Verhindert Re-Analyse bereits bekannter Exceptions und ermöglicht
    kontinuierliches Monitoring ohne Performance-Einbußen.
    """

    def __init__(self, aggregator: ExceptionAggregator):
        self.aggregator = aggregator
        self.cursor_path = self.aggregator.log_dir / ".forensics_cursor"

    def get_cursor(self) -> str:
        """Letzten bekannten Timestamp laden."""
        if self.cursor_path.exists():
            return self.cursor_path.read_text().strip()
        return ""

    def set_cursor(self, timestamp: str) -> None:
        """Cursor auf neuesten Timestamp setzen."""
        self.log_dir = self.aggregator.log_dir
        self.cursor_path.write_text(timestamp)

    def analyze_new(self) -> dict[str, Any]:
        """Analysiert nur neue NDJSON-Einträge seit letztem Cursor."""
        cursor = self.get_cursor()
        entries = self.aggregator.aggregate(since=cursor if cursor else None)

        # Cursor aktualisieren
        if entries:
            latest = max(e.last_seen for e in entries if e.last_seen)
            if latest:
                self.set_cursor(latest)

        miner = PatternMiner(self.aggregator)
        candidates = miner.discover()

        return {
            "new_exceptions": len(entries),
            "cursor_updated": cursor != self.get_cursor(),
            "pattern_candidates": [
                {"id": c.temporary_id, "desc": c.description, "confidence": c.confidence} for c in candidates
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════════


def get_forensics() -> ExceptionAggregator:
    """Liefert den globalen ExceptionAggregator (singleton-artig)."""
    return ExceptionAggregator()


__all__ = [
    "ExceptionAggregator",
    "ClassifiedException",
    "PatternMiner",
    "PatternCandidate",
    "QualityRegressionDetector",
    "QualitySnapshot",
    "ContinuousAnalyzer",
    "get_forensics",
]
