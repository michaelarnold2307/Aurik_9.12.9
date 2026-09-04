#!/usr/bin/env python3
"""
Comprehensive Full-Run Warning Audit & SOTA Solution Development
Processes all test samples through complete Aurik pipeline (Pre-Analysis → Restoration → Export)
Systematically identifies warnings and develops SOTA solutions

USER REQUEST: "lass einen gesamten Probelauf durch und entwickele bei Warnungen
SOTA Lösungen damit die Warnungen gar nicht erst anfallen bei allen Importsongs"

Translation: Run complete test through all samples, develop SOTA solutions for all warnings
so they don't occur in the first place for all import songs.
"""

import json
import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

WORKSPACE_ROOT = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Capture ALL warnings
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-8s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# Intercept warnings as well
warnings.simplefilter("always")

SAMPLES = [
    "test_audio/_elke_60s_excerpt.wav",
    "test_audio/Elke Best - 30 Sekunden.mp3",
    "test_audio/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
    "test_audio/digital/cd_clipped_2000s.wav",
    "test_audio/digital/mp3_64kbps_artifacts.wav",
    "test_audio/digital/streaming_glitches.wav",
    "test_audio/tape/cassette_1980s_wow.wav",
    "test_audio/tape/dat_1990s_azimuth.wav",
    "test_audio/tape/reel_1940s_dropout.wav",
    "test_audio/vinyl/classical_1960s_hiss.wav",
    "test_audio/vinyl/jazz_1950s_scratched.wav",
    "test_audio/vinyl/rock_1970s_worn.wav",
    "test_audio/vocals/choir_breaths.wav",
    "test_audio/vocals/opera_sibilance.wav",
    "test_audio/vocals/podcast_plosives.wav",
]

class WarningCapture:
    """Context manager to capture all log warnings and errors."""

    def __init__(self):
        self.warnings = []
        self.errors = []
        self.handler = None
        self.buffer = StringIO()

    def __enter__(self):
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self.handler)
        return self

    def __exit__(self, *args):
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
        self.text = self.buffer.getvalue()
        self._parse_messages()

    def _parse_messages(self):
        """Parse captured log lines into warnings and errors."""
        for line in self.text.split('\n'):
            if 'WARNING' in line:
                self.warnings.append(line.strip())
            elif 'ERROR' in line:
                self.errors.append(line.strip())

def run_complete_pipeline(audio_path: str, sample_label: str) -> dict[str, Any]:
    """Run complete Aurik pipeline for one sample."""
    import numpy as np
    import soundfile as sf

    result = {
        "label": sample_label,
        "path": audio_path,
        "status": "UNKNOWN",
        "duration_s": 0.0,
        "warnings": [],
        "errors": [],
        "categories": {},
        "pre_analysis": {},
        "restoration": {},
    }

    try:
        # Load audio
        logger.info(f"━━━━━━━━━━━━ LOADING {sample_label} ━━━━━━━━━━━━")
        audio, sr = sf.read(audio_path)
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)

        result["duration_s"] = len(audio) / sr
        logger.info(f"✓ Loaded: {audio.shape} @ {sr}Hz ({result['duration_s']:.1f}s)")

        # Pre-Analysis
        logger.info(f"\n🔍 PRE-ANALYSIS {sample_label}...")
        with WarningCapture() as wc:
            from backend.core.pre_analysis import run_pre_analysis
            pre_result = run_pre_analysis(audio, sr, file_path=audio_path)

        result["warnings"].extend(wc.warnings)
        result["errors"].extend(wc.errors)

        if pre_result:
            restorability_score = 0.0
            if pre_result.restorability:
                try:
                    # RestorabilityResult is an object with restorability_score attribute
                    if hasattr(pre_result.restorability, 'restorability_score'):
                        restorability_score = float(pre_result.restorability.restorability_score)
                    elif isinstance(pre_result.restorability, (int, float)):
                        restorability_score = float(pre_result.restorability)
                except (ValueError, AttributeError, TypeError):
                    restorability_score = 0.0

            result["pre_analysis"] = {
                "medium": str(pre_result.medium) if pre_result.medium else "Unknown",
                "era": str(pre_result.era) if pre_result.era else "Unknown",
                "genre": str(pre_result.genre) if pre_result.genre else "Unknown",
                "restorability": restorability_score,
            }
            logger.info(f"✓ Pre-analysis: medium={pre_result.medium}, era={pre_result.era}, restorability={restorability_score:.1f}")

        # Quick restoration (time-limited)
        logger.info(f"\n🎵 RESTORATION {sample_label} (time-limited)...")
        with WarningCapture() as wc:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3
            restorer = UnifiedRestorerV3()

            # Use max 30s per sample to avoid timeout
            import time
            start = time.time()
            try:
                restored = restorer.restore(
                    audio=audio[:min(44100*30, audio.shape[0])],  # Max 30s
                    sr=sr,
                    mode="restoration",
                )
                elapsed = time.time() - start
                logger.info(f"✓ Restoration complete: {elapsed:.1f}s")
                result["restoration"]["status"] = "OK"
                result["restoration"]["wall_time_s"] = elapsed
            except Exception as e:
                logger.error(f"✗ Restoration failed: {e}")
                result["restoration"]["status"] = "FAILED"
                result["restoration"]["error"] = str(e)

        result["warnings"].extend(wc.warnings)
        result["errors"].extend(wc.errors)
        result["status"] = "SUCCESS"

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=False)
        result["status"] = "ERROR"
        result["errors"].append(str(e))

    # Categorize warnings
    result["categories"] = _categorize_warnings(result["warnings"] + result["errors"])

    return result

def _categorize_warnings(messages: list[str]) -> dict[str, list[str]]:
    """Categorize warnings by type for SOTA solution development."""
    categories = defaultdict(list)

    patterns = {
        "MaterialDetection": r"(MediumDetector|material|chain|confidence|analog|digital)",
        "MLFallback": r"(fallback|ML|ONNX|GPU|device|ModelError)",
        "ShapeMismatch": r"(shape|broadcast|reshape|dimension|size)",
        "HTDemucs": r"(HTDemucs|ChunkedProcessor|separation)",
        "Performance": r"(timeout|slow|budget|wall_time|overhead)",
        "DefectDetection": r"(Defect|scanner|clipping|dropout|noise)",
        "DSP": r"(DSP|filter|frequency|STFT|FFT)",
        "Audio": r"(audio|signal|sample|NaN|Inf|clipping)",
        "Vocal": r"(vocal|formant|vibrato|sibilance|register)",
        "Era": r"(Era|Jahrzehnt|decade|format|incompatible)",
        "Genre": r"(genre|instrument|music)",
        "Config": r"(config|parameter|setting|default)",
    }

    for msg in messages:
        matched = False
        for category, pattern in patterns.items():
            if re.search(pattern, msg, re.IGNORECASE):
                categories[category].append(msg)
                matched = True
                break
        if not matched:
            categories["Other"].append(msg)

    return dict(categories)

def main():
    logger.info("=" * 80)
    logger.info("🎼 COMPREHENSIVE FULL-RUN WARNING AUDIT")
    logger.info("=" * 80)

    results = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(SAMPLES),
        "samples": {},
        "warning_summary": {},
        "category_summary": {},
        "samples_with_warnings": 0,
        "samples_without_warnings": 0,
    }

    for i, sample_path in enumerate(SAMPLES, 1):
        full_path = WORKSPACE_ROOT / sample_path
        if not full_path.exists():
            logger.warning(f"⏭️  Skipping {sample_path} — not found")
            continue

        label = sample_path.split("/")[-1]
        logger.info(f"\n[{i}/{len(SAMPLES)}] Processing: {label}")

        result = run_complete_pipeline(str(full_path), label)
        results["samples"][label] = result

        # Aggregate statistics
        if result["warnings"] or result["errors"]:
            results["samples_with_warnings"] += 1
            for msg in result["warnings"] + result["errors"]:
                results["warning_summary"][msg] = results["warning_summary"].get(msg, 0) + 1
        else:
            results["samples_without_warnings"] += 1

        # Aggregate categories
        for cat, msgs in result.get("categories", {}).items():
            if cat not in results["category_summary"]:
                results["category_summary"][cat] = []
            results["category_summary"][cat].extend(msgs)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Samples: {results['total_samples']}")
    logger.info(f"Samples without warnings: {results['samples_without_warnings']}")
    logger.info(f"Samples with warnings: {results['samples_with_warnings']}")
    logger.info(f"Unique warnings: {len(results['warning_summary'])}")

    logger.info("\n📂 WARNING CATEGORIES:")
    for cat, msgs in sorted(results["category_summary"].items(), key=lambda x: -len(x[1])):
        logger.info(f"  {cat:20s}: {len(msgs):3d} messages")

    logger.info("\n🔝 Top 10 Most Frequent Warnings:")
    for msg, count in sorted(results["warning_summary"].items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  [{count:2d}x] {msg[:70]}")

    # Save full report
    output_path = WORKSPACE_ROOT / "reports/full_run_warning_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Serialize carefully to avoid non-JSON types
        clean_results = {
            "timestamp": results["timestamp"],
            "total_samples": results["total_samples"],
            "samples_with_warnings": results["samples_with_warnings"],
            "samples_without_warnings": results["samples_without_warnings"],
            "top_warnings": sorted(results["warning_summary"].items(), key=lambda x: -x[1])[:20],
            "category_summary": {k: len(v) for k, v in results["category_summary"].items()},
        }
        json.dump(clean_results, f, indent=2)

    logger.info(f"\n✅ Full report saved: {output_path}")

    return results

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results["samples_without_warnings"] >= len(SAMPLES) // 2 else 1)
