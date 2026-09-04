#!/usr/bin/env python3
"""
Fast Log Parser — Extract actionable warnings from Aurik pipeline
Processes stderr/stdout to identify real issues, not test artifacts
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_ROOT))

def run_sample_and_capture_logs(sample_path: str, sample_label: str, max_duration: float = 30.0):
    """Run one sample with extensive log capture and analysis."""
    import io
    import logging

    import numpy as np
    import soundfile as sf

    # Setup detailed log capture
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)-8s] %(name)s: %(message)s')
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    result = {
        "label": sample_label,
        "logs": [],
        "warnings": [],
        "errors": [],
        "severity_issues": [],  # Issues that affect quality
    }

    try:
        # Load audio
        audio, sr = sf.read(sample_path)
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)

        # Run pre-analysis
        from backend.core.pre_analysis import run_pre_analysis
        pre_result = run_pre_analysis(audio, sr, file_path=sample_path)

        # Get all captured logs
        logs_text = log_capture.getvalue()
        result["logs"] = logs_text.split('\n')

        # Parse warnings and errors
        for line in result["logs"]:
            if line.strip():
                if 'WARNING' in line:
                    result["warnings"].append(line.strip())
                elif 'ERROR' in line:
                    result["errors"].append(line.strip())

                # Identify severity issues
                if any(kw in line.lower() for kw in [
                    'fallback', 'failed', 'error', 'exception',
                    'infinite', 'nan', 'clipping', 'timeout',
                    'shape mismatch', 'out of memory', 'crash'
                ]):
                    result["severity_issues"].append(line.strip())

    except Exception as e:
        result["errors"].append(f"PIPELINE ERROR: {str(e)}")

    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)

    return result

def main():
    """Run one sample as representative test."""
    samples = [
        ("test_audio/_elke_60s_excerpt.wav", "Elke Best 60s"),
        ("test_audio/Elke Best - 30 Sekunden.mp3", "Elke Best 30s"),
        ("test_audio/tape/reel_1940s_dropout.wav", "Reel Tape 1940s"),
        ("test_audio/vinyl/jazz_1950s_scratched.wav", "Vinyl Jazz 1950s"),
    ]

    all_results = {}
    all_issues = defaultdict(int)

    for sample_path, label in samples:
        full_path = WORKSPACE_ROOT / sample_path
        if not full_path.exists():
            print(f"⏭️  Skipping {label} — file not found")
            continue

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━ {label} ━━━━━━━━━━━━━━━━━━━━━━━━━")
        result = run_sample_and_capture_logs(str(full_path), label)
        all_results[label] = result

        # Aggregate issues
        for issue in result["severity_issues"]:
            all_issues[issue] += 1

        # Print summary
        print(f"✓ Warnings: {len(result['warnings'])}")
        print(f"✓ Errors: {len(result['errors'])}")
        print(f"✓ Severity Issues: {len(result['severity_issues'])}")

        if result["severity_issues"]:
            print("  Top issues:")
            for issue in result["severity_issues"][:3]:
                print(f"    - {issue[:80]}")

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 AGGREGATED ISSUES ACROSS SAMPLES")
    print("=" * 80)

    for issue, count in sorted(all_issues.items(), key=lambda x: -x[1])[:15]:
        print(f"  [{count:2d}x] {issue[:70]}")

    # Save report
    output_path = WORKSPACE_ROOT / "reports/log_parser_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize for JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "samples_analyzed": len(all_results),
        "top_issues": [
            {"issue": issue, "count": count}
            for issue, count in sorted(all_issues.items(), key=lambda x: -x[1])[:20]
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Report saved: {output_path}")

if __name__ == "__main__":
    main()
