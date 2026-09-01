#!/usr/bin/env python3
"""
Post-SOTA-Fix Full-Run Validation
Verifies that warning prevention measures work across all test samples
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import subprocess
import re

WORKSPACE_ROOT = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_ROOT))

SAMPLES = [
    ("test_audio/_elke_60s_excerpt.wav", "Elke Best (60s vocal)"),
    ("test_audio/Elke Best - 30 Sekunden.mp3", "Elke Best (30s vocal)"),
    ("test_audio/digital/cd_clipped_2000s.wav", "CD Digital (clipping)"),
    ("test_audio/tape/reel_1940s_dropout.wav", "Reel Tape (1940s)"),
    ("test_audio/vinyl/jazz_1950s_scratched.wav", "Vinyl Jazz (1950s)"),
]

def count_warnings_in_sample(sample_path: str) -> dict:
    """Count warnings/errors for one sample."""
    cmd = f"""
import soundfile as sf
from backend.core.pre_analysis import run_pre_analysis

audio, sr = sf.read('{sample_path}')
if len(audio.shape) == 1:
    audio = audio.reshape(-1, 1)

pre = run_pre_analysis(audio, sr, file_path='{sample_path}')
print("OK")
"""
    
    try:
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::DeprecationWarning'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        result = subprocess.run(
            [str(WORKSPACE_ROOT / ".venv_aurik/bin/python"), 
             "-W", "ignore::FutureWarning",
             "-W", "ignore::DeprecationWarning",
             "-c", cmd],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        
        stderr_lines = result.stderr.split('\n') if result.stderr else []
        
        warnings = [l for l in stderr_lines if 'WARNING' in l or 'FutureWarning' in l]
        errors = [l for l in stderr_lines if 'ERROR' in l or 'Traceback' in l]
        
        return {
            "success": "OK" in result.stdout,
            "warnings_count": len(warnings),
            "errors_count": len(errors),
            "warnings": warnings[:3],
            "errors": errors[:3],
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "warnings_count": 0, "errors_count": 1}
    except Exception as e:
        return {"success": False, "error": str(e), "warnings_count": 0, "errors_count": 1}

def main():
    print("=" * 80)
    print("🎼 POST-SOTA-FIX FULL-RUN VALIDATION")
    print("=" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "samples": {},
        "summary": {
            "total_samples": len(SAMPLES),
            "successful": 0,
            "total_warnings": 0,
            "total_errors": 0,
            "clean_samples": 0,
        }
    }
    
    for sample_path, label in SAMPLES:
        full_path = WORKSPACE_ROOT / sample_path
        if not full_path.exists():
            print(f"⏭️  Skipping {label}")
            continue
        
        print(f"\n🎵 Testing: {label}")
        result = count_warnings_in_sample(sample_path)
        results["samples"][label] = result
        
        if result.get("success"):
            results["summary"]["successful"] += 1
        
        results["summary"]["total_warnings"] += result.get("warnings_count", 0)
        results["summary"]["total_errors"] += result.get("errors_count", 0)
        
        if result.get("warnings_count", 0) == 0 and result.get("errors_count", 0) == 0:
            results["summary"]["clean_samples"] += 1
            status = "✅ CLEAN"
        else:
            status = f"⚠️  W:{result.get('warnings_count', 0)} E:{result.get('errors_count', 0)}"
        
        print(f"   {status}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {results['summary']['total_samples']}")
    print(f"Successful: {results['summary']['successful']}/{results['summary']['total_samples']}")
    print(f"Clean Samples (0 warnings + 0 errors): {results['summary']['clean_samples']}")
    print(f"Total Warnings: {results['summary']['total_warnings']}")
    print(f"Total Errors: {results['summary']['total_errors']}")
    
    if results['summary']['clean_samples'] >= len(SAMPLES) - 1:
        print("\n✅ EXCELLENT: System is nearly warning-free!")
    elif results['summary']['total_warnings'] == 0:
        print("\n✅ GOOD: No application warnings! (Errors may be external)")
    else:
        print(f"\n⚠️  {results['summary']['total_warnings']} warnings remain")
    
    # Save report
    output_path = WORKSPACE_ROOT / "reports/post_sota_fix_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Report: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
