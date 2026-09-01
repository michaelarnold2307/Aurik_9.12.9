#!/usr/bin/env python3
"""
Autonomous Real-Audio Validation with Material-Uncertainty Watchdog
Tests Elke Best samples with Restoration mode to verify:
- separation_fidelity improvement over 0.7232 baseline
- Material-Uncertainty Watchdog behavior on real vocal music
- Musical goals measurement
- Wall-time performance
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add workspace to path
WORKSPACE_ROOT = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_ROOT))

def load_audio_safe(filepath: str) -> tuple[bytes, int] | None:
    """Load audio file safely."""
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None

def run_restoration_with_monitoring(audio: bytes, sr: int, material_hint: str = "unknown") -> dict:
    """Run restoration with Material-Uncertainty Watchdog monitoring."""
    import numpy as np
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3
    from backend.core.pre_analysis import run_pre_analysis
    
    result = {
        "status": "UNKNOWN",
        "material_hint": material_hint,
        "material_confidence": 0.0,
        "global_scalar": 1.0,
        "watchdog_triggered": False,
        "separation_fidelity": 0.0,
        "musical_goals": {},
        "wall_time_s": 0.0,
    }
    
    try:
        import time
        start = time.time()
        
        # Step 1: Pre-analysis with Material-Uncertainty Watchdog
        logger.info("🔍 Running pre-analysis...")
        pre_result = run_pre_analysis(audio, sr, file_path="real_audio_validation")
        
        if pre_result and pre_result.medium:
            material_conf = float(getattr(pre_result.medium, "confidence", 0.0) or 0.0)
            result["material_confidence"] = material_conf
            
            if material_conf < 0.30:
                result["watchdog_triggered"] = True
                logger.warning(f"⚠️  §v10.712.5 Watchdog triggered: confidence={material_conf:.2f} < 0.30")
        
        # Step 2: Restoration (Restoration Mode, not Studio 2026)
        logger.info("🎵 Running restoration pipeline...")
        restorer = UnifiedRestorerV3()
        restored = restorer.restore(
            audio=audio,
            sr=sr,
            mode="restoration",  # Conservative mode
        )
        
        # Step 3: Measure musical goals
        logger.info("📊 Measuring musical goals...")
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
        from backend.core.ml.batch_processor import ChunkedHTDemuxProcessor
        
        goals_checker = MusicalGoalsChecker()
        # Use restored as reference for comparison
        goals = goals_checker.measure_all(
            audio=restored,
            sr=sr,
            reference=audio,  # Original for comparison
        )
        
        if goals:
            result["musical_goals"] = goals
            if "separation_fidelity" in goals:
                result["separation_fidelity"] = float(goals["separation_fidelity"])
        
        result["status"] = "SUCCESS"
        result["wall_time_s"] = time.time() - start
        
        logger.info(f"✅ Restoration complete: {result['wall_time_s']:.2f}s")
        logger.info(f"   separation_fidelity: {result['separation_fidelity']:.4f}")
        logger.info(f"   watchdog_triggered: {result['watchdog_triggered']}")
        
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        logger.error(f"❌ Restoration failed: {e}", exc_info=True)
    
    return result

def validate_against_baseline(result: dict, baseline: float = 0.7232) -> dict:
    """Compare result against known baseline."""
    sep_fid = result.get("separation_fidelity", 0.0)
    delta = sep_fid - baseline
    improvement_pct = (delta / baseline * 100) if baseline > 0 else 0.0
    
    return {
        "baseline": baseline,
        "measured": sep_fid,
        "delta": delta,
        "improvement_pct": improvement_pct,
        "meets_target": sep_fid >= 0.80,
        "status": "✅ EXCELLENT" if sep_fid >= 0.85 else "✅ GOOD" if sep_fid >= 0.80 else "⚠️  FAIR",
    }

def main():
    """Run comprehensive real-audio validation."""
    logger.info("=" * 80)
    logger.info("🎼 AUTONOMOUS REAL-AUDIO VALIDATION WITH MATERIAL-UNCERTAINTY WATCHDOG")
    logger.info("=" * 80)
    
    samples = [
        ("test_audio/_elke_60s_excerpt.wav", "Elke Best (60s excerpt)"),
        ("test_audio/Elke Best - 30 Sekunden.mp3", "Elke Best (30s)"),
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "watchdog_enabled": True,
        "baseline_separation_fidelity": 0.7232,
        "samples": {}
    }
    
    for filepath, label in samples:
        full_path = WORKSPACE_ROOT / filepath
        if not full_path.exists():
            logger.warning(f"⏭️  Skipping {label} — file not found")
            continue
        
        logger.info(f"\n📁 Processing: {label}")
        logger.info(f"   Path: {full_path}")
        
        # Load audio
        audio_result = load_audio_safe(str(full_path))
        if not audio_result:
            continue
        
        audio, sr = audio_result
        logger.info(f"   Loaded: {audio.shape} @ {sr}Hz")
        
        # Run restoration
        restoration = run_restoration_with_monitoring(audio, sr, material_hint="vocal")
        
        # Validate against baseline
        validation = validate_against_baseline(restoration)
        
        results["samples"][label] = {
            "restoration": restoration,
            "validation": validation,
        }
        
        # Print summary
        logger.info(f"\n   📊 VALIDATION RESULTS:")
        logger.info(f"      Baseline:     {validation['baseline']:.4f}")
        logger.info(f"      Measured:     {validation['measured']:.4f}")
        logger.info(f"      Delta:        {validation['delta']:+.4f} ({validation['improvement_pct']:+.1f}%)")
        logger.info(f"      Status:       {validation['status']}")
    
    # Save results
    output_path = WORKSPACE_ROOT / "reports/real_audio_validation_elke_best.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    all_excellent = all(
        v["validation"].get("status", "").startswith("✅ EXCELLENT")
        for v in results["samples"].values()
    )
    all_good = all(
        v["validation"].get("status", "").startswith("✅")
        for v in results["samples"].values()
    )
    
    if all_excellent:
        logger.info("🏆 RESULT: EXCELLENT — All samples exceeded 0.85 separation_fidelity")
    elif all_good:
        logger.info("✅ RESULT: GOOD — All samples met or exceeded 0.80 separation_fidelity")
    else:
        logger.info("⚠️  RESULT: FAIR — Some samples below 0.80 threshold")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
