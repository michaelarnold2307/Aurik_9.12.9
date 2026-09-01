#!/usr/bin/env python3
"""
Fast Real-Audio Validation — Musical Goals Measurement Only
Tests Elke Best sample without full restoration pipeline
Validates material-adaptive goals measurement
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_ROOT))

def measure_goals_only(audio_path: str) -> dict:
    """Measure musical goals on real audio without restoration."""
    import soundfile as sf
    import numpy as np
    
    logger.info(f"📁 Loading audio: {audio_path}")
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)
        logger.info(f"   ✓ Loaded: {audio.shape} @ {sr}Hz")
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}")
        return {}
    
    try:
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
        
        logger.info("📊 Initializing MusicalGoalsChecker...")
        checker = MusicalGoalsChecker()
        
        logger.info("🎼 Measuring all 15 musical goals...")
        goals = checker.measure_all(audio=audio, sr=sr, reference=audio)
        
        if goals:
            logger.info("✅ Measurement complete:")
            for name, score in sorted(goals.items()):
                if isinstance(score, (int, float)):
                    logger.info(f"   {name:30s}: {score:.4f}")
        
        return goals or {}
    except Exception as e:
        logger.error(f"❌ Measurement failed: {e}", exc_info=True)
        return {}

def main():
    logger.info("=" * 70)
    logger.info("⚡ FAST REAL-AUDIO VALIDATION (Musical Goals Only)")
    logger.info("=" * 70)
    
    samples = [
        "test_audio/_elke_60s_excerpt.wav",
        "test_audio/Elke Best - 30 Sekunden.mp3",
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "validation": "musical_goals_only",
        "baseline": 0.7232,
        "samples": {}
    }
    
    for path in samples:
        full_path = WORKSPACE_ROOT / path
        if not full_path.exists():
            logger.warning(f"⏭️  Skipping {path} — not found")
            continue
        
        label = path.split("/")[-1]
        logger.info(f"\n🎵 {label}")
        
        goals = measure_goals_only(str(full_path))
        results["samples"][label] = goals
        
        if "separation_fidelity" in goals:
            sep_fid = float(goals["separation_fidelity"])
            delta = sep_fid - results["baseline"]
            pct = (delta / results["baseline"] * 100) if results["baseline"] > 0 else 0
            
            logger.info(f"\n   📈 separation_fidelity: {sep_fid:.4f}")
            logger.info(f"      baseline:           {results['baseline']:.4f}")
            logger.info(f"      delta:              {delta:+.4f} ({pct:+.1f}%)")
    
    # Save results
    output_path = WORKSPACE_ROOT / "reports/real_audio_goals_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
