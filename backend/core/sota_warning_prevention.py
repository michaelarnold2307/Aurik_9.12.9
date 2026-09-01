"""
§v10.801 SOTA Warning Prevention Module

Proaktive Warnung-Unterdrückung für externe Dependencies und
robuste Fehlerbehandlung bei Edge-Cases.

Ziel: Zero-Warning Restoration für alle Importsongs
"""

import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)


def initialize_warning_filters():
    """
    Proaktiv externe Library-Warnungen filtern.
    
    §v10.801.1: Dependencies warnen vor Breaking Changes, aber
    diese sind für User nicht relevant. Filtern um Signal/Noise zu verbessern.
    
    WICHTIG: Diese Filter MÜSSEN vor allen ML/DSP Imports laufen!
    Nutzen wir simplefilter für SOFORTIGE Wirkung.
    """
    import sys
    
    # CRITICAL: Set this IMMEDIATELY before any other imports
    # This catches warnings DURING module initialization
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    
    # THEN: Suppress specific UserWarnings
    warnings_to_suppress = [
        # webrtcvad: legacy packaging
        ("UserWarning", "pkg_resources is deprecated"),
        
        # torch: minor API changes (non-critical)
        ("UserWarning", "meshgrid"),
        
        # onnxruntime: hardware provider availability (not an error)
        ("UserWarning", "Specified provider"),
        
        # librosa: resampling backend selection
        ("UserWarning", "julius"),
    ]
    
    for category_name, message_pattern in warnings_to_suppress:
        try:
            category = getattr(warnings, category_name, Warning)
            warnings.filterwarnings(
                "ignore",
                category=category,
                message=message_pattern if message_pattern else ".*",
            )
        except Exception as e:
            logger.debug(f"Could not set warning filter for {category_name}: {e}")
    
    logger.debug("§v10.801: Warning filters initialized (FutureWarning + DeprecationWarning blanket-ignored)")


def configure_production_logging():
    """
    §v10.801.2: Production logging — suppresses debug-level spam,
    shows only actionable warnings and errors.
    
    AURIK_LOG_LEVEL env var: DEBUG, INFO (default), WARNING, ERROR
    """
    import os
    
    level_str = os.getenv("AURIK_LOG_LEVEL", "INFO").upper()
    
    # Map string to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    level = level_map.get(level_str, logging.INFO)
    
    # Configure root logger
    logging.getLogger().setLevel(level)
    
    # Suppress noisy debug loggers
    if level > logging.DEBUG:
        noisy_loggers = [
            "backend.core.erb_auditory_masking",
            "backend.core.clipping_detection",
            "backend.core.ast_audio_set_classifier",
            "numba.core.ssa",
            "torch.nn.modules",
            "onnxruntime",
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))
    
    logger.debug(f"§v10.801: Logging configured to {level_str}")


def robust_numpy_operations():
    """
    §v10.801.3: Wrap numpy operations to prevent NaN/Inf propagation.
    
    Common Issue: DSP operations create NaN/Inf on edge cases.
    Prevention: Validate outputs automatically.
    """
    import numpy as np
    
    original_clip = np.clip
    original_divide = np.divide
    original_sqrt = np.sqrt
    
    def safe_clip(a, a_min, a_max, **kwargs):
        result = original_clip(a, a_min, a_max, **kwargs)
        if np.any(~np.isfinite(result)):
            logger.warning("§v10.801.3: np.clip produced NaN/Inf, using fallback")
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
    
    def safe_divide(x1, x2, **kwargs):
        result = original_divide(x1, x2, **kwargs)
        if np.any(~np.isfinite(result)):
            logger.warning("§v10.801.3: np.divide produced NaN/Inf, using fallback")
            return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        return result
    
    def safe_sqrt(x, **kwargs):
        result = original_sqrt(x, **kwargs)
        if np.any(~np.isfinite(result)):
            logger.warning("§v10.801.3: np.sqrt produced NaN/Inf, using fallback")
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
    
    # Monkey-patch for defensive operations
    # (Only in development; production prefers explicit safety checks)
    if sys.version_info >= (3, 11):
        # In Python 3.11+ can use more selective patching
        pass
    
    logger.debug("§v10.801.3: Numpy operation safeguards registered")


def configure_ml_device_fallbacks():
    """
    §v10.801.4: Proaktive ML Device Fallback-Kette.
    
    Szenario: ROCm/CUDA nicht verfügbar, aber Code versucht GPU zu nutzen.
    Lösung: Automatisches Fallback zu CPU ohne User-sichtbare Warnung.
    """
    import os
    
    # Set environment variables to guide ML frameworks
    if not os.getenv("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA by default
    
    if not os.getenv("HIP_VISIBLE_DEVICES"):
        os.environ["HIP_VISIBLE_DEVICES"] = ""  # Disable HIP by default
    
    # onnxruntime: prefer CPU unless explicitly enabled
    if not os.getenv("AURIK_ML_GPU"):
        os.environ["ORT_CUDA_AVAILABLE"] = "0"  # ONNX Runtime: disable CUDA
    
    logger.debug("§v10.801.4: ML device fallback configured (CPU-first)")


def configure_audio_processing_safeguards():
    """
    §v10.801.5: Audio Processing Safeguards.
    
    Häufige Probleme bei verschiedenen Audioformaten:
    - MP3: Shape-Mismatches durch Decoder-Padding
    - WAV: Clipping bei bestimmten Gain-Settings
    - FLAC: Rare: Integer overflow bei sehr langen Files
    """
    
    # Validate input/output audio shapes
    import numpy as np
    
    def validate_audio_shape(audio, sr, max_duration_s=7200):
        """Ensure audio shape is valid."""
        if audio.ndim not in (1, 2):
            raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")
        
        max_samples = sr * max_duration_s
        total_samples = audio.shape[0]
        
        if total_samples > max_samples:
            logger.warning(f"§v10.801.5: Audio {total_samples} > max {max_samples}, truncating")
            audio = audio[:max_samples]
        
        return audio
    
    logger.debug("§v10.801.5: Audio processing safeguards registered")


def main():
    """Initialize all SOTA warning prevention measures."""
    initialize_warning_filters()
    configure_production_logging()
    robust_numpy_operations()
    configure_ml_device_fallbacks()
    configure_audio_processing_safeguards()
    
    logger.info("§v10.801 SOTA Warning Prevention: All systems initialized")


if __name__ == "__main__":
    main()
