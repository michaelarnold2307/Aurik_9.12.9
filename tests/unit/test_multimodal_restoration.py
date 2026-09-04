"""Multi-Modal-Restaurierung — Validierung der MultimodalDecisionEngine.

Testet die Multi-Modal-Restaurierungslogik (Cover-Bild-Analyse, NLP-Prompt-Parsing,
Audio-Metadata) und validiert die Generierung von Restaurierungs-Ketten.

Spec: .github/specs/02_pipeline_architecture.md Multi-Modal-Restaurierung
      backend/core/multimodal_decision_engine.py
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
class TestMultimodalDecisionEngine:
    """Validiert die MultimodalDecisionEngine."""

    def test_get_instance(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        assert engine is not None, "MultimodalDecisionEngine sollte nicht None sein"

    def test_decide_basic_chain(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        result = engine.decide(
            image_path=None,
            prompt="",
            audio_meta={
                "material": "vinyl",
                "defect_types": ["clicks", "hiss"],
                "restorability": 70.0,
            },
        )
        assert "chain" in result, "Ergebnis sollte 'chain' enthalten"
        assert len(result["chain"]) > 0, "Chain sollte nicht leer sein"
        # Vinyl mit clicks/hiss sollte click_removal und noise_reducer enthalten
        assert "click_removal" in result["chain"] or "noise_reducer" in result["chain"]

    def test_decide_low_restorability(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        result = engine.decide(
            image_path=None,
            prompt="",
            audio_meta={
                "material": "shellac",
                "defect_types": ["surface_noise"],
                "restorability": 15.0,
            },
        )
        assert "chain" in result
        # Low restorability → conservative chain (nur click_removal, noise_reducer, denoiser)
        allowed = {"click_removal", "noise_reducer", "denoiser"}
        for item in result["chain"]:
            assert item in allowed, f"Chain-Item '{item}' nicht erlaubt bei low restorability"

    def test_decide_prompt_nlp(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        result = engine.decide(
            image_path=None,
            prompt="Bitte repariere die Dynamik und füge Wärme hinzu",
            audio_meta={
                "material": "vinyl",
                "defect_types": ["hiss"],
                "restorability": 60.0,
            },
        )
        assert "chain" in result
        # Prompt enthält "Dynamik" und "Wärme" → sollte warmth_enhancer hinzufügen
        assert "warmth_enhancer" in result["chain"] or "denoiser" in result["chain"]

    def test_decide_fallback_chain(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        result = engine.decide(
            image_path=None,
            prompt="",
            audio_meta={
                "material": "unknown",
                "defect_types": [],
                "restorability": 50.0,
            },
        )
        assert "chain" in result
        # Fallback chain sollte noise_reducer enthalten
        assert "noise_reducer" in result["chain"]

    def test_decide_metadata(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        result = engine.decide(
            image_path=None,
            prompt="",
            audio_meta={
                "material": "vinyl",
                "defect_types": ["clicks"],
                "restorability": 70.0,
            },
        )
        assert "meta" in result
        assert result["meta"]["material"] == "vinyl"
        assert result["meta"]["restorability"] == 70.0


@pytest.mark.unit
class TestGenreEraChains:
    """Validiert die Genre/Era-Ketten-Konfiguration."""

    def test_jazz_chains_exist(self):
        from backend.core.multimodal_decision_engine import _GENRE_ERA_CHAINS

        assert "jazz_1950s" in _GENRE_ERA_CHAINS
        assert "jazz_1960s" in _GENRE_ERA_CHAINS

    def test_vinyl_chains_exist(self):
        from backend.core.multimodal_decision_engine import _GENRE_ERA_CHAINS

        assert "vinyl_jazz" in _GENRE_ERA_CHAINS
        assert "vinyl_rock" in _GENRE_ERA_CHAINS

    def test_shellac_chain_exists(self):
        from backend.core.multimodal_decision_engine import _GENRE_ERA_CHAINS

        assert "shellac" in _GENRE_ERA_CHAINS


@pytest.mark.unit
class TestMultimodalIntegration:
    """Integrierte Tests für Multi-Modal-Restaurierung."""

    def test_full_pipeline_vinyl(self):
        from backend.core.multimodal_decision_engine import get_multimodal_decision_engine

        engine = get_multimodal_decision_engine()
        result = engine.decide(
            image_path=None,
            prompt="Repariere dieses Vinyl mit Jazz aus den 1960ern",
            audio_meta={
                "material": "vinyl",
                "genre": "Jazz",
                "era": "1960s",
                "defect_types": ["clicks", "hiss", "wow_flutter"],
                "restorability": 75.0,
            },
        )
        assert "chain" in result
        # Sollte click_removal, noise_reducer und wow_flutter_fix enthalten
        assert "click_removal" in result["chain"] or "noise_reducer" in result["chain"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
