import pytest

pytest.importorskip("pydantic")  # CI-Minimal-Umgebung (cross-platform)

from backend.core.aesthetic_judgment import GenreWeightRegistry


def test_genre_weights_exist():
    GenreWeightRegistry()
    for genre in GenreWeightRegistry.GENRE_WEIGHTS:
        weights = GenreWeightRegistry.GENRE_WEIGHTS[genre]
        assert hasattr(weights, "brilliance")
        assert hasattr(weights, "transparency")
        assert hasattr(weights, "naturalness")
        assert hasattr(weights, "authenticity")
        assert hasattr(weights, "emotionality")
        assert hasattr(weights, "warmth")
        assert hasattr(weights, "spatiality")


@pytest.mark.parametrize("genre", list(GenreWeightRegistry.GENRE_WEIGHTS.keys()))
def test_genre_weight_sum_to_one(genre):
    weights = GenreWeightRegistry.GENRE_WEIGHTS[genre]
    total = (
        weights.brilliance
        + weights.transparency
        + weights.naturalness
        + weights.authenticity
        + weights.emotionality
        + weights.warmth
        + weights.spatiality
    )
    assert abs(total - 1.0) < 1e-6
