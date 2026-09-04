"""Performance-Guard Cache-Hit-Rate — Validierung der Analyse-Caches.

Testet die LRU-Cache-Infrastruktur (bridge_cache.py) und validiert, dass
die Cache-Hit-Rate innerhalb der erwarteten Grenzen liegt.

Spec: .github/specs/08_software_architecture.md §11 Caching
      backend/api/bridge_cache.py
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
class TestAnalysisLruCache:
    """Validiert die _AnalysisLruCache Infrastruktur."""

    def test_put_get(self):
        from backend.api.bridge_cache import _AnalysisLruCache

        cache = _AnalysisLruCache(maxsize=10)
        cache.put("key1", {"value": 1})
        result = cache.get("key1")
        assert result == {"value": 1}, "Cache sollte den gespeicherten Wert zurückgeben"

    def test_lru_eviction(self):
        from backend.api.bridge_cache import _AnalysisLruCache

        cache = _AnalysisLruCache(maxsize=3)
        # Fülle Cache bis zur Kapazität
        for i in range(5):
            cache.put(f"key{i}", {"value": i})
        # key0 und key1 sollten evicted sein (LRU)
        assert cache.get("key0") is None, "key0 sollte evicted sein"
        assert cache.get("key1") is None, "key1 sollte evicted sein"
        # key2, key3, key4 sollten noch vorhanden sein
        assert cache.get("key2") == {"value": 2}
        assert cache.get("key3") == {"value": 3}
        assert cache.get("key4") == {"value": 4}

    def test_path_alias(self):
        from backend.api.bridge_cache import _AnalysisLruCache

        cache = _AnalysisLruCache(maxsize=10)
        cache.put("content_key_abc", {"result": "ok"}, path_alias="/tmp/test.wav")
        result = cache.get_by_path("/tmp/test.wav")
        assert result == {"result": "ok"}, "Path-Alias sollte zum cached Wert führen"

    def test_cache_hit_rate(self):
        from backend.api.bridge_cache import _AnalysisLruCache

        cache = _AnalysisLruCache(maxsize=100)
        hits = 0
        misses = 0

        # Erstelle Cache-Einträge
        for i in range(50):
            cache.put(f"key{i}", {"value": i})

        # Simuliere Cache-Zugriffe (70% Hits, 30% Misses)
        rng = np.random.default_rng(42)
        for _ in range(100):
            idx = int(rng.integers(0, 80))  # 0-79: 50 existieren, 50-79 nicht
            if cache.get(f"key{idx}") is not None:
                hits += 1
            else:
                misses += 1

        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        # Cache-Hit-Rate sollte > 50% sein (bei 50/80 existierenden Keys)
        assert hit_rate > 0.5, f"Cache-Hit-Rate zu niedrig: {hit_rate:.2f}"


@pytest.mark.unit
class TestContentCacheKey:
    """Validiert die content_cache_key Funktion."""

    def test_content_key_deterministic(self):
        from backend.api.bridge_cache import content_cache_key

        # Gleiche Datei sollte gleichen Key zurückgeben
        key1 = content_cache_key("/tmp/test.wav")
        key2 = content_cache_key("/tmp/test.wav")
        assert key1 == key2, "Content-Key sollte deterministisch sein"


@pytest.mark.unit
class TestDefectCache:
    """Validiert die Defekt-Cache-Funktionen."""

    def test_cache_defect_result(self):
        from backend.api.bridge_cache import cache_defect_result, get_cached_defect_result, clear_defect_cache

        # Speichere Defekt-Ergebnis
        defect_data = {"defects": ["clicks", "hiss"], "severity": 0.7}
        cache_defect_result("/tmp/test.wav", defect_data)

        # Hole cached Ergebnis
        result = get_cached_defect_result("/tmp/test.wav")
        assert result == defect_data, "Cached Defekt-Ergebnis sollte identisch sein"

        # Lösche Cache
        clear_defect_cache()
        result2 = get_cached_defect_result("/tmp/test.wav")
        assert result2 is None, "Cache sollte nach clear_defect_cache leer sein"


@pytest.mark.unit
class TestPerformanceGuardIntegration:
    """Validiert PerformanceGuard-Cache-Interaktion."""

    def test_performance_guard_with_cache(self):
        from backend.core.performance_guard import PerformanceGuard, QualityMode
        from backend.api.bridge_cache import cache_defect_result, get_cached_defect_result

        # Erstelle PerformanceGuard
        guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True)
        # Starte Monitoring mit realistischer Audio-Dauer
        guard.start_monitoring(180.0)  # 3 Minuten Audio
        
        # should_skip_phase sollte False zurückgeben bei niedriger RT
        skip = guard.should_skip_phase("phase_01_denoise", estimated_time_seconds=5.0, remaining_phases=20)
        assert not skip, "Phase 01 sollte nicht geskippt werden bei niedriger RT"

        # Cache Defekt-Ergebnis für Performance-Monitoring
        defect_data = {"defects": ["hiss"], "severity": 0.5}
        cache_defect_result("/tmp/perf_test.wav", defect_data)
        result = get_cached_defect_result("/tmp/perf_test.wav")
        assert result == defect_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
