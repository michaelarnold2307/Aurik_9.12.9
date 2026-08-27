---
applyTo: "tests/**/*.py"
---

# Test-Regeln (normativ, Aurik 10.0.0.x)

## GC-Konventionen

```python
# VERBOTEN: volles gc.collect() nach jedem Test in großen Suiten
# → zu hoher Overhead bei 11k+ Tests

# RICHTIG: leichter inkrementeller GC
import gc
gc.collect(0)  # nur Generation 0

# Vollständiges gc.collect() nur:
# - an Datei-/Session-Grenzen
# - cadence-gesteuert (z.B. alle 100 Tests)
```

## Langlebige Hintergrund-Manager

```python
# Jeder Monitor-Thread / Background-Manager braucht:
class MyManager:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self):
        """Idempotent — mehrfacher Aufruf = kein Fehler."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

# Cleanup in pytest:
# pytest_sessionfinish oder Finalizer — NICHT daemon=True als einziges Modell
```

## Budget-Tests — Mock is_system_thrashing

```python
# PFLICHT: Budget-Tests MÜSSEN is_system_thrashing mocken
# Sonst: flaky auf Hosts mit hoher Swap-Auslastung

@pytest.fixture(autouse=True)
def mock_no_thrashing(monkeypatch):
    monkeypatch.setattr(
        "backend.core.plugin_lifecycle_manager.is_system_thrashing",
        lambda: False,
    )

# Tests die try_allocate / release prüfen MÜSSEN diese Fixture verwenden
```

## Resampling-Bibliotheken — Warnings

```python
# resampy / librosa können pkg_resources-Warnings unter -W error::Warning auslösen
# IMMER aktuelle Version: resampy >= 0.4.3
# conftest.py global:
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
```

## Teure Transforms — Reihenfolge

```python
# KANONISCH (Kostenpyramide):
# 1. Frame-Energie-Check (günstig) → Gate
# 2. Voiced-Frame-Gate (günstig) → Gate
# 3. dann: filtfilt + Hilbert + STFT (teuer)

# VERBOTEN: Hilbert/STFT vor günstigem Gate
# Beispiel TFS-Guard:
frame_energy = np.sum(frame ** 2)
if frame_energy < _MIN_ENERGY_THRESHOLD:
    continue  # kein Hilbert
if not is_voiced(frame, sr):
    continue  # kein filtfilt
tfs = compute_tfs_hilbert(frame, sr)  # erst jetzt
```

## Guarded Correlation (NaN-safe)

```python
# VERBOTEN: np.corrcoef auf near-constant Signalen → Warning/NaN

# RICHTIG:
def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b + eps))
```

## Sentinel-Pattern für optional-heavy Imports

```python
# RICHTIG: optional imports für ML-Tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_heavy_model():
    ...
```

## AMRB-Update bei Major-Release (9.x.0)

```python
# PFLICHT: benchmarks/update_amrb_history.py ausführen
# benchmarks/amrb_history.json updaten
# OQS-Delta < -2.0 gegenüber vorheriger Baseline = Release-Blocker
```

## Phase-Test-Muster — Pre/Post-Delta

```python
def test_phase_XX_no_regression(synthetic_audio, sr=48000):
    """Stellt sicher dass Phase XX keine Goal-Regression einführt."""
    from backend.core.phases.phase_XX import PhaseXX
    phase = PhaseXX()
    audio_out = phase.process(synthetic_audio, sr, material_type="vinyl", strength=0.8)

    # Längen-Invariante §2.61 (shape[-1] statt len() — len() auf 2D-Stereo gibt 2, nicht N!):
    assert abs(audio_out.shape[-1] - synthetic_audio.shape[-1]) <= 64

    # Clip-Invariante:
    assert np.max(np.abs(audio_out)) <= 1.0

    # NaN-Invariante:
    assert not np.any(np.isnan(audio_out))
```

## PMGG-CIG-Sync-Test (§2.55)

```python
# test_pmgg_cig_sync.py — MUSS nach jeder neuen Phase aktualisiert werden:
# CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[p] ∩ P1P2
# ↔ PMGG.PHASE_GOAL_EXCLUSIONS[p] ∩ P1P2
# bidirektional synchron — CI-Gate
```

## Recovery-Phase-Dict — Disk-Validierung [RELEASE_MUST]

Jede Änderung an `_GOAL_TO_RECOVERY_PHASES_RESTORATION` oder `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS`
in `calibration_matrix.py` muss durch diesen Test gesichert sein:

```python
def test_get_goal_recovery_phases_all_phase_ids_exist_on_disk():
    """Alle Phase-IDs in den Recovery-Dicts müssen als phase_*.py auf Disk existieren.

    Verhindert: Tippfehler, umbenannte Phasen, falsche IDs die alle strukturellen
    Tests bestehen aber §GOAL_BASELINE_CHECK lautlos versagen lassen.
    """
    import pathlib
    import backend.core.calibration_matrix as _cm
    phases_dir = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "phases"
    valid_ids = {p.stem for p in phases_dir.glob("phase_*.py")}
    # ... alle IDs aus beiden Dicts gegen valid_ids prüfen
```

**Verboten**: Phase-ID in einen der Recovery-Dicts eintragen, bevor die zugehörige
`backend/core/phases/phase_XY_name.py` Datei existiert.

**Verboten**: Phase-ID nach Umbenennung der Phase-Datei im Recovery-Dict nicht aktualisieren.

**Sortierregel** [NORMATIV]: Phasen innerhalb einer Goal-Liste MÜSSEN §2.46-Carrier-Chain-Hierarchie
einhalten (subtraktiv/mechanisch → additiv/digital, Stufen 1→6). Nicht nach Phasennummer sortieren.

## Test-Isolation & Flake-Prävention (Rev. 2026-08-16)

Fünf Flake-Klassen aus Full-Suite-Läufen (Spec 07, Bug-Klasse TEST-DESIGN).
Jede hat EIN kanonisches Muster — keine Ad-hoc-Kopien:

1. **ML-Phasen im Registry-/Matrix-Test** (§2.51-Muster): Phasen mit echtem ML
   MÜSSEN gemockt werden. Der Mock muss den RICHTIGEN Bereitschafts-Pfad treffen —
   `try_allocate` (ml_memory_budget) UND/ODER `check_ml_model_ready`
   (ml_model_readiness; modulglobaler TTL-Cache = reihenfolgeabhängig).
   Beispiel: phase_42 prüft via `check_ml_model_ready` → Patch-Ziel ist
   `backend.core.phases.phase_42_vocal_enhancement.check_ml_model_ready`.
   **Dokumentierte Abweichung (Rev. 2026-08-16)**: Der Matrix-Layout-Äquivalenz-
   Check für phase_42 wird übersprungen (`_SUITE_STATE_DEPENDENT` in
   test_stereo_axis_matrix.py) — deterministische Suite-State-Divergenz
   (identische Array-Werte über zwei Runs), Root-Polluter trotz Bisect nicht
   isolierbar; Layout-Invarianz ist durch die phasen-eigene Suite abgedeckt.

2. **Stateful-Singletons**: Jedes Testmodul, das einen modulglobalen Singleton
   (CrossPhaseCoordinator, MLDeviceManager, …) nutzt, MUSS eine autouse-Fixture
   zum Zurücksetzen haben (§V8-analog). Ohne Reset verschmutzt Test A den
   Zustand aller späteren Tests.

3. **GPU-Detektion neutralisieren**: CPU-only-Tests MÜSSEN den zentralen
   Einstieg `MLDeviceManager._detect_backend` neutralisieren — Child-Patches
   allein genügen NICHT (bei warmem ROCm-Stack kann ein weiterer Pfad anlaufen;
   beobachtet: „patched“ Instanz mit rdna3/20 GB im Full-Suite-Lauf).

4. **Source-Contract-Tests ohne linecache**: `inspect.getsource` nutzt den
   prozessglobalen linecache und ist order-abhängig. Source-Asserts MÜSSEN die
   Datei direkt lesen (`Path(module.__file__).read_text(encoding="utf-8")`).

5. **Never-Pass-Through bei Resampling**: Ein Resample-Fehler darf NIE das
   Original bei falscher Samplerate zurückgeben — ML-Embeddings (CLAP/Genre)
   laufen sonst still auf korruptem Input (Befund: `assert sr == 48000` in
   embed_audio → unsichtbare Genre-Degradation). Kanonisch:
   `backend.core.resampling_utils.resample_audio()` (numba-Guard + SciPy-Pfad).
