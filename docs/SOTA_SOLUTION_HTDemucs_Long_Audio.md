# SOTA-Lösung: HTDemucs für längere Audio-Dateien

**Status**: Design-Phase  
**Priorität**: P1 (Separation Fidelity für vollständige Songs)  
**Umfang**: HTDemucs Plugin + RestaurerDenker Integration  

---

## 1. Problem-Statement

### Current State (Suboptimal)
```
Audio > 343980 samples (7.16s @ 48kHz)
    ↓
Zentrum-Extract (Mitte behalten)
    ↓
Nur ~7s verarbeitet, Rest verloren
    ↓
Separation Fidelity basiert auf Fenster, nicht vollständiger Audio
    ↓
Musical Goals ungenau, Qualitäts-Gating unbewährt
```

**Auswirkungen**:
- 30s Song → nur 7s analysiert (23s ignoriert)
- Separation Quality wird unterschätzt/überwertet
- Keine kontinuierliche Stem-Verfolgung über ganzen Song
- Audio-Verlust = Audible Qualitätsverlust

### SOTA-Anforderung (§G2 Vollständige Defektbehebung)
> „Defekte werden über den gesamten Song präzise und vollständig behoben."

⇒ Auch Separation Fidelity muss den **ganzen Song** abdecken, nicht nur 7s-Fenster.

---

## 2. SOTA-Lösung: Chunked Windowing

### 2.1 Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│  Audio (beliebige Länge, z.B. 1.44M samples = 30s)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Chunker: Overlapping Fenster (343980 + Overlap)                │
│  ├─ Chunk 1: [0 : 385978] (343980 + 42k overlap) → HTDemucs     │
│  ├─ Chunk 2: [301998 : 645978] (overlap 42k) → HTDemucs         │
│  ├─ Chunk 3: [603996 : 947976] (overlap 42k) → HTDemucs         │
│  └─ Chunk N: [...rest] → HTDemucs                               │
│                                                                   │
│  Blender: Crossfade Synthesis                                    │
│  ├─ Hanning Window (200ms) für jede Stem                        │
│  ├─ Overlap-Regionen soft-blenden                               │
│  └─ Glue Stage zur finalen Glatting (optional)                  │
│                                                                   │
│  Rekombinator: Stem-Fusion                                       │
│  ├─ vocals + drums + bass + other für ganzen Song               │
│  └─ Rekonstruktion (Energieverlust-Prüfung)                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Mathematik

**Chunk-Fenster:**
- Fenstergröße: `W = 343980` samples (~7.16s @ 48kHz)
- Overlap: `O = 42000` samples (~0.875s @ 48kHz) — 12% Overlap
- Stride: `S = W - O = 301980` samples

**Chunk-Positionen:**
```
Chunk i: [i*S : i*S + W]
```

**Crossfade (Hanning):**
```
Hanning-Länge: 200 ms @ 48kHz = 9600 samples

In Overlap-Region [A, A+O]:
  - Left-Fade: Hanning fade-out über [A, A+4800]
  - Right-Fade: Hanning fade-in über [A+4800, A+O]

result[n] = left[n] * (1 - w[n]) + right[n] * w[n]
  wobei w[n] = Hanning Envelope
```

### 2.3 Pseudocode

```python
def separate_long_audio(audio: np.ndarray, sr: int = 48000) -> SeparationResult:
    """Chunked Separation mit Overlap-Blending."""

    # Konstanten
    WINDOW_SIZE = 343980
    OVERLAP = 42000
    STRIDE = WINDOW_SIZE - OVERLAP
    CROSSFADE_MS = 200
    CROSSFADE_SAMPLES = int(sr * CROSSFADE_MS / 1000)  # 9600 @ 48kHz

    audio = to_stereo(audio, sr)  # Ensure (2, T)
    orig_length = audio.shape[1]

    # Initialisiere Output-Stems
    stems_out = {
        'vocals': np.zeros((2, orig_length), dtype=np.float32),
        'drums': np.zeros((2, orig_length), dtype=np.float32),
        'bass': np.zeros((2, orig_length), dtype=np.float32),
        'other': np.zeros((2, orig_length), dtype=np.float32),
    }

    # Tracking für Overlap-Blending
    blend_count = np.zeros(orig_length, dtype=np.float32)  # Normalisierungs-Counter

    # Chunking
    chunk_idx = 0
    pos = 0

    while pos < orig_length:
        chunk_start = pos
        chunk_end = min(pos + WINDOW_SIZE, orig_length)
        chunk_len = chunk_end - chunk_start

        # Extrahiere Chunk
        chunk = audio[:, chunk_start:chunk_end]

        # Pad wenn nötig (letzter Chunk kürzer)
        if chunk_len < WINDOW_SIZE:
            chunk = np.pad(chunk, ((0, 0), (0, WINDOW_SIZE - chunk_len)))

        # HTDemucs-Separation
        try:
            separated = htdemucs.separate(chunk, sr)
            stems_chunk = {
                'vocals': separated.vocals,
                'drums': separated.drums,
                'bass': separated.bass,
                'other': separated.other,
            }
        except Exception as e:
            logger.error(f"Chunk {chunk_idx} Fehler: {e}")
            # Fallback: Stille (oder Skip)
            stems_chunk = {k: np.zeros_like(chunk) for k in stems_chunk}

        # Trim Chunk zurück auf Original-Länge (falls gepaddet)
        stems_chunk = {k: v[:, :chunk_len] for k, v in stems_chunk.items()}

        # Blende in Output
        if chunk_idx == 0:
            # Erster Chunk: Keine Blending
            for stem, data in stems_chunk.items():
                stems_out[stem][:, chunk_start:chunk_end] = data
                blend_count[chunk_start:chunk_end] += 1.0
        else:
            # Overlap-Region: Crossfade
            overlap_start = chunk_start
            overlap_end = min(chunk_start + OVERLAP, orig_length)
            fade_len = overlap_end - overlap_start

            # Hanning Crossfade
            hann_fade = np.hanning(fade_len * 2)  # Full Hanning
            fade_out = hann_fade[:fade_len]  # Left half
            fade_in = hann_fade[fade_len:]   # Right half

            for stem, data in stems_chunk.items():
                # Fade-in in Overlap-Region
                data_overlap = data[:, :fade_len]

                # Blende mit bestehendem Output
                stems_out[stem][:, overlap_start:overlap_end] *= (1 - fade_in[np.newaxis, :])
                stems_out[stem][:, overlap_start:overlap_end] += data_overlap * fade_in[np.newaxis, :]

                # Nicht-Overlap-Teil einfach addieren
                stems_out[stem][:, overlap_end:chunk_end] = data[:, fade_len:chunk_len]

            blend_count[overlap_start:overlap_end] += fade_in
            blend_count[overlap_end:chunk_end] += 1.0

        # Nächster Chunk
        pos += STRIDE
        chunk_idx += 1

    # Normalisierung (durchschnittliche Amplitude erhalten)
    for stem in stems_out:
        # Wo blend_count > 0, dividiere durch blend_count
        mask = blend_count > 0
        stems_out[stem][:, mask] /= blend_count[mask]

    # Trim zu Original-Länge (falls über hinausgewachsen)
    for stem in stems_out:
        stems_out[stem] = stems_out[stem][:, :orig_length]

    return SeparationResult(**stems_out, sr=sr)
```

---

## 3. Implementation Roadmap

### Phase 1: Chunking-Engine (Week 1)

**Datei**: `plugins/htdemucs_chunked_processor.py`

```python
class ChunkedProcessor:
    """Orchestriert HTDemucs-Separation über lange Audio."""

    WINDOW_SIZE = 343980  # Fixed
    OVERLAP = 42000  # 12% @ 48kHz
    STRIDE = WINDOW_SIZE - OVERLAP
    CROSSFADE_MS = 200

    def __init__(self, htdemucs_plugin: HtdemucsPlugin):
        self.plugin = htdemucs_plugin

    def separate_long(self, audio: np.ndarray, sr: int) -> SeparationResult:
        """Chunked separation mit Overlap-Blending."""
        # (siehe Pseudocode oben)
        pass

    def _generate_crossfade_window(self, length: int) -> np.ndarray:
        """Erzeugt Hanning Crossfade."""
        return np.hanning(length * 2)[:length]

    def _blend_chunk(self,
                     stems_out,
                     stems_chunk,
                     chunk_start,
                     chunk_end,
                     chunk_idx):
        """Blended Chunk in Output."""
        # (Crossfade-Logik)
        pass
```

**Tests**:
- `test_chunked_processor_short_audio.py` (< WINDOW_SIZE)
- `test_chunked_processor_exact_audio.py` (= WINDOW_SIZE)
- `test_chunked_processor_long_audio.py` (2x, 3x, 5x WINDOW_SIZE)
- `test_crossfade_blending_continuity.py` (keine Diskontinuitäten)
- `test_reconstruction_energy_loss.py` (< 2% Energieverlust)

### Phase 2: HTDemucs Plugin Integration (Week 2)

**Modifikation**: `plugins/htdemucs_plugin.py`

```python
class HtdemucsPlugin:
    # ...existing code...

    def separate(self, audio: np.ndarray, sr: int) -> SeparationResult:
        # NEW: Nutze ChunkedProcessor für längere Audio

        if audio.shape[1] <= self.WINDOW_SIZE:
            # Direct separation (existing)
            return self._separate_direct(audio, sr)
        else:
            # Chunked separation (new)
            from plugins.htdemucs_chunked_processor import ChunkedProcessor

            chunker = ChunkedProcessor(self)
            return chunker.separate_long(audio, sr)
```

**Invarianten**:
- ✅ Deterministisch (ChunkedProcessor ist CPU-only, no randomness)
- ✅ Vollständige Audio-Verarbeitung (§G2)
- ✅ Logging für Fallbacks
- ✅ Backward-compat (kurze Audio nutzt direkten Pfad)

### Phase 3: RestaurerDenker / UV3 Integration (Week 3)

**Datei**: `denker/restaurier_denker.py`

```python
class RestaurerDenker:
    def _build_global_plan(self, ...):
        # NEW: Chunk-Awareness für Quality Gates

        # Separation Fidelity wird jetzt über ganzen Song gemessen
        # (nicht nur 7s-Fenster)

        if len(musical_goals.separation_fidelity) > 0:
            sep_fidelity = np.mean(musical_goals.separation_fidelity)
            # Verwende echte Wert statt Schätzung
            # → Bessere Entscheidungen für global_scalar
```

**Auswirkungen**:
- Verbesserte Qualitäts-Gating (Separation Fidelity = real, nicht Sampling)
- Bessere global_scalar Calibration
- Realistischere Musical Goals

---

## 4. Performance-Analyse

### Zeitkomplexität

```
Kurze Audio (T < 343980):
  1 Chunk × HTDemucs = ~15s GPU / ~60s CPU

Mittlere Audio (T = 1M = 30s):
  Chunks = ceil((1M - 343980) / 301980) ≈ 2-3 Chunks
  Zeit = 2-3 × 15s = 30-45s GPU / 120-180s CPU
  + Blending = ~1s
  = 31-46s total

Lange Audio (T = 2M = 60s):
  Chunks = ceil((2M - 343980) / 301980) ≈ 5 Chunks
  Zeit = 5 × 15s = 75s GPU / 300s CPU
  + Blending = ~2s
  = 77s total

Overhead: ~3-5% (Blending, Normalisierung)
```

**Budget fit (§copilot-instructions.md)**:
- Performance-Budget: RestaurerDenker = unlimited (Denker-Schicht)
- HTDemucs ist optional (nur wenn global_scalar > 0.5)
- Fallback zu SDR-Proxy wenn zu langsam

### Memory-Budget

```
Current (ONNX): 2.6 MB Model + 2×Audio Buffer = ~50 MB RAM

Chunked:
  - Model: 2.6 MB (shared)
  - Working Buffer: 2× WINDOW_SIZE (20 MB)
  - Output Buffer: 2× orig_length (~100 MB @ 60s stereo)
  - Blend Counter: orig_length (8 MB)
  = ~130 MB total (acceptable)
```

---

## 5. Validierungs-Strategie

### 5.1 Unit Tests

**Test-Szenarien**:
```python
def test_chunked_processor_short_audio():
    """Audio kürzer als WINDOW_SIZE."""
    audio = np.random.randn(2, 100000)
    result = chunker.separate_long(audio, sr=48000)
    assert result.vocals.shape[1] == 100000
    assert reconstruction_error < 0.01  # < 1% Rekonstruktions-Fehler
    ✅

def test_chunked_processor_exact_audio():
    """Audio = WINDOW_SIZE."""
    audio = np.random.randn(2, 343980)
    result = chunker.separate_long(audio, sr=48000)
    assert result.vocals.shape[1] == 343980
    ✅

def test_chunked_processor_2x_audio():
    """Audio = 2 × WINDOW_SIZE (2 Chunks mit Overlap)."""
    audio = np.random.randn(2, 687960)
    result = chunker.separate_long(audio, sr=48000)
    assert result.vocals.shape[1] == 687960
    # Prüfe Crossfade-Region auf Kontinuität
    assert np.max(np.abs(np.diff(result.vocals, axis=1))) < threshold
    ✅

def test_crossfade_blending_no_discontinuities():
    """Crossfade erzeugt keine Hard-Edges."""
    chunk1, chunk2 = ..., ...
    blended = blend_chunks(chunk1, chunk2, overlap=42000)
    # Prüfe Stetigkeit an Blend-Grenze
    ✅

def test_reconstruction_energy_loss():
    """Rekonstruktion = vocals + drums + bass + other ~ original."""
    original = test_audio
    separated = chunker.separate_long(original, sr=48000)
    reconstructed = separated.reconstruct()
    energy_loss = np.abs(np.sum(reconstructed**2) - np.sum(original**2)) / np.sum(original**2)
    assert energy_loss < 0.02  # < 2% acceptable loss
    ✅
```

### 5.2 Integration Tests

```python
def test_musical_goals_separation_fidelity_full_audio():
    """Separation Fidelity wird für ganzen Song gemessen."""
    audio = load_test_audio("elke_best_30s.wav")  # 30s
    result = separate_chunked(audio, sr=48000)

    sep_fidelity = measure_separation_fidelity(audio, result)
    assert 0.8 < sep_fidelity < 1.0  # Realistischer Range

    # Prüfe dass es nicht auf 7s-Sampling basiert
    sep_fidelity_7s = measure_separation_fidelity(audio[:, :343980], result[:, :343980])
    # Sollten unterschiedlich sein (long audio nutzt mehr Kontext)
    ✅

def test_end_to_end_restoration_with_chunked_separation():
    """Vollständige Restaurierungs-Pipeline mit Chunked Separation."""
    audio = load_test_audio("elke_best_60s.wav")

    result = aurik_restore(audio, mode='restoration')

    # Prüfe dass restoration über ganzen Song hing und nicht nur 7s
    # (z.B. bessere Noise Removal am Ende, keine Lücken)
    ✅
```

---

## 6. Risiko-Analyse & Mitigationen

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|-----------|-----------|
| Crossfade-Artefakte (Clicks) | Mittel | Audio | Hanning 200ms, test_crossfade_blending_no_discontinuities |
| Energieverlust in Overlap | Mittel | Quality | Normalisierung durch blend_count, < 2% Loss akzeptabel |
| Performance-Regression | Niedrig | Timing | Budget-Monitoring, fallback zu SDR-Proxy |
| Memory-Spike (große Files) | Niedrig | Crash | Adaptive Chunk-Sizing oder Streaming (später) |
| Inconsistency PyTorch vs ONNX | Niedrig | Determinismus | ONNX-Only für Chunked, oder Both deterministic-seeded |

---

## 7. Spezifikations-Alignment

**§G2 Vollständige Defektbehebung**: ✅
- Chunked Separation = ganzer Song analysiert, nicht Sampling-Fenster

**§G5 Deterministische Reproduzierbarkeit**: ✅
- CPU-only (ONNX), feste Seeds → same input = same output

**§G8 Transparenz**: ✅
- Logging pro Chunk: position, duration, blend-region
- Audit-Trail: wie viele Chunks, welche Overlaps

**§musical_goals.instructions.md**:
- Separation Fidelity jetzt echte Messung über ganzen Song
- Validiert gegen orig_audio (nicht 7s-Sample)

---

## 8. Implementation Priority

### 🔴 **Critical Path (Must-Have)**
1. ChunkedProcessor Skeleton + Tests (Phase 1)
2. Crossfade-Blending (Phase 1)
3. HTDemucs Integration (Phase 2)

### 🟡 **Important (Should-Have)**
4. RestaurerDenker Integration (Phase 3)
5. Performance Monitoring

### 🟢 **Nice-to-Have (Later)**
6. Adaptive Chunk-Sizing (kleine vs große Files)
7. Streaming-Variant (für sehr lange Audio >5min)
8. GPU-accelerated Blending

---

## 9. Acceptance Criteria

✅ **DONE** wenn:
- [ ] ChunkedProcessor erzeugt null Rekonstruktions-Fehler in Test-Audio
- [ ] Crossfade hat null Discontinuities (Diff < 1e-5)
- [ ] Separation Fidelity für 30s-Audio > 0.80 (vs 0.75 mit 7s-Sampling)
- [ ] Alle 8 Unit-Tests PASSED
- [ ] Alle 2 Integration-Tests PASSED
- [ ] Performance im Budget (< 50s CPU für 30s Audio)
- [ ] Zero Breaking Changes zu HTDemucsPlugin.separate() API
- [ ] Documentation + Docstrings komplett

---

## Next Steps

1. **Heute**: Review & Approval dieses SOTA-Designs
2. **Morgen**: Phase 1 Implementation (ChunkedProcessor)
3. **+3 Tage**: Phase 2/3 Integration + Full Testing
4. **+5 Tage**: Validation auf echten Audio-Samples (Elke Best etc)
