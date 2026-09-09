---
applyTo: "{backend/core/dsp/*.py,plugins/*.py}"
---

# DSP / Plugin-Regeln (normativ, Aurik 10.0.0.x)

## ML-Device — IMMER über ml_device_manager

```python
# VERBOTEN:
model.to("cuda")                        # direkt, ignoriert AMD/ROCm
providers = ["CUDAExecutionProvider"]   # ignoriert DirectML/ROCm

# RICHTIG (Heavy-Plugin):
from backend.core.ml_device_manager import get_torch_device, get_ort_providers
model.to(get_torch_device("PluginName"))   # fp16 + Tier automatisch
session = ort.InferenceSession(path, providers=get_ort_providers("PluginName"))

# Light-Plugin / DSP:
model.to("cpu")
torch.set_num_threads(os.cpu_count())
providers = ["CPUExecutionProvider"]
```

## §2.62 Psychoakustischer Masking-Guard (NR-Algorithmen)

```python
# PFLICHT vor jedem NR-Aufruf (DeepFilterNet, OMLSA, SGMSE+):
from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172

masking_threshold = compute_masking_threshold_iso11172(audio, sr)
# Bark-Skala (ISO 11172-3): 0–24 Bark, ~24 kritische Bänder bis 22 kHz
# ERB-Skala (ISO 532-1, Moore-Glasberg): genauer für Lautheit — für Loudness-Normalisierung
# → Masking-Guard nutzt Bark (ISO 11172-3); Loudness-Normalisierung nutzt ERB (ISO 532-1)

# MMSE-LSA Gain (Ephraim-Malah 1985, Log-Spectral Amplitude Estimator):
# BESSER als einfacher Wiener-Filter-Floor — erhält Spektralform, reduziert Musical Noise:
from scipy.special import exp1 as expint1  # E1(v) = exponential integral; exp1(v) = ∫_v^∞ exp(-t)/t dt
for band in range(n_bands):
    xi = max(noisy_power[band] / noise_estimate[band] - 1.0, 0.0)  # a-priori SNR
    gamma = noisy_power[band] / noise_estimate[band]               # a-posteriori SNR
    v = xi / (1.0 + xi) * gamma
    G_mmse_lsa = xi / (1.0 + xi) * np.exp(0.5 * expint1(v))       # MMSE-LSA gain
    G_floor[band] = max(G_mmse_lsa, 0.10, masking_threshold[band] / noise_estimate[band])
    # VERBOTEN: G_floor < 0.10 in Bändern mit Musik-Energie > -60 dBFS

# Ergebnis: kein "totes Stille"-Artefakt zwischen Phrasen
# VERBOTEN: einfacher Wiener-Filter ohne Masking-Floor — verursacht Musical-Noise
```

## Noise-Schätzung — IMCRA/OMLSA (stationär + nicht-stationär)

```python
# KANONISCH — Rausch-Schätzung für Musik (nicht-stationäre Rausch-Quellen):
# OMLSA (Cohen & Berdugo, 2004) = Optimal Modified Log-Spectral Amplitude
# → besser als Spectral Subtraction für Musik (verarbeitet harmonische Störgeräusche)
# Noise-Schätzung via IMCRA (Cohen, 2003):
#   - gleitende Minimum-Schätzung + Sprachpräsenz-Posterior
#   - adaptive Zeitkonstanten: tau_min=0.04s, alpha_d=0.85, alpha_s=0.9
# DeepFilterNet + OMLSA gemeinsam: DFN für Breitband-NR, OMLSA für Restgeräusch
from backend.core.dsp.noise_estimator import compute_imcra_noise_estimate
noise_psd = compute_imcra_noise_estimate(audio, sr, alpha_d=0.85, alpha_s=0.9)
# Initialphase (2s) → konservative Schätzung (Faktor 1.3 × Minimum)
```

## DeepFilterNet — Energy-Bias-Pflicht (§0j)

```python
# VERBOTEN: DeepFilterNet ohne energy_bias auf Vokal/Instrumental
# Harmonik-Regionen werden ohne Bias als Rauschen klassifiziert

# RICHTIG — energy_bias modifiziert DFNs internen Noise-Floor-Schätzwert:
# DFN schätzt Rauschumgebung → energy_bias verschiebt die Entscheidungsgrenze
# sodass harmonische Energie NICHT als Rauschen abgetragen wird
# (energy_bias in dB → intern: noise_floor_estimate *= 10^(energy_bias/20))
if panns_singing >= 0.35:
    # Vokal-Material: Register-adaptiv für präziseste Einstellung
    # (Register-Detektor NUR auf Vokal-Material aufrufen — für Instrumental sinnlos!)
    from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal
    register = detect_vocal_register_temporal(audio, sr)
    energy_bias = _REGISTER_BIAS.get(register.dominant, -6.0)
    # Fallback -6.0 dB wenn register.dominant kein Eintrag in _REGISTER_BIAS
elif is_instrumental:
    energy_bias = -9.0  # dB — Instrumental: aggressivere Schutzgrenze; kein Register
else:
    energy_bias = -6.0  # dB — Default (unklares Material)
```

## HNR-Guard nach NR auf Gesangsmaterial

```python
# PFLICHT wenn panns_singing >= 0.25 + nach DFN/SGMSE+/OMLSA:
from backend.core.dsp.hnr_guard import apply_hnr_blend

audio_out = apply_hnr_blend(audio_pre_nr, audio_post_nr, sr)
# ΔHNR > 3 dB → automatischer Dry-Blend
# verhindert "klinischen" Klang nach aggressivem NR
```

## LPC-Ordnung — Material-abhängig

```python
# VERBOTEN: LPC-Ordnung < 16 bei 16 kHz oder < 30 bei 48 kHz
# RICHTIG:
if analysis_sr == 48000:
    lpc_order = 30  # bis 40 für breite Formant-Tracks
elif analysis_sr == 16000:
    lpc_order = 16  # bevorzugt für Shellac BW ≤ 8 kHz (Downsampling → 16k)
# lpc_formant_tracker.py verwendet _LPC_ORDER=16 + _LPC_ANALYSIS_SR=16000
```

## Singleton-Pattern — ALLE Kernmodule

```python
import threading

_instance = None
_lock = threading.Lock()

def get_my_plugin():
    global _instance  # oder list-Container: _holder = [None]
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyPlugin()
    return _instance
```

## ONNX-Chunking (Heavy-Plugins)

```python
# KANONISCH — Chunk-Verarbeitung mit Overlap-Add:
chunk_size = 65536  # ~1.4s bei 48kHz
overlap = 4096
for i in range(0, n_samples, chunk_size - overlap):
    chunk = audio[..., i: i + chunk_size]
    out_chunk = session.run(None, {"input": chunk})[0]
    # shape[-1] statt len() — len(out_chunk) wäre 2 für 2D-Stereo-Output!
    out_len = out_chunk.shape[-1]
    # Overlap-Add mit Hann-Fenster (window: shape (out_len,) oder (1, out_len) für Broadcasting)
    output[..., i: i + out_len] += out_chunk * window[:out_len]

# OOM-Fallback → DSP-Kette, nie Crash:
try:
    result = heavy_model.process(audio)
except (RuntimeError, MemoryError):
    metadata["ml_fallbacks_used"]["model_name"] = True
    result = _dsp_fallback(audio, sr)
```

## MIIPHER-Fallback (Codec-Degradation — Spec 04 Rev. 2026-08-15)

Die MIIPHER-Stufe ist Codec-Restaurierer (`mp3_low`/`streaming`/`aac`/`minidisc`).
Stark degradierter Gesang (SNR < 10 dB) läuft über SGMSE+ v2, nicht über MIIPHER.
Legacy-Pfad des alten Plugins (operative Implementierung ist der offene DiT,
`plugins/miipher_dit_plugin.py`, Gate `should_apply`):

```python
from plugins.miipher_plugin import get_miipher_plugin

if material_type in {"mp3_low", "streaming", "aac", "minidisc"} and panns_singing >= 0.35:
    miipher = get_miipher_plugin()
    if miipher.should_activate(material_type, panns_singing):
        audio_pre_miipher = audio.copy()  # für HNR-Blend-Referenz
        audio = miipher.enhance(audio, sr)
        # Intern: Stub → DeepFilterNet(-6dB) → Wiener-Fallback
        # PFLICHT: apply_hnr_blend() nach MIIPHER — ΔHNR > 3 dB → Dry-Blend (§0p HNR-Schutz)
        from backend.core.dsp.hnr_guard import apply_hnr_blend
        audio = apply_hnr_blend(audio_pre_miipher, audio, sr)
```

## §0p Formant-Schutz (Pflicht bei `panns_singing ≥ 0.25`)

```python
# KANONISCH — nach jeder DSP-Phase auf Gesangsmaterial:
# §Align 2026-09-08: Der Guard lebt in lpc_formant_tracker (nicht in einem
# separaten formant_guard-Modul); die Frequenz-JND kommt aus resolve_jnd_
# tolerance_db, die lokale Maskierungs-JND aus residuum_masking.
from backend.core.dsp.lpc_formant_tracker import check_formant_shift_db
from backend.core.dsp.jnd_tolerance import resolve_jnd_tolerance_db

# §2.71 (v10.0.0): Per-Formant-Toleranz — F1/F2 ±1 dB, F3/F4 ±1.5 dB
# (war: global ±2 dB — zu grob für Weltklasse-Vokalrestauration).
# §V43: min(threshold_db, resolve_jnd_tolerance_db(f_hz)) — immer das Strengere.
# §P1-3 (2026-09-08, Hörordnung Ebene 2): Die lokale Maskierungs-Marge des
# Formant-Bands (delta_masking_margin_db_per_band, max. 6 dB) ERHÖHT die
# Toleranz — eine maskierte Formant-Verschiebung ist unhörbar und löst
# keinen Rollback aus. Rollback-Warnung enthält den Maskierungskontext.
# §V6-Fix (2026-09-08): _burg_lpc hatte einen Shape-Mismatch ab order ≥ 2 →
# der Guard war ein stummer No-op. Jetzt aktiv; Exceptions loggen als warning.
rollback, max_shift_db = check_formant_shift_db(audio_pre, audio_post, sr, threshold_db=2.0)
if rollback:
    logger.warning("formant_shift %.2f dB über Toleranz → rollback", max_shift_db)
    return audio_pre  # keine Ausnahmen
```

## §0p Vibrato-Schutzzone (Pflicht bei `panns_singing ≥ 0.25`)

```python
# KANONISCH — vor jeder NR/Transient/Dynamics-Phase auf Gesangsmaterial:
from backend.core.dsp.natural_performance_detector import detect_performance_artifacts

artifacts = detect_performance_artifacts(audio, sr)
vibrato_mask = artifacts.vibrato_mask  # bool-Array, True = geschützte Frame

# Alle Phasen in Vibrato-Zonen: strength auf max. 0.20 begrenzen
if np.any(vibrato_mask):
    strength = min(strength, 0.20)  # Vibrato ist Naturalness-Marker
    logger.debug("vibrato_guard: strength capped to 0.20 (%d frames)", np.sum(vibrato_mask))

# §2.72 (v10.0.0) Vibrato-Tiefe — F0-Modulationstiefe darf nicht > ±10 % reduziert werden:
from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation
_vdp = check_vibrato_depth_preservation(audio_pre, audio_post, sr)
if _vdp.depth_reduction_pct > 10.0:
    # Strength in Vibrato-Zonen halbieren
    strength_in_vibrato = strength * 0.5
    logger.warning("vibrato_depth_guard: reduction=%.1f%% → strength %.2f→0.5×",
                   _vdp.depth_reduction_pct, strength)
    # Blend: 50 % Dry in Vibrato-Frames, volle Strength außerhalb
```

## §0p Passaggio-Zone-Awareness (MIIPHER/DFN bei Gesang)

```python
# KANONISCH — Registerübergangszonen bei ML-NR:
from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal

register_map = detect_vocal_register_temporal(audio, sr)
# register_map.passaggio_zones: List[(start_s, end_s)]

# Energy-Bias in Übergangszone = Mittelwert Brust+Kopf:
_REGISTER_BIAS = {
    "chest": -6.0, "head": -6.0, "falsetto": -9.0,
    "passaggio": -3.0,  # Mittelwert Brust/Kopf
    "fry": -12.0, "whisper": -15.0,
}
energy_bias = _REGISTER_BIAS.get(register_map.dominant, -6.0)
```

## Timbral Coherence Guard

```python
from backend.core.dsp.timbral_coherence_guard import (
    extract_song_noise_profile,
    compute_timbral_coherence_score,
)

noise_profile = extract_song_noise_profile(audio_pre_pipeline, sr)
# ... nach NR-Phasen ...
coherence = compute_timbral_coherence_score(audio_post_nr, noise_profile, sr)
if coherence < 0.80:
    logger.warning("timbral_coherence=%.3f < 0.80 — Rollback auf pre-NR (§CSTC)", coherence)
    audio_post_nr = audio_pre_pipeline  # Rollback
# VERBOTEN: assert coherence >= 0.80 — assert ist in Produktionscode durch -O deaktivierbar
# Vinyl → rosa Rauschtextur; Tape → Brown+HF-Hiss; CD → Flat/Weiß
```

## Cross-Segment Timbral Coherence — Rauschtextur

```python
# Spektrale Form des Restrauschens MUSS zum Trägerprofil passen:
# Vinyl:   rosa   (1/f)
# Tape:    Brown + HF-Hiss
# CD:      Weiß / Flat
# Shellac: Spezifisches Oberflächenprofil
# Kohärenz-Score ≥ 0.80 Pflicht (§0a Rauschtextur-Invariante)
```

## Multi-Singer Detection

```python
from backend.core.dsp.vocal_register_detector import detect_multi_singer

if detect_multi_singer(audio, sr, panns_singing):
    metadata["multi_singer"] = True
    # Resemblyzer-Gate ÜBERSPRINGEN (Embedding für Einzelidentität, nicht Duett)
    # singer_identity_cosine-Gate deaktiviert
```

## Signal-relatives Gate (V04)

```python
# VERBOTEN:
gate_dbfs = -36.0  # feste Konstante

# RICHTIG:
from backend.core.dsp.gain_utils import compute_signal_relative_gate_dbfs
gate_dbfs = compute_signal_relative_gate_dbfs(
    pre_phase_audio, material_key=material_type
)
# reference_for_gate=pre_phase_audio — IMMER
```

## Bandfilter — Zero-Phase

```python
# VERBOTEN: sosfilt(sos, audio) addiert zu Original
# RICHTIG:
from scipy.signal import sosfiltfilt
filtered = sosfiltfilt(sos, audio)  # zero-phase überall wo Band auf Signal addiert
```

## STFT — Fenster und Overlap-Pflicht

```python
# KANONISCH für alle STFT-basierten Phasen (Rekonstruktionsqualität):
_STFT_HOP_FRACTION = 0.25     # 75 % Overlap — Pflichtstandard
_STFT_WINDOW = "hann"         # Hann-Fenster: perfekte Rekonstruktion bei 75 % Overlap
# VERBOTEN: hop_fraction > 0.5 (< 50 % Overlap) bei Synthesis-STFT
# VERBOTEN: Rechteck-Fenster (Rectangular) für Analyse und Synthese gleichzeitig

# Konsistenz-Invariante: Analyse-Fenster und Synthese-Fenster MÜSSEN gleich sein
# Ausnahme: PGHI/Vocos nutzen eigene interne Fensterung — nicht manuell überschreiben
```

## Phasenrekonstruktion — PGHI statt Griffin-Lim

```python
# VERBOTEN: griffinlim() als Endschritt (vgl. VERBOTEN.md V05)
# Ursache: iterative Phase-Estimation ist nicht-deterministisch + zu langsam

# RICHTIG — Option 1: PGHI (Phase Gradient Heap Integration, Prusa et al. 2017):
# Deterministisch, ein Pass, minimal-phasenäquivalent
# Parameter:
_PGHI_GAMMA = 0.25 * frame_size**2 / sr  # Frequenz-Zeit-Kopplung
_PGHI_TOLERANCE = 1e-6                    # Konvergenz-Schwelle
from backend.core.dsp.pghi import pghi_reconstruct
audio_out = pghi_reconstruct(magnitude, sr, n_fft=n_fft, hop_length=hop_length,
                              gamma=_PGHI_GAMMA, tol=_PGHI_TOLERANCE)

# RICHTIG — Option 2: Vocos (Siuzdak 2023) — vollständig neural, kein STFT-invert:
# Besser für hochqualitative Vokale; nutzt iSTFT-basierte Architektur ohne iterative Optimierung
# Weniger anfällig auf spektrale Löcher als PGHI
from plugins.vocos_plugin import get_vocos_plugin
voc = get_vocos_plugin()
audio_out = voc.decode(magnitude_features, sr=sr)  # CPU-only empfohlen (light)
# Entscheid PGHI vs Vocos: PGHI wenn Magnitude aus linearem DSP stammt;
# Vocos wenn Magnitude aus ML-Modell (DFN, SGMSE+) stammt
```

## STFT — Sample-Exaktheit der Zeitachse

```python
# Hörordnung Ebene 1 (hoerordnung.instructions.md §3): Timing-/Klang-Invarianten
# sind unverhandelbar. Jede STFT-basierte Phase MUSS die Signallänge sample-
# exakt erhalten — außer den expliziten Zeit-Korrektur-Phasen 12 (Wow/Flutter)
# und 31 (Speed/Pitch), die ihre Längenänderung dokumentieren müssen.
#
# KANONISCH: hop_length ganzzahlig und durch frame_length teilbar wählen;
# keine Slice-/Pad-Arithmetik mit float-Hops; Länge vor/nach Phase prüfen:
if len(audio_out) != len(audio_in):
    audio_out = _align_len(audio_out, len(audio_in))  # trim/pad am ENDE, nie mitten drin
# VERBOTEN: Off-by-one-Slice-/Reshape-Bugs, die die Zeitachse verschieben
# (Befund 2026-08-23: MDEM-Broadcast 224.27s vs 224.28s).
```

## §SCK-R Rücknahme bei Spektralfarben-Verletzung (Konkretisierung zu V24)

```python
# Hörordnung Ebene 1: Verletzt eine Phase die Spektralfarben-Invariante
# (V24, 1/3-Oktav-Energiekurve 200–8000 Hz, Korrelation < 0.97), wird die
# PHASE zurückgenommen oder geblendet — nicht erst am Pipeline-Ende
# „wiederhergestellt“ (Ursache statt Symptom, §V7 (copilot-instructions.md)).
# RICHTIG:
if spectral_color_corr < 0.97:
    audio = pre_phase_audio  # oder Blend; Log mit Begründung (kein Silent-Fail)
    logger.warning("§SCK-R: Phase zurückgenommen (Spektralfarben-Verletzung)")
```

## §WBG-R Rücknahme bei Wärmeband-Verletzung (Konkretisierung zu V25)

```python
# Hörordnung Ebene 1: Senkt eine EINZELNE Phase das Wärmeband (200–800 Hz)
# um > 3 dB (V25: kumulativ max. 4 dB), wird sie zurückgenommen — kein
# Wärme-Boost auf Kosten von Natürlichkeit, keine End-Gate-Kompensation
# als Ersatz für Ursachen-Beseitigung.
# RICHTIG:
if warmth_band_loss_db > 3.0:  # Einzelphasen-Delta, nicht kumulativ
    audio = pre_phase_audio
    logger.warning("§WBG-R: Phase zurückgenommen (Wärmeband-Verlust %.2f dB)", warmth_band_loss_db)
```

## Peak-Guard

```python
# VERBOTEN:
peak = np.max(np.abs(audio))

# RICHTIG:
peak = np.percentile(np.abs(audio), 99.9)
```

---

## §NTI Noise-Textur-Invariante (V19) [RELEASE_MUST v10.0.0]

**Regel**: Das durch NR entfernte Material (`residual = pre − post`) muss dem erwarteten
Defektprofil des Trägers entsprechen — kein Whitening (NR hat Musikinhalt entfernt).

**Semantik**: `_residual` ist der _entfernte Inhalt_, nicht das verbleibende Rauschen.
Sein Spektral-Slope wird gegen die Residual-Erwartungsrange des Trägers geprüft.
Liegt der Slope außerhalb → NR hat Träger-fremdes Material entfernt → Whitening-Warnung.

```python
# PFLICHT nach jeder NR-Phase (DFN, OMLSA, SGMSE+, MIIPHER):
from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

_residual = audio_pre_nr - audio_post_nr  # Entfernter Inhalt (kein in-situ Rauschen!)
_ntd = compute_noise_texture_distance(_residual, material=material_type)
if _ntd > 0.25:
    # Rauschtextur zu weit vom Träger entfernt → Strength halbieren, nicht voll anwenden
    _blend_factor = 0.5
    audio_post_nr = audio_pre_nr * (1.0 - _blend_factor) + audio_post_nr * _blend_factor
    logger.warning("noise_texture_guard: ntd=%.3f > 0.25 → strength x0.5 (V19)", _ntd)
    metadata["noise_texture_distance"] = _ntd

# Residual-Slope-Ranges per Material (_MATERIAL_RESIDUAL_SLOPE_RANGES, 100–8000 Hz):
# SHELLAC:      (-2.0, +8.0) dB/oct  — entferntes Kratz/Knistern ist HF-reich
# VINYL:        (-4.5, +1.5) dB/oct  — entferntes Rauschen leicht LF-betont (1/f)
# TAPE/REEL:    (-4.0, +2.0) dB/oct  — Band-Hiss leicht breitbandig
# CASSETTE:     (-4.0, +2.0) dB/oct  — ähnlich Tape
# CD/DAT:       (-2.5, +2.5) dB/oct  — digitales Quantisierungsrauschen flat
# WAX_CYLINDER: (-1.5, +9.0) dB/oct  — extremes HF-Kratzen
# Scope außerhalb Range → NR hat musik-ähnlichen LF-Inhalt entfernt → Whitening
# VERBOTEN: NR die reines Stille-Fundament (-100 dBFS) auf analog-Material erzeugt
```

## §MKK Mikrodynamik-Korrelation (V20) [RELEASE_MUST v10.0.0]

**Regel**: Das "Atmen" einer Stimme (10ms-Frame-Energie in voiced-Zonen) muss überlebt jede NR/Dynamics-Phase.

```python
# PFLICHT nach jeder NR/Kompressor-Phase wenn panns_singing >= 0.25:
from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation

_corr = frame_energy_correlation(audio_pre, audio_post, sr, frame_ms=10)
if _corr < 0.97:
    # Dry-Wet-Blend: bei corr=0.90 → wet=0.0 (vollständiges Dry); bei 0.97 → wet=1.0
    _wet = float(min(1.0, max(0.0, (_corr - 0.90) / 0.07)))
    audio_post = audio_pre * (1.0 - _wet) + audio_post * _wet
    logger.warning("mikrodynamik_guard: corr=%.3f < 0.97 → wet_blend=%.2f (V20)", _corr, _wet)
    metadata["mikrodynamik_corr"] = _corr

# Warum wichtig: Der Unterschied zwischen "restauriert" und "klinisch"
# liegt fast ausschließlich in der Mikrodynamik der Stimme.
# Kompressor-/NR-bedingte Glättung vernichtet Expressivität.
```

## §MNF Mindestrauschboden in Pausen (V21) [RELEASE_MUST v10.0.0]

**Regel**: Digitale Stille (−∞ dBFS) in Pausenzonen von analog-Material klingt sofort unnatürlich.

```python
# PFLICHT nach jeder NR-Phase auf analog-Material (Shellac, Vinyl, Tape, Cassette):
from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum

_MATERIAL_NOISE_FLOOR_MIN_DBFS = {
    "shellac": -42.0,   # ~15 dB SNR — Rauschen ist hörbar
    "vinyl":   -55.0,   # ~60 dB SNR — subtiles Trägerrauschen
    "tape":    -52.0,   # ~60-70 dB SNR — Brown-Hiss muss bleiben
    "cassette": -50.0,  # ~55 dB SNR — leises HF-Hiss
    "unknown_analog": -52.0,  # konservativer Fallback
}
audio_post = apply_noise_floor_minimum(
    audio_post, sr, material=material_type,
    floor_dbfs=_MATERIAL_NOISE_FLOOR_MIN_DBFS.get(material_type, -55.0)
)
# VERBOTEN: Pausenzonen auf unter -70 dBFS abfallen lassen bei analog-Material
```

## §PEP Pre-Echo-Prevention (V22) [RELEASE_MUST v10.0.0]

**Regel**: Additive ML-Phasen (AudioSR, Harmonik-Extrapolation) können Transient-Onsets zeitlich verschieben — das zerstört den "Punch" einer Aufnahme.

```python
# PFLICHT nach phase_06, phase_07, phase_23 (additive ML-Phasen):
from backend.core.dsp.transient_guard import detect_transient_shifts

_ts = detect_transient_shifts(audio_pre, audio_post, sr)
if _ts.max_shift_ms > 2.0:
    _blend_reduction = min(_ts.max_shift_ms / 2.0, 1.0)
    audio_post = audio_pre * _blend_reduction + audio_post * (1.0 - _blend_reduction)
    logger.warning("pre_echo_guard: onset_shift=%.1fms > 2ms → blend_reduction=%.2f (V22)",
                   _ts.max_shift_ms, _blend_reduction)
    metadata["onset_shift_ms"] = _ts.max_shift_ms
# Non-blocking: Shift bis 2ms tolerierbar; über 2ms → proportionaler Blend
```

## §MKI Mono-Kompatibilität (V23) [RELEASE_MUST v10.0.0]

**Regel**: Viele Restaurierungen klingen in Stereo gut, kollabieren beim Mono-Sum (Radio, TV, Streaming-Normalisierung).

```python
# PFLICHT vor Export wenn Stereo + panns_singing >= 0.25:
from backend.core.dsp.stereo_guard import check_mono_compatibility

if audio.ndim == 2 and panns_singing >= 0.25:
    _mc = check_mono_compatibility(audio, sr)
    if _mc.phase_cancellation_db > 3.0:
        # Intensity-Stereo-Softening: Phasendifferenz reduzieren
        _mid = (audio[0] + audio[1]) * 0.5
        _side_blended = (audio[0] - audio[1]) * 0.5 * (1.0 - _mc.phase_cancellation_db / 12.0)
        audio = np.stack([_mid + _side_blended, _mid - _side_blended])
        metadata["mono_compatibility_warning"] = True
        metadata["phase_cancellation_db"] = _mc.phase_cancellation_db
        logger.warning("mono_compat_guard: cancellation=%.1f dB > 3 dB (V23)",
                       _mc.phase_cancellation_db)
# Non-blocking; niemals Veto
```

## §SCK Spektralfarbe (V24) [RELEASE_MUST v10.0.0]

**Regel**: Der EQ-Charakter einer Aufnahme ist Teil der künstlerischen Substanz. NR whitened subtil.

```python
# PFLICHT nach EQ/NR-Phasen:
from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

_scp = check_spectral_color_preservation(audio_pre, audio_post, sr)
# §P1-3 (2026-09-08, Hörordnung Ebene 2): Die feste Schwelle (0.97, auto-adaptiv
# bis 0.70) wird durch die lokale Maskierungs-JND des Phasen-Deltas begrenzt
# relaxiert (max. −0.20 bei voller 6-dB-JND, linear) — eine maskierte
# Spektralfarben-Abweichung ist unhörbar und löst keine Rücknahme aus.
if not _scp.ok:
    _strength_reduction = 0.30  # Strength 30 % reduzieren
    # Blend: audio_out = audio_pre * 0.3 + audio_post * 0.7
    audio_post = audio_pre * _strength_reduction + audio_post * (1.0 - _strength_reduction)
    logger.warning("spectral_color_guard: corr=%.3f → strength-0.30 (V24, §P1-3 Schwelle=%.3f)", _scp.correlation, getattr(_scp, "effective_threshold", 0.97))
    metadata["spectral_color_corr"] = _scp.correlation
# Messung: 1/3-Oktav-Energiekurve 200–8000 Hz, exkl. DefectScanner-Defektfrequenzen
```

## §WBG Wärmeband-Guard (V25) [RELEASE_MUST v10.0.0]

**Regel**: 200–800 Hz ist das „Wärme“-Band. Kumulativer Verlust durch mehrere NR/EQ-Phasen macht Aufnahmen „dünn“.

```python
# PFLICHT nach jeder Phase in Restoration-Modus:
from backend.core.dsp.warmth_guard import measure_warmth_band_delta

_wbd = measure_warmth_band_delta(audio_pre, audio_post, sr)
# §P1-3 (2026-09-08, Hörordnung Ebene 2): Der maskierte Anteil des aktuellen
# Phasen-Verlusts (lokale Maskierungs-JND im Wärmeband, max. 6 dB) zählt
# NICHT zum kumulativen Verlust — vorherige hörbare Verluste bleiben wirksam.
# In _restoration_context akkumulieren (nur der hörbare Anteil):
_audible_loss = max(0.0, _wbd.loss_db - float(getattr(_wbd, "masking_jnd_db", 0.0)))
_ctx["warmth_band_loss_db"] = _ctx.get("warmth_band_loss_db", 0.0) + _audible_loss

if _ctx["warmth_band_loss_db"] > 2.5:
    # Alle weiteren Phasen skalieren bis Verlust unter 2.5 dB
    _warmth_blend = float(1.0 - _ctx["warmth_band_loss_db"] / 5.0)
    _warmth_blend = max(0.0, min(1.0, _warmth_blend))
    audio_post = audio_pre * (1.0 - _warmth_blend) + audio_post * _warmth_blend
    logger.warning("warmth_guard: cum_loss=%.1f dB > 2.5 → blend=%.2f (V25)",
                   _ctx["warmth_band_loss_db"], _warmth_blend)
    metadata["warmth_band_loss_db"] = _ctx["warmth_band_loss_db"]
# Frequenzbereich: 200–800 Hz (Butter-Bandpass 4. Ordnung, zero-phase)
# VERBOTEN: Wärme-Verlust > 4 dB akkumuliert → Aufnahme klingt unnatürlich „radiogefärbt“
```

## §ATI Angriffstransienten-Integrität (V26) [RELEASE_MUST v10.0.0]

**Regel**: Die ersten 0–20 ms eines Phonem-Onsets sind perceptuell die wichtigsten Frames — sie bestimmen Punch, Klarheit und Artikulation.

```python
# PFLICHT nach NR/EQ-Phasen (alle Phasen die Energie verändern):
from backend.core.dsp.onset_guard import apply_onset_protection_mask

# onset_mask: bool-Array aus HPSS-Transient-Detektor (True = Onset-Frame)
audio_post = apply_onset_protection_mask(
    audio_pre=audio_pre,
    audio_post=audio_post,
    onset_mask=onset_mask,   # aus _restoration_context["onset_mask"]
    max_delta_db=1.5,        # feste Basis-Toleranz
    # §P1-3 (2026-09-08, Hörordnung Ebene 2): effektive Toleranz =
    # max(1.5 dB, lokale Maskierungs-JND des Phasen-Deltas, max. 6 dB) —
    # laute Transients maskieren die Abweichung, dann wird erst bei
    # größeren Ratios geblendet (weniger unnötige Dry-Wet-Blends).
)
# onset_mask: HPSS → Transient-Komponente → Frames mit Energie > Schwelle
# Onset-Fenster: 0–20 ms nach erstem Transient-Frame (Hörfenster für Attack)
# Strength-Cap in Onset-Frames: 0.15 (absolut)
# Non-blocking: Überschreitung blend-reduziert, kein Veto
```

## §WLPC Noise-Robuste Formant-Extraktion (V55) [RELEASE_MUST v10.0.0]

**Regel**: `lpc_formant_enhance()` und `_LPCFormantTracker.enhance()` MÜSSEN `era_decade` übergeben,
wenn der Aufrufkontext das Era-Jahrzehnt kennt — insbesondere bei `era_decade < 1960`
(Shellac, frühe elektrische Aufnahmen, Wachszylinder).

**Hintergrund**: Standard-Burg-LPC erkennt Rauschspitzen als Formanten wenn SNR < 15 dB
(typisch für pre-1960-Material). Das führt zu falschen Formant-Korrekturen und Stimm-Verfärbung.

```python
# PFLICHT-Aufruf in allen Phasen mit era_decade-Kontext:
from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker

# era_decade aus _restoration_context oder kwargs:
_era = kwargs.get("era_decade") or (kwargs.get("_restoration_context") or {}).get("era_decade")

audio_out = get_lpc_formant_tracker().enhance(
    audio, sr,
    era_decade=int(_era) if _era is not None else None,
    # Optional: snr_hint_db für noch präzisere SNR-Schätzung
)
# WLPC-Pfad aktiviert sich automatisch wenn:
#   era_decade < 1960  ODER  effective_snr < 15 dB
# Pre-Whitening (Wiener-Gain) AUSSCHLIESSLICH für LPC-Koeffizienten-Schätzung.
# Das Output-Audio wird NICHT modifiziert — §0 Primum non nocere.
```

**VERBOTEN**: `get_lpc_formant_tracker().enhance(audio, sr)` ohne `era_decade`
wenn era_decade < 1960 bekannt ist (V55). Standard-Burg-LPC → Rausch-Formanten als
Korrekturen angewendet → Stimm-Verfärbung in historischem Material.
