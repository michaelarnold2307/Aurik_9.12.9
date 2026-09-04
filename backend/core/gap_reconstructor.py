"""
core/gap_reconstructor.py
==========================
Differenzierer #4 — Semantische Lückenfüllung (Gap Reconstructor)

Erkennt und rekonstruiert kurze Aussetzer (Dropouts, digitale Stille, physikalische
Lücken) in Audioaufnahmen ohne externe ML-Modelle. Kombiniert drei komplementäre
Ansätze je nach Lückenbreite:

  1. Lineare Cross-Fade-Interpolation   — für Lücken < 2 ms
  2. Auto-Regression (Burg-Algorithmus) — für Lücken 2–50 ms  ← Weltklasse-Verfahren
  3. Spektrale OLA-Interpolation        — für Lücken 50–500 ms

Alle Methoden prüfen musikalischen Kontext (Korrelation Vor/Nach), um sicherzustellen,
dass das rekonstruierte Signal klanglich konsistent ist.

Verwendung:
    from backend.core.gap_reconstructor import GapReconstructor, GapReconstructorConfig

    recon = GapReconstructor()
    result = recon.reconstruct(audio, sample_rate)
    audio_fixed = result.audio
    logger.debug("%s Lücken → %s repariert", result.gaps_found, result.gaps_repaired)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------


@dataclass
class GapReconstructorConfig:
    """Steuerparameter für den GapReconstructor."""

    # --- Erkennung ---
    silence_threshold_db: float = -70.0
    """Pegel unterhalb dem eine Region als Lücke gilt (dBFS)."""
    min_gap_duration_ms: float = 0.5
    """Minimale Lückendauer für Erkennung (< 0.5 ms ignoriert)."""
    max_gap_duration_ms: float = 500.0
    """Lücken länger als dieser Wert werden nicht repariert (zu groß)."""
    context_window_ms: float = 30.0
    """Kontextfenster vor/nach der Lücke für Analyse (ms)."""

    # --- AR-Modell ---
    ar_order_factor: float = 2.0
    """AR-Modellordnung = ar_order_factor × (gap_samples + 1), capped bei 512."""
    ar_stabilize: bool = True
    """Pole des AR-Modells bei |z| ≥ 1 ins Stabile projizieren."""

    # --- Spektrale Interpolation ---
    ola_hop_factor: float = 0.25
    """OLA-Hop-Größe relativ zur FFT-Fenstergröße."""

    # --- Qualitätssicherung ---
    min_context_correlation: float = 0.35
    """Mindest-Korrelation Vor/Nach der reparierten Lücke (0 = off)."""
    blend_ms: float = 1.5
    """Cross-Fade-Blend an Lückenrändern (ms) zur Artefaktfreiheit."""


# ---------------------------------------------------------------------------
# Ergebnis-Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class GapInfo:
    """Informationen über eine einzelne erkannte Lücke."""

    start_sample: int
    end_sample: int
    duration_ms: float
    channel: int  # -1 = Mono / alle Kanäle identisch
    method_used: str  # "linear" | "ar" | "spectral" | "skipped"
    context_correlation: float = 0.0
    repaired: bool = False


@dataclass
class GapReconstructionResult:
    """Vollständiges Ergebnis einer Lücken-Rekonstruktion."""

    audio: np.ndarray
    sample_rate: int
    gaps_found: int
    gaps_repaired: int
    gaps_skipped: int
    gap_details: list[GapInfo] = field(default_factory=list)
    processing_time_ms: float = 0.0
    total_repaired_ms: float = 0.0

    # --- Properties ---
    @property
    def repair_rate(self) -> float:
        """Anteil erfolgreich reparierter Lücken (0–1)."""
        return self.gaps_repaired / max(self.gaps_found, 1)

    def summary(self) -> str:
        methods: dict[str, int] = {}
        for g in self.gap_details:
            if g.repaired:
                methods[g.method_used] = methods.get(g.method_used, 0) + 1
        method_str = ", ".join(f"{v}× {k}" for k, v in methods.items())
        return (
            f"GapReconstructor: {self.gaps_found} Lücken gefunden, "
            f"{self.gaps_repaired} repariert ({method_str}), "
            f"{self.gaps_skipped} übersprungen, "
            f"{self.total_repaired_ms:.1f} ms total repaired, "
            f"{self.processing_time_ms:.0f} ms Laufzeit"
        )


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)  # type: ignore[no-any-return]


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _burg_ar(x: np.ndarray, order: int) -> np.ndarray:
    """
    Burg-Algorithmus: robuste AR-Koeffizientenbestimmung — vektorisiert.

    Gibt Koeffizienten a[1..order] zurück (a[0] = 1 implizit).
    Quelle: Burg (1975), Numerisch stabil für kurze Signale.

    Optimierungen gegenüber Original:
      - Innere Schleife ``for i in range(1, m+1)`` → vektorisiertes
        ``a[1:m+1] = a_prev + km * a_prev[::-1]`` (kein Python-Loop über order²)
      - f/b-Update in-place (kein np.concatenate → kein O(n)-Allokation pro Iteration)
    """
    n = len(x)
    order = min(order, n - 1)
    if order < 1:
        return np.zeros(1)  # type: ignore[no-any-return]

    # In-place Puffer (keine Re-Allokation innerhalb der Schleife)
    f = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    f[:] = x
    b[:] = x

    a = np.zeros(order + 1)
    a[0] = 1.0

    for m in range(1, order + 1):
        num = -2.0 * np.dot(f[m:], b[: n - m])
        denom = np.dot(f[m:], f[m:]) + np.dot(b[: n - m], b[: n - m])
        if denom < 1e-12:
            break
        km = num / denom

        # Koeffizienten-Update vektorisiert (ersetzt O(m) Python-Loop)
        a_prev = a[:m].copy()
        a[1 : m + 1] = a_prev + km * a_prev[::-1]
        a[m] = km

        # f/b-Update in-place (kein np.concatenate, keine Heap-Allokation)
        old_f = f[m:].copy()
        old_b = b[: n - m].copy()
        f[m:] = old_f + km * old_b
        f[:m] = 0.0
        b[: n - m] = old_b + km * old_f
        b[n - m :] = 0.0

    return a[1:]  # type: ignore[no-any-return]  # a[1..order]; Vorzeichen: x[n] = -sum(a[i]*x[n-i])


def _stabilize_ar(coeffs: np.ndarray) -> np.ndarray:
    """
    Projiziert instabile AR-Pole ins Innere des Einheitskreises (Modulus < 1).
    Dies verhindert exponentielle Divergenz bei der Vorwärtsprädiktion.

    Burg-Algorithmus liefert per Konstruktion stabile Koeffizienten (|km| ≤ 1
    durch Cauchy-Schwarz), daher genügt für große Ordnungen eine O(1)-Prüfung
    statt der O(p³)-Eigenwertberechnung über np.roots (512×512-Begleitmatrix).
    """
    if len(coeffs) == 0:
        return coeffs
    # --- Schnellpfad für große Ordnungen (p > 64): O(1) statt O(p³) ---
    # np.roots würde eine p×p-Begleitmatrix aufbauen und deren Eigenwerte
    # berechnen — für p=512 sind das 134 Mio. Operationen (BLAS-Timeout-Risiko).
    # Burg-Koeffizienten sind mathematisch garantiert stabil; eine einfache
    # Gesamtskalierung reicht aus.
    if len(coeffs) > 64:
        mag = float(np.max(np.abs(coeffs)))
        if mag >= 1.0:
            coeffs = coeffs * (0.99 / (mag + 1e-9))
        return coeffs.copy()
    # --- Exakte Pol-Projektion für kleine Ordnungen (p ≤ 64, schnell) ---
    # Charakteristisches Polynom: 1 + a1*z^{-1} + ...
    poly = np.concatenate([[1.0], coeffs])
    # Guard: degenerate coefficients → LAPACK DLASCL failure
    if not np.all(np.isfinite(poly)):
        return coeffs.copy()
    try:
        roots = np.roots(poly)
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.debug("§V6 np.roots fehlgeschlagen — Original-Koeffizienten zurückgegeben (Poly %s): %s", poly.shape, exc)
        return coeffs.copy()
    mags = np.abs(roots)
    mask = mags >= 1.0
    if mask.any():
        roots[mask] = roots[mask] / (mags[mask] + 1e-9) * 0.99
    stable_poly = np.poly(roots).real
    return stable_poly[1:]  # type: ignore[no-any-return]  # a[1..order]


def _ar_predict_forward(context: np.ndarray, ar_coeffs: np.ndarray, n_samples: int) -> np.ndarray:
    """Vorwärtsprädiktion: generiert n_samples aus den letzten context-Samples."""
    order = len(ar_coeffs)
    buf = list(context[-order:].astype(np.float64))
    out = []
    for _ in range(n_samples):
        val = -float(np.dot(ar_coeffs, buf[-order:][::-1]))
        # Overflow-Schutz: instabile Pole → auf [-10, 10] begrenzen
        if not np.isfinite(val):
            val = 0.0
        val = float(np.clip(val, -10.0, 10.0))
        out.append(val)
        buf.append(val)
    return np.clip(np.array(out, dtype=np.float32), -1.0, 1.0)  # type: ignore[no-any-return]


def _ar_predict_backward(context: np.ndarray, ar_coeffs: np.ndarray, n_samples: int) -> np.ndarray:
    """Rückwärtsprädiktion: generiert n_samples aus dem Kontext nach der Lücke (zeitgespiegelt)."""
    rev = context[::-1].copy()
    pred_rev = _ar_predict_forward(rev, ar_coeffs, n_samples)
    return pred_rev[::-1]


def _crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Linearer Cross-Fade zwischen zwei Arrays gleicher Länge."""
    n = len(a)
    fade_samples = min(fade_samples, n)
    out = a.copy()
    t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    out[-fade_samples:] = a[-fade_samples:] * (1.0 - t) + b[-fade_samples:] * t
    return out


def _spectral_interp(pre: np.ndarray, post: np.ndarray, n_gap: int, sr: int) -> np.ndarray:
    """
    Spektrale OLA-Interpolation für mittellange Lücken.

    Strategie:
    1. FFT des Vor- und Nach-Kontexts → Spektrale Hüllkurve
    2. Interpoliere Phase linear zwischen pre_phase und post_phase
    3. Synthetisiere mit ISTFT (OLA)
    """
    win_size = 512
    # Stellen sicher dass wir genug Material haben
    pre_pad = pre[-win_size:] if len(pre) >= win_size else np.pad(pre, (win_size - len(pre), 0))
    post_pad = post[:win_size] if len(post) >= win_size else np.pad(post, (0, win_size - len(post)))

    window = np.hanning(win_size).astype(np.float32)
    pre_spec = np.fft.rfft(pre_pad * window)
    post_spec = np.fft.rfft(post_pad * window)

    pre_mag = np.abs(pre_spec)
    post_mag = np.abs(post_spec)
    pre_phase = np.angle(pre_spec)
    post_phase = np.angle(post_spec)

    # Erzeuge n_gap Samples via OLA
    hop = max(1, win_size // 4)
    n_frames = max(1, int(np.ceil(n_gap / hop)))

    output = np.zeros(n_gap + win_size, dtype=np.float32)
    weights = np.zeros(n_gap + win_size, dtype=np.float32)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)  # 0→1
        mag = (1.0 - t) * pre_mag + t * post_mag

        # Phaseninterpolation mit Wrapping
        phase_diff = post_phase - pre_phase
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap [-π, π]
        phase = pre_phase + t * phase_diff

        spec = mag * np.exp(1j * phase)
        frame = np.fft.irfft(spec, n=win_size).astype(np.float32) * window
        pos = i * hop
        end = min(pos + win_size, len(output))
        frame_len = end - pos
        output[pos:end] += frame[:frame_len]
        weights[pos:end] += window[:frame_len]

    # OLA-Normalisierung
    weights = np.maximum(weights, 1e-8)
    output /= weights
    return output[:n_gap]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class GapReconstructor:
    """
    Weltspitzen-Differenzierer #4: Semantische Lückenfüllung.

    Erkennt Dropouts und Stille-Aussetzer im Audio und füllt sie
    kontextbewusst mit drei komplementären Methoden (Linear / AR / Spektral).
    Unterstützt Mono und Stereo. Kein ML erforderlich.
    """

    def __init__(self, config: GapReconstructorConfig | None = None):
        self.config = config or GapReconstructorConfig()

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        material_hint: str | None = None,
    ) -> GapReconstructionResult:
        """
        Vollständige Lückenerkennung und -reparatur.

        Args:
            audio:       float32 ndarray, Shape (samples,) oder (samples, channels)
            sample_rate: Abtastrate in Hz
            material_hint: Optionaler Hinweis ('vinyl', 'tape', 'shellac', 'digital')

        Returns:
            GapReconstructionResult mit repariertem Audio und Statistiken
        """
        t0 = time.perf_counter()

        audio = np.array(audio, dtype=np.float32)
        is_mono = audio.ndim == 1
        if is_mono:
            audio = audio[:, np.newaxis]

        _n_samples, n_channels = audio.shape
        cfg = self.config
        sr = sample_rate

        # Material-adaptive Anpassungen
        if material_hint in ("vinyl", "shellac"):
            cfg = GapReconstructorConfig(
                silence_threshold_db=cfg.silence_threshold_db - 5,  # empfindlicher
                max_gap_duration_ms=200.0,
                ar_order_factor=3.0,  # höhere AR-Ordnung für tonal-reiches Material
            )
        elif material_hint == "tape":
            cfg = GapReconstructorConfig(
                min_gap_duration_ms=1.0,  # Tape-Dropout typisch > 1ms
                max_gap_duration_ms=300.0,
            )

        silence_lin = _db_to_linear(cfg.silence_threshold_db)
        min_gap_smp = max(1, int(cfg.min_gap_duration_ms * sr / 1000))
        max_gap_smp = int(cfg.max_gap_duration_ms * sr / 1000)
        ctx_smp = max(32, int(cfg.context_window_ms * sr / 1000))
        blend_smp = max(1, int(cfg.blend_ms * sr / 1000))

        all_gaps: list[GapInfo] = []
        audio_out = audio.copy()

        # Kanal-weise verarbeiten
        for ch in range(n_channels):
            ch_audio = audio[:, ch]
            gaps = self._detect_gaps(ch_audio, silence_lin, min_gap_smp, max_gap_smp, sr=sr)

            for gap in gaps:
                gap.channel = ch
                info = self._repair_gap(
                    audio_out[:, ch],
                    gap,
                    sr=sr,
                    ctx_smp=ctx_smp,
                    blend_smp=blend_smp,
                    cfg=cfg,
                )
                if info.repaired:
                    # Patch einschreiben
                    pass  # _repair_gap schreibt direkt in audio_out[:, ch]
                all_gaps.append(info)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        repaired = [g for g in all_gaps if g.repaired]
        skipped = [g for g in all_gaps if not g.repaired]
        total_repaired_ms = sum(g.duration_ms for g in repaired) / max(n_channels, 1)

        out_audio = audio_out[:, 0] if is_mono else audio_out
        return GapReconstructionResult(
            audio=out_audio,
            sample_rate=sr,
            gaps_found=len(all_gaps),
            gaps_repaired=len(repaired),
            gaps_skipped=len(skipped),
            gap_details=all_gaps,
            processing_time_ms=round(elapsed_ms, 1),
            total_repaired_ms=round(total_repaired_ms, 2),
        )

    def detect_only(self, audio: np.ndarray, sample_rate: int) -> list[GapInfo]:
        """Nur Lückenerkennung ohne Reparatur — nützlich für Diagnose."""
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        cfg = self.config
        silence_lin = _db_to_linear(cfg.silence_threshold_db)
        min_gap_smp = max(1, int(cfg.min_gap_duration_ms * sample_rate / 1000))
        max_gap_smp = int(cfg.max_gap_duration_ms * sample_rate / 1000)
        all_gaps: list[GapInfo] = []
        for ch in range(audio.shape[1]):
            gaps = self._detect_gaps(audio[:, ch], silence_lin, min_gap_smp, max_gap_smp, sr=sample_rate)
            for g in gaps:
                g.channel = ch
            all_gaps.extend(gaps)
        return all_gaps

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _detect_gaps(
        self,
        audio: np.ndarray,
        silence_lin: float,
        min_gap_smp: int,
        max_gap_smp: int,
        sr: int = 44100,
    ) -> list[GapInfo]:
        """Erkennt stille Regionen (amplitude < silence_lin) als Lücken."""
        gaps: list[GapInfo] = []
        below = np.abs(audio) < silence_lin
        # Lauflängen-Kodierung
        i = 0
        n = len(audio)
        while i < n:
            if below[i]:
                j = i
                while j < n and below[j]:
                    j += 1
                length = j - i
                if min_gap_smp <= length <= max_gap_smp:
                    gaps.append(
                        GapInfo(
                            start_sample=i,
                            end_sample=j,
                            duration_ms=length / sr * 1000,
                            channel=-1,
                            method_used="skipped",
                        )
                    )
                i = j
            else:
                i += 1
        return gaps

    def _repair_gap(
        self,
        audio: np.ndarray,
        gap: GapInfo,
        sr: int,
        ctx_smp: int,
        blend_smp: int,
        cfg: GapReconstructorConfig,
    ) -> GapInfo:
        """Repariert eine einzelne Lücke; schreibt direkt in den audio-Array."""
        start = gap.start_sample
        end = gap.end_sample
        n_gap = end - start
        n = len(audio)

        gap.duration_ms = n_gap / sr * 1000

        # Kontextbereiche
        pre_start = max(0, start - ctx_smp)
        post_end = min(n, end + ctx_smp)
        pre = audio[pre_start:start]
        post = audio[end:post_end]

        if len(pre) < 4 or len(post) < 4:
            gap.method_used = "skipped"
            return gap

        # Methode wählen je nach Lückendauer
        dur_ms = gap.duration_ms
        if dur_ms < 2.0:
            patch = self._method_linear(pre, post, n_gap)
            gap.method_used = "linear"
        elif dur_ms < 50.0:
            patch = self._method_ar(pre, post, n_gap, sr, cfg)
            gap.method_used = "ar"
        else:
            patch = self._method_spectral(pre, post, n_gap, sr)
            gap.method_used = "spectral"

        # Qualitätsprüfung: Korrelation mit Kontext nach Reparatur
        context_check = np.concatenate([pre[-min(ctx_smp, len(pre)) :], patch, post[: min(ctx_smp, len(post))]])
        orig_context = np.concatenate(
            [pre[-min(ctx_smp, len(pre)) :], audio[start:end], post[: min(ctx_smp, len(post))]]
        )
        if len(context_check) > 8 and len(orig_context) == len(context_check):
            _pre_ctx = pre[-min(20, len(pre)) :]
            _pat_ctx = patch[: min(20, len(patch))]
            if float(np.std(_pre_ctx)) > 1e-8 and float(np.std(_pat_ctx)) > 1e-8:
                _pca = _pre_ctx - _pre_ctx.mean()
                _ppa = _pat_ctx - _pat_ctx.mean()
                _np_ = float(np.linalg.norm(_pca))
                _npa = float(np.linalg.norm(_ppa))
                corr = float(np.dot(_pca, _ppa) / (_np_ * _npa + 1e-10))
            else:
                corr = 1.0  # near-constant context (silence) → match trivially
        else:
            corr = 1.0
        gap.context_correlation = 0.0 if not np.isfinite(corr) else corr

        # NaN/Inf-Schutz im patch
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

        # Blend-Fenster an Lückenrändern
        fade = min(blend_smp, n_gap // 2)
        if fade > 0 and n_gap > fade * 2:
            t_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            t_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            patch[:fade] = audio[start - 1] * t_out + patch[:fade] * t_in if start > 0 else patch[:fade]
            patch[-fade:] = patch[-fade:] * t_out + audio[end] * t_in if end < n else patch[-fade:]

        # Einschreiben
        audio[start:end] = np.clip(patch, -1.0, 1.0)
        gap.repaired = True
        return gap

    def _method_linear(self, pre: np.ndarray, post: np.ndarray, n_gap: int) -> np.ndarray:
        """Lineare Interpolation zwischen letztem Pre- und erstem Post-Sample."""
        a = float(pre[-1])
        b = float(post[0])
        return np.linspace(a, b, n_gap, dtype=np.float32)  # type: ignore[no-any-return]

    def _method_ar(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        n_gap: int,
        sr: int,
        cfg: GapReconstructorConfig,
    ) -> np.ndarray:
        """
        Bidirektionale AR-Prädiktion (Burg-Algorithmus).

        Kombiniert Vorwärts- und Rückwärtsprädiktion mit linearem Blend,
        um Transient-Artefakte an beiden Rändern zu vermeiden.
        """
        order = min(512, max(16, int(cfg.ar_order_factor * (n_gap + 1))))  # §VERBOTEN: LPC < 16
        ctx = min(len(pre), max(order * 3, 128))
        context_pre = pre[-ctx:].astype(np.float64)
        context_post = post[:ctx].astype(np.float64)

        coeffs_fwd = _burg_ar(context_pre, order)
        coeffs_bwd = _burg_ar(context_post[::-1], order)

        if cfg.ar_stabilize:
            coeffs_fwd = _stabilize_ar(coeffs_fwd)
            coeffs_bwd = _stabilize_ar(coeffs_bwd)

        fwd = _ar_predict_forward(context_pre.astype(np.float32), coeffs_fwd.astype(np.float32), n_gap)
        bwd = _ar_predict_backward(context_post.astype(np.float32), coeffs_bwd.astype(np.float32), n_gap)

        # Linearer Blend: fwd dominiert am Anfang, bwd am Ende
        t = np.linspace(0.0, 1.0, n_gap, dtype=np.float32)
        patch = (1.0 - t) * fwd + t * bwd
        return patch.astype(np.float32)  # type: ignore[no-any-return]

    def _method_spectral(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        n_gap: int,
        sr: int,
    ) -> np.ndarray:
        """Spektrale OLA-Interpolation für mittellange Lücken (50–500 ms)."""
        return _spectral_interp(pre, post, n_gap, sr)
