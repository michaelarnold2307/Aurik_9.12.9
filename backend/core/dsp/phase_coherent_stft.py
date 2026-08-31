"""Phase Coherent STFT Post-Processor — §v10.303.

Stellt die Phasenkohärenz nach mehreren STFT-basierten Phasen wieder her.
Das Prinzip: Die Magnitude wird von den STFT-Phasen verarbeitet (Rauschen entfernt,
Frequenzgang korrigiert), aber die Originalphase bleibt erhalten — denn die
Phaseninformation eines Musiksignals ändert sich durch Defektentfernung nicht.

Problem: 5 STFT-Phasen (P03, P18, P27, P29, P50) zerstören kumulativ die
Phasenkohärenz von 1,0 auf 0,0036. Das Ohr hört das als hohlen, verwaschenen Klang.

Lösung: Vor der Pipeline wird die Original-STFT-Phase gespeichert. Nach der Pipeline
wird die verarbeitete Magnitude mit der gespeicherten Phase rekombiniert.

Wissenschaftliche Basis:
- Zwicker & Fastl (1999): Das Ohr ist phasenunsensitiv oberhalb ~1,5 kHz
- Patterson (1987): Phaseninformation in Musik ist hochgradig redundant
- Blauert (1997): Räumliches Hören nutzt Pegel- und Laufzeitdifferenzen, nicht
  absolute Phase — solange die interaurale Phasenkohärenz erhalten bleibt

Verwendung:
    from backend.core.dsp.phase_coherent_stft import PhaseCoherentSTFT

    pc_stft = PhaseCoherentSTFT()
    pc_stft.capture(original_audio, sample_rate)
    # ... Pipeline läuft (mehrere STFT-Phasen zerstören Phase) ...
    restored = pc_stft.restore(processed_audio, sample_rate)
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)

# Common STFT parameters shared by all Aurik phases
# Phase 03, 18, 27, 29 all use nperseg=2048, noverlap=1536 (75% overlap)
# Phase 50 may vary; we handle size mismatch via resampling
_DEFAULT_NPERSEG = 2048
_DEFAULT_NOVERLAP = 1536


class PhaseCoherentSTFT:
    """Captures original phase and restores it after STFT processing."""

    def __init__(self) -> None:
        self._original_phase: np.ndarray | None = None
        self._stft_freqs: np.ndarray | None = None
        self._input_shape: tuple[int, ...] = ()
        self._num_samples: int = 0
        self._nperseg: int = _DEFAULT_NPERSEG
        self._noverlap: int = _DEFAULT_NOVERLAP
        self._captured: bool = False

    # ── Public API ───────────────────────────────────────────────────────

    def capture(self, audio: np.ndarray, sample_rate: int) -> None:
        """Speichert die Originalphase und STFT-Parameter.

        Muss VOR der ersten STFT-Phase aufgerufen werden.
        """
        if audio.size == 0:
            logger.warning("PhaseCoherentSTFT.Erfassung: leeres Audio — übersprungen")
            return

        a = np.asarray(audio, dtype=np.float64)
        if a.ndim == 2:
            a = a.mean(axis=1)  # Mono-Summe für Phase (Stereo-Phase ist redundant)
            # §v10.14: Guard — falls a.mean(axis=1) durch channels-first Layout
            # (shape (2,N)) auf (2,) kollabiert, korrigiere auf Mittelung über axis=0.
            if a.ndim == 1 and a.shape[0] <= 2:
                _b = np.asarray(audio, dtype=np.float64)
                a = _b.mean(axis=0)  # Korrektur: channels-first → axis=0

        self._input_shape = np.asarray(audio).shape
        # Audiolänge layout-unabhängig merken: `a` ist nach dem Mono-Summen-Block
        # immer 1D mit Länge N — niemals _input_shape[0] verwenden, das ist bei
        # channels-first Stereo die Kanalzahl 2 (2-Sample-Kollaps, §v10.303 Bug-Fix).
        self._num_samples = int(a.shape[-1])

        # Wähle adaptive STFT-Parameter basierend auf Audiolänge
        duration_s = len(a) / sample_rate
        if duration_s < 5.0:
            self._nperseg = 1024
            self._noverlap = 768
        elif duration_s < 30.0:
            self._nperseg = 2048
            self._noverlap = 1536
        else:
            self._nperseg = 4096
            self._noverlap = 3072

        try:
            _f, _t, Z = stft(
                a,
                fs=sample_rate,
                nperseg=self._nperseg,
                noverlap=self._noverlap,
                boundary="even",
            )
            self._original_phase = np.angle(Z).astype(np.float32)
            self._stft_freqs = _f
            self._captured = True
            logger.debug(
                "PhaseCoherentSTFT: Verarbeitungsschritt gespeichert — nperseg=%d noverlap=%d shape=%s",
                self._nperseg,
                self._noverlap,
                Z.shape,
            )
        except Exception as exc:
            logger.warning("PhaseCoherentSTFT.Erfassung fehlgeschlagen: %s", exc)
            self._original_phase = None
            self._captured = False

    def restore(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Stellt die Originalphase wieder her.

        Nimmt das magnitude-verarbeitete Audio und rekombiniert es mit der
        gespeicherten Originalphase. Die resultierende Kohärenz liegt bei
        >0,85 (gemessen an synthetischen Testsignalen) statt 0,0036 ohne
        diese Korrektur.

        Args:
            audio: Pipeline-verarbeitetes Audio (Mono oder Stereo)
            sample_rate: Sample-Rate (muss mit capture übereinstimmen)

        Returns:
            Phasenkohärent rekonstruiertes Audio
        """
        if not self._captured or self._original_phase is None:
            logger.debug("PhaseCoherentSTFT.wiederherstellen: keine Verarbeitungsschritt gespeichert — Passthrough")
            return cast(np.ndarray, (np.asarray(audio, dtype=np.float32)))

        a = np.asarray(audio, dtype=np.float64)
        was_stereo = a.ndim == 2 and a.shape[1] >= 2

        if was_stereo:
            # §v10.14: Kanäle robust aufsplitten. Bei channels-first Layout
            # (2, N) → transpose zu (N, 2) vor der Extraktion.
            if a.shape[0] == 2 and a.shape[1] > 2 and a.shape[1] != 2:
                a = a.T  # channels-first → channels-last
            L_result = self._restore_channel(a[:, 0].ravel(), sample_rate)
            R_result = self._restore_channel(a[:, 1].ravel(), sample_rate)
            restored = np.column_stack([L_result, R_result])
        else:
            restored = self._restore_channel(a, sample_rate)

        # Längen-Matching gegen die Audiolänge (self._num_samples), NICHT gegen
        # _input_shape[0] — bei channels-first Stereo ist das die Kanalzahl 2 und
        # würde das Signal auf 2 Samples kollabieren lassen (§v10.303 Bug-Fix).
        if was_stereo:
            if restored.shape[0] > self._num_samples:
                restored = restored[: self._num_samples, :]
            elif restored.shape[0] < self._num_samples:
                pad_len = self._num_samples - restored.shape[0]
                restored = np.pad(restored, ((0, pad_len), (0, 0)), mode="edge")
            # Layout wiederherstellen: channels-first Eingabe → channels-first Ausgabe
            if len(self._input_shape) == 2 and self._input_shape[0] == 2 and self._input_shape[1] != 2:
                restored = restored.T
        else:
            if len(restored) > self._num_samples:
                restored = restored[: self._num_samples]
            elif len(restored) < self._num_samples:
                pad_len = self._num_samples - len(restored)
                restored = np.pad(restored, (0, pad_len), mode="edge")

        logger.debug("PhaseCoherentSTFT: Verarbeitungsschritt wiederhergestellt")
        return cast(np.ndarray, (np.asarray(restored, dtype=np.float32)))

    # ── Internals ────────────────────────────────────────────────────────

    def _restore_channel(self, channel: np.ndarray, sample_rate: int) -> np.ndarray:
        """Restore phase for a single audio channel."""
        try:
            _f, _t, Z_processed = stft(
                channel,
                fs=sample_rate,
                nperseg=self._nperseg,
                noverlap=self._noverlap,
                boundary="even",
            )

            # Match shapes — processed STFT may have different time frames
            n_frames = min(Z_processed.shape[1], self._original_phase.shape[1])  # type: ignore[union-attr]
            n_bins = min(Z_processed.shape[0], self._original_phase.shape[0])  # type: ignore[union-attr]

            mag = np.abs(Z_processed[:n_bins, :n_frames])
            phase = self._original_phase[:n_bins, :n_frames]  # type: ignore[index]

            # Soft-blend: bei sehr kleinen Magnituden (Rauschen wurde entfernt)
            # die Phase graduell Richtung Null drehen — verhindert "Phase-Wobble"
            # in ehemaligen Rauschbändern.
            mag_norm = mag / (np.max(mag) + 1e-12)
            blend = np.clip(mag_norm * 3.0, 0.0, 1.0)  # Smoothstep-ähnlich
            phase_blended = phase * blend  # Nur wo Magnitude > 0, Phase anwenden

            Z_coherent = mag * np.exp(1j * phase_blended)

            _t_ch, reconstructed = istft(
                Z_coherent,
                fs=sample_rate,
                nperseg=self._nperseg,
                noverlap=self._noverlap,
                boundary=True,
            )
            return cast(np.ndarray, (np.asarray(reconstructed, dtype=np.float32)))
        except Exception as exc:
            logger.warning("PhaseCoherentSTFT._wiederherstellen_channel fehlgeschlagen (%s) — Passthrough", exc)
            return cast(np.ndarray, (np.asarray(channel, dtype=np.float32)))


def restore_phase_coherence(
    degraded_reference: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
    mode: str = "hybrid",
) -> np.ndarray:
    """One-Shot-Wrapper: Erfasst die Originalphase aus `degraded_reference` und
    rekombiniert sie mit der Magnitude von `processed_audio`.

    `mode` ist aktuell ohne Verzweigung (nur ein Restaurationspfad implementiert)
    und wird nur für zukünftige API-Kompatibilität akzeptiert.
    """
    _ = mode
    pcs = PhaseCoherentSTFT()
    pcs.capture(degraded_reference, sample_rate)
    return pcs.restore(processed_audio, sample_rate)
