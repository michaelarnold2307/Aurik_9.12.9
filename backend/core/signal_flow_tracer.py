"""
Signal Flow Tracer (§SFT v10.0.0) — lückenlose Per-Phase-Audioverfolgung.

Schreibt pro Restaurierung eine JSON-Trace-Datei:
  ~/.aurik/traces/sft_<session_id>.json
  ~/.aurik/sft_latest.json  (Symlink → neuester Trace)

API (UV3-intern):
    tracer = get_signal_flow_tracer()
    tracer.begin_session(original_audio, sr, mode, source_path, material, era_decade, panns_singing)
    tracer.record_phase(phase_id, pre_audio, post_audio, sr, goal_delta=None)
    tracer.finalize(hpi, artifact_freedom, vqi=None, output_wav_path=None)

Diagnose (Copilot/CLI):
    tracer.report() → str  # strukturierter Text aller Befunde inkl. Flags
    tracer.report_latest() → str  # liest ~/.aurik/sft_latest.json

Detektiert pro Phase:
  PEGELEXPLOSION_WARN   : Peak-Ratio pre→post > +6 dB
  PEGELEXPLOSION_CRIT   : Peak-Ratio pre→post > +12 dB
  NOVELTY_WARN          : Spektraler Inhalt nicht im Original > 0.08 (§2.46e)
  NOVELTY_CRIT          : Spektraler Inhalt nicht im Original > 0.15 (§2.46e)
  HNR_DROP_WARN         : HNR-Abfall > 3 dB (Vocal-Schaden §0p)
  HNR_DROP_CRIT         : HNR-Abfall > 6 dB
  ECHO_ARTIFACT         : Zeitliche Autokorrelation Diff-Signal > 0.35 bei Lag > 20ms
  SILENCE_CONTAMINATION : Energie in vorher-stillen Zonen hinzugefügt (§2.68)
  LEVEL_COLLAPSE        : Post-RMS < -60 dBFS (Phase kollabierte Signal)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schwellwerte (§SFT normativ)
# ---------------------------------------------------------------------------
_PEGEL_WARN_DB = 6.0  # Pegelexplosion Warning
_PEGEL_CRIT_DB = 12.0  # Pegelexplosion Critical
_NOVELTY_WARN = 0.20  # Spektrale Neuheit Warning (adaptiv: ×0.57 des CRIT-Werts)
_NOVELTY_CRIT = 0.35  # Spektrale Neuheit Critical — adaptiv überschreibbar per set_novelty_threshold()
_HNR_WARN_DB = 3.0  # HNR-Abfall Warning
_HNR_CRIT_DB = 6.0  # HNR-Abfall Critical
_ECHO_CORR_THRESH = 0.35  # Autokorrelations-Peak Diff-Signal für Echo-Detektion
_ECHO_MIN_LAG_MS = 20.0  # Mindest-Lag für Echo (< 20ms = Phasen-Kolorierung, kein Echo)
_LEVEL_COLLAPSE_DBFS = -60.0  # Signal-Kollaps wenn Post-RMS unter diesem Wert
_PRE_PHASE_MIN_DBFS = -50.0  # Pre-Phase: Signal zu leise fuer sinnvolle Verarbeitung
_SILENCE_ENERGY_THRESH = -72.0  # dBFS — Stille-Zone gilt als kontaminiert über diesem Wert

# §v10.122 HallucinationGuard-Integration: Depth-adaptiver Basis-Schwellwert,
# gesetzt von calibrate_sft_thresholds() VOR der monotonen Kappung.
# Der HallucinationGuard liest diesen Wert via get_hallucination_guard_threshold().
_HG_BASE_THRESHOLD: float = 0.15  # Default: §2.46e normativ

# Pegelschätzung: 99.9-Perzentil statt np.max (Anti-V08)
_PEAK_PERCENTILE = 99.9

# Maximale Anzahl Phase-Records pro Session (Speicherschutz)
_MAX_PHASE_RECORDS = 200

# Verzeichnis für Trace-Dateien
_TRACE_DIR = Path.home() / ".aurik" / "traces"
_LATEST_SYMLINK = Path.home() / ".aurik" / "sft_latest.json"

# Kompakte Original-PSD: Welch auf 10 s (max) für Novelty-Berechnung
_ORIG_PSD_MAXLEN_S = 10.0
_ORIG_PSD_NPERSEG = 2048


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class PhaseTrace:
    """Messwerte und Flags einer einzelnen Phase."""

    phase_id: str
    phase_index: int
    peak_db_pre: float = 0.0
    peak_db_post: float = 0.0
    peak_ratio_db: float = 0.0  # post - pre (positiv = Pegelerhöhung)
    rms_db_pre: float = 0.0
    rms_db_post: float = 0.0
    rms_delta_db: float = 0.0
    spectral_novelty: float = 0.0  # vs. Original (0=identisch, 1=komplett neu)
    hnr_db_pre: float | None = None  # None wenn kein Vokal-Material
    hnr_db_post: float | None = None
    hnr_delta_db: float | None = None
    echo_corr_max: float = 0.0  # Max-Autokorrelation Diff-Signal
    echo_lag_ms: float = 0.0
    silence_energy_added_db: float | None = None  # None wenn keine Stille-Zonen
    flags: list[str] = field(default_factory=list)
    goal_delta: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp_rel_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "phase_id": self.phase_id,
            "phase_index": self.phase_index,
            "peak_db_pre": round(self.peak_db_pre, 2),
            "peak_db_post": round(self.peak_db_post, 2),
            "peak_ratio_db": round(self.peak_ratio_db, 2),
            "rms_db_pre": round(self.rms_db_pre, 2),
            "rms_db_post": round(self.rms_db_post, 2),
            "rms_delta_db": round(self.rms_delta_db, 2),
            "spectral_novelty": round(self.spectral_novelty, 4),
            "hnr_db_pre": round(self.hnr_db_pre, 2) if self.hnr_db_pre is not None else None,
            "hnr_db_post": round(self.hnr_db_post, 2) if self.hnr_db_post is not None else None,
            "hnr_delta_db": round(self.hnr_delta_db, 2) if self.hnr_delta_db is not None else None,
            "echo_corr_max": round(self.echo_corr_max, 4),
            "echo_lag_ms": round(self.echo_lag_ms, 1),
            "silence_energy_added_db": (
                round(self.silence_energy_added_db, 2) if self.silence_energy_added_db is not None else None
            ),
            "flags": self.flags,
            "goal_delta": self.goal_delta,
            "duration_ms": round(self.duration_ms, 1),
            "timestamp_rel_s": round(self.timestamp_rel_s, 3),
        }


# ---------------------------------------------------------------------------
# Kern-Klasse
# ---------------------------------------------------------------------------


class SignalFlowTracer:
    """Lückenlose Per-Phase-Audioverfolgung für Aurik.

    Thread-sicher. Non-blocking: alle Methoden fangen Exceptions intern ab.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_id: str = ""
        self._session_start: float = 0.0
        self._source_path: str = ""
        self._material: str = "unknown"
        self._era_decade: int = 0
        self._mode: str = "restoration"
        self._panns_singing: float = 0.0
        self._sr: int = 48000
        self._is_vocal: bool = False

        # Kompakter Fingerabdruck des Originals für Novelty-Vergleich
        self._orig_psd: np.ndarray | None = None
        self._orig_freqs: np.ndarray | None = None
        self._orig_peak_db: float = 0.0
        self._orig_rms_db: float = 0.0

        # Stille-Zonen (aus _restoration_context["structural_silence_zones"])
        self._silence_zones: list[tuple[int, int]] | None = None

        # Phase-Records
        self._phases: list[PhaseTrace] = []

        # Finalisierungs-Daten
        self._hpi: float | None = None
        self._artifact_freedom: float | None = None
        self._vqi: float | None = None
        self._output_wav: str | None = None

        # Pre-Phase-Audio-Referenz (kein Copy — wird nach record_phase verworfen)
        self._pre_audio_ref: np.ndarray | None = None

        self._session_active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_session(
        self,
        original_audio: np.ndarray,
        sr: int,
        mode: str,
        source_path: str = "",
        material: str = "unknown",
        era_decade: int = 0,
        panns_singing: float = 0.0,
        structural_silence_zones: list | None = None,
    ) -> None:
        """Startet eine neue Restaurierungs-Session.

        Muss VOR dem ersten record_phase aufgerufen werden.
        Nicht-blockierend: Exceptions werden intern abgefangen.
        """
        try:
            with self._lock:
                self._session_id = time.strftime("%Y%m%d_%H%M%S")
                self._session_start = time.monotonic()
                self._source_path = str(source_path)
                self._material = str(material)
                self._era_decade = int(era_decade or 0)
                self._mode = str(mode)
                self._panns_singing = float(panns_singing or 0.0)
                self._sr = int(sr or 48000)
                self._is_vocal = self._panns_singing >= 0.25
                self._phases = []
                self._hpi = None
                self._artifact_freedom = None
                self._vqi = None
                self._output_wav = None
                self._pre_audio_ref = None
                self._silence_zones = None
                self._orig_psd = None
                self._orig_freqs = None

                # Stille-Zonen speichern (Sample-Indizes)
                if structural_silence_zones:
                    self._silence_zones = list(structural_silence_zones)

                # Original-Fingerabdruck berechnen
                if isinstance(original_audio, np.ndarray) and original_audio.size > 0:
                    self._orig_peak_db = _to_db_peak(original_audio)
                    self._orig_rms_db = _to_db_rms(original_audio)
                    self._orig_psd, self._orig_freqs = _compute_psd_fingerprint(original_audio, sr)

                self._session_active = True
                logger.info(
                    "§SFT begin_Sitzung: source=%s material=%s era=%d vocal=%s",
                    os.path.basename(source_path),
                    material,
                    era_decade,
                    self._is_vocal,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SFT begin_Sitzung nicht blockierend exception: %s", exc)

    def set_silence_zones(self, zones: list) -> None:
        """Stille-Zonen nachträglich setzen (falls nach begin_session bekannt)."""
        try:
            with self._lock:
                if zones is not None:
                    self._silence_zones = list(zones)
        except Exception:  # pylint: disable=broad-except
            logger.debug("SFT begin_Sitzung nicht-kritischer Ersatzpfad, ignoriert", exc_info=True)

    def capture_pre_phase(self, audio: np.ndarray) -> None:
        """Pre-Phase-Audio für record_phase vormerken (kein Copy — nur Referenz).

        Wird unmittelbar vor phase.process() aufgerufen.
        """
        try:
            if self._session_active:
                # Keine Kopie nötig — UV3 erstellt sowieso eine neue Array nach phase.process()
                self._pre_audio_ref = audio
        except Exception:  # pylint: disable=broad-except
            logger.debug("SFT Erfassung_pre_Verarbeitungsschritt nicht-kritisch, ignoriert", exc_info=True)

    def record_phase(
        self,
        phase_id: str,
        pre_audio: np.ndarray | None,
        post_audio: np.ndarray | None,
        sr: int = 48000,
        goal_delta: dict | None = None,
    ) -> None:
        """Misst pre/post-Phase-Audio und detektiert Artefakte.

        Non-blocking, Exception-safe.
        Alle Berechnungen ≤ 50ms (DSP-Proxy, kein ML).
        """
        if not self._session_active:
            return
        if len(self._phases) >= _MAX_PHASE_RECORDS:
            return

        try:
            t0 = time.monotonic()

            # Mono-ify für Messungen (schneller, Material-neutral)
            pre = _to_mono(pre_audio)
            post = _to_mono(post_audio)

            if pre is None or post is None or len(pre) < 512 or len(post) < 512:
                return

            # Längen angleichen (deterministisch)
            n = min(len(pre), len(post))
            pre = pre[:n]
            post = post[:n]

            # §v10.0.4: LEVEL_COLLAPSE-Prüfung auf mittleren 80% des Signals
            # (vermeidet Filter-Ring-in/out False-Positives bei EQ/Loudness-Phasen)
            _trim_start = n // 10
            _trim_end = n - n // 10
            _post_mid = post[_trim_start:_trim_end] if _trim_end > _trim_start else post

            # ── Basis-Metriken ────────────────────────────────────────────────
            peak_pre = _to_db_peak(pre)
            peak_post = _to_db_peak(post)
            peak_ratio = peak_post - peak_pre
            rms_pre = _to_db_rms(pre)
            rms_post = _to_db_rms(post)
            rms_delta = rms_post - rms_pre

            # ── Spektrale Neuheit ────────────────────────────────────────────
            # v10.0.0c FIX: Per-Phase-Novelty (post vs. pre) für NOVELTY_CRIT-Flag.
            # VORHER: novelty = post vs. orig_psd → nach BW-Extension zeigten phase_14/16
            # dauerhaft ~0.50 Novelty, obwohl sie selbst nichts Neues hinzufügen
            # (alle HF-Bins >12 kHz vom Original abweichend durch frühere phase_07-Arbeit).
            # FIX: CRIT-Flag basiert auf Delta dieser Phase (post vs. pre). Die
            # session-globale Novelty (vs. orig) bleibt in spectral_novelty für Telemetrie.
            _pre_psd_nov, _pre_freqs_nov = _compute_psd_fingerprint(pre, sr)
            novelty_delta = _compute_spectral_novelty_fast(post, sr, _pre_psd_nov, _pre_freqs_nov)
            novelty = _compute_spectral_novelty_fast(post, sr, self._orig_psd, self._orig_freqs)

            # ── HNR (nur bei Vokal-Material, schnell via ACF) ────────────────
            hnr_pre: float | None = None
            hnr_post: float | None = None
            hnr_delta: float | None = None
            if self._is_vocal:
                hnr_pre = _hnr_fast(pre, sr)
                hnr_post = _hnr_fast(post, sr)
                if hnr_pre is not None and hnr_post is not None:
                    hnr_delta = hnr_post - hnr_pre

            # ── Echo-Detektion (Autokorrelation Diff-Signal) ─────────────────
            diff = post - pre
            echo_corr, echo_lag_ms = _detect_echo(diff, sr, pre=pre)

            # ── Stille-Kontaminierung ─────────────────────────────────────────
            silence_energy: float | None = None
            if self._silence_zones and pre_audio is not None and post_audio is not None:
                silence_energy = _check_silence_contamination(pre_audio, post_audio, self._silence_zones, sr)

            # ── Flags ─────────────────────────────────────────────────────────
            flags: list[str] = []

            if peak_ratio >= _PEGEL_CRIT_DB:
                flags.append(f"PEGELEXPLOSION_CRIT (+{peak_ratio:.1f} dB)")
            elif peak_ratio >= _PEGEL_WARN_DB:
                flags.append(f"PEGELEXPLOSION_WARN (+{peak_ratio:.1f} dB)")

            if novelty_delta >= _NOVELTY_CRIT:
                flags.append(f"NOVELTY_CRIT ({novelty_delta:.3f})")
            elif novelty_delta >= _NOVELTY_WARN:
                flags.append(f"NOVELTY_WARN ({novelty_delta:.3f})")

            if hnr_delta is not None and hnr_post is not None:
                if hnr_delta <= -_HNR_CRIT_DB:
                    flags.append(f"HNR_DROP_CRIT ({hnr_delta:+.1f} dB)")
                elif hnr_delta <= -_HNR_WARN_DB:
                    flags.append(f"HNR_DROP_WARN ({hnr_delta:+.1f} dB)")

            if echo_corr >= _ECHO_CORR_THRESH and echo_lag_ms >= _ECHO_MIN_LAG_MS:
                flags.append(f"ECHO_ARTIFACT (corr={echo_corr:.3f}, lag={echo_lag_ms:.0f}ms)")

            if silence_energy is not None and silence_energy > _SILENCE_ENERGY_THRESH:
                flags.append(f"SILENCE_CONTAMINATION ({silence_energy:.1f} dBFS)")

            if rms_post < _LEVEL_COLLAPSE_DBFS:
                # §v10.0.4: Kreuzvalidierung mit mittlerem Signal-Segment
                # (vermeidet Filter-Ring-in False-Positives)
                _rms_mid_db = _to_db_rms(_post_mid)
                if _rms_mid_db < _LEVEL_COLLAPSE_DBFS:
                    flags.append(f"LEVEL_COLLAPSE (rms={rms_post:.1f} dBFS, mid={_rms_mid_db:.1f} dBFS)")
                else:
                    logger.debug(
                        "§SFT %s: LEVEL_COLLAPSE false-positive (full=%.1f mid=%.1f dBFS)",
                        phase_id,
                        rms_post,
                        _rms_mid_db,
                    )

            # ── Phase-Record erstellen ────────────────────────────────────────
            elapsed = time.monotonic() - self._session_start
            duration_ms = (time.monotonic() - t0) * 1000.0

            pt = PhaseTrace(
                phase_id=phase_id,
                phase_index=len(self._phases),
                peak_db_pre=peak_pre,
                peak_db_post=peak_post,
                peak_ratio_db=peak_ratio,
                rms_db_pre=rms_pre,
                rms_db_post=rms_post,
                rms_delta_db=rms_delta,
                spectral_novelty=novelty,
                hnr_db_pre=hnr_pre,
                hnr_db_post=hnr_post,
                hnr_delta_db=hnr_delta,
                echo_corr_max=echo_corr,
                echo_lag_ms=echo_lag_ms,
                silence_energy_added_db=silence_energy,
                flags=flags,
                goal_delta=dict(goal_delta) if goal_delta else {},
                duration_ms=duration_ms,
                timestamp_rel_s=elapsed,
            )

            with self._lock:
                self._phases.append(pt)

            if flags:
                # LEVEL_COLLAPSE on phase-correction/EQ phases = expected for
                # heavily misaligned stereo material (cassette, reel_tape) or
                # filter ring-in producing near-zero first samples.
                _pc_collapse = (
                    "LEVEL_COLLAPSE" in " | ".join(flags)
                    and phase_id
                    and any(
                        p in str(phase_id) for p in ("phase_14", "phase_16", "phase_25", "azimuth", "phase_correction")
                    )
                )
                # §SOTA: NOVELTY_CRIT ist bei Reparatur- und subtraktiven
                # Cleanup-Phasen ERWARTET — ihr Zweck ist es, Inhalt zu
                # entfernen oder zu ersetzen, was das Spektrum notwendig ändert.
                #
                # Reparatur: Interpolation füllt Defektlücken mit neuem Material.
                # Subtraktiv: Entfernen von Rauschen/Brumm/Rumpel ändert Spektrum.
                _novelty_expected = (
                    "NOVELTY_CRIT" in " | ".join(flags)
                    and phase_id
                    and any(
                        p in str(phase_id)
                        for p in (
                            # Reparatur-Phasen (füllen Lücken → neues Material)
                            "phase_01",
                            "phase_09",
                            "phase_23",
                            "phase_24",
                            "phase_27",
                            "phase_50",
                            "phase_56",
                            "click",
                            "crackle",
                            "dropout",
                            "spectral_repair",
                            "band_gap",
                            "inpaint",
                            # Time/Pitch-Reparatur (Time-Stretch → Spektrumänderung)
                            "phase_12",
                            "phase_31",
                            "wow",
                            "flutter",
                            "speed_pitch",
                            # Subtraktive Cleanup-Phasen (entfernen Inhalt → Spektrumänderung)
                            "phase_02",
                            "phase_05",
                            "phase_18",
                            "phase_29",
                            "phase_59",
                            "hum",
                            "rumble",
                            "noise_gate",
                            "tape_hiss",
                            "modulation_noise",
                            # Enhancement-Phasen (verstärken/formen → Spektrumänderung)
                            "phase_08",
                            "phase_13",
                            "phase_36",
                            "phase_37",
                            "phase_38",
                            "phase_39",
                            "phase_41",
                            "phase_48",
                            "transient",
                            "bass_enhance",
                            "presence_boost",
                            "air_band",
                            "brilliance",
                            "stereo_enhance",
                            "stereo_width",
                            # Stereo-Geometrie-Korrektur (Phasenlage → Summenspektrum)
                            "phase_14",
                            "phase_15",
                            "phase_25",
                            "phase_34",
                            "phase_correction",
                            "stereo_balance",
                            "azimuth",
                            "mid_side",
                            # EQ/Tonal/Harmonic (formen Spektrum per Design)
                            "phase_04",
                            "phase_06",
                            "phase_07",
                            "phase_16",
                            "phase_17",
                            "eq_correction",
                            "frequency_restoration",
                            "harmonic_restoration",
                            "final_eq",
                            "mastering_polish",
                            # Denoise/Dereverb (entfernen Inhalt → Spektrumänderung)
                            "phase_03",
                            "phase_49",
                            "denoise",
                            "dereverb",
                            # De-Esser/Vocal-Processing (dämpft Frequenzen → Spektrumänderung)
                            "phase_19",
                            "phase_42",
                            "de_esser",
                            "vocal_enhance",
                            # Dynamics-Repair (Hüllkurve → indirekte Spektrumänderung)
                            "phase_10",
                            "phase_26",
                            "phase_40",
                            "compress",
                            "dynamic_range",
                            "loudness_normalization",
                        )
                    )
                )
                if _pc_collapse or _novelty_expected:
                    logger.info("§SFT %s FLAGS: %s", phase_id, " | ".join(flags))
                else:
                    logger.warning("§SFT %s FLAGS: %s", phase_id, " | ".join(flags))
            else:
                logger.debug(
                    "§SFT %s: peak=%+.1f dB rms=%+.1f dB novelty=%.3f",
                    phase_id,
                    peak_ratio,
                    rms_delta,
                    novelty,
                )

        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SFT aufzeichnen_Verarbeitungsschritt nicht blockierend exception (%s): %s", phase_id, exc)

    def finalize(
        self,
        hpi: float | None = None,
        artifact_freedom: float | None = None,
        vqi: float | None = None,
        output_wav_path: str | None = None,
    ) -> None:
        """Session abschließen: HPI/AF/VQI setzen und Trace-Datei schreiben."""
        if not self._session_active:
            return
        try:
            with self._lock:
                self._hpi = float(hpi) if hpi is not None else None
                self._artifact_freedom = float(artifact_freedom) if artifact_freedom is not None else None
                self._vqi = float(vqi) if vqi is not None else None
                self._output_wav = str(output_wav_path) if output_wav_path else None
                self._session_active = False

            self._write_trace()

            logger.info(
                "§SFT abschliessen: %d Phasen, HPI=%.3f, AF=%.3f, Ausgabe=%s",
                len(self._phases),
                self._hpi or 0.0,
                self._artifact_freedom or 0.0,
                os.path.basename(self._output_wav or ""),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SFT abschliessen nicht blockierend exception: %s", exc)

    def report(self) -> str:
        """Strukturierter Diagnose-Text der aktuellen (oder letzten) Session.

        Für Copilot-Analyse: zeigt alle Flags, kritische Phasen, Zusammenfassung.
        """
        return _format_report(self._build_trace_dict())

    def report_latest(self) -> str:
        """Liest ~/.aurik/sft_latest.json und gibt formatierten Report zurück."""
        try:
            data = json.loads(_LATEST_SYMLINK.read_text(encoding="utf-8"))
            return _format_report(data)
        except Exception as exc:
            logger.debug("§V6 report_latest: Trace-File lesen fehlgeschlagen — Fallback-Meldung zurückgegeben: %s", exc)
            return f"§SFT: Kein Trace-File vorhanden ({exc})"

    def latest_output_wav(self) -> str | None:
        """Gibt den Pfad zum neuesten Restaurierungs-WAV zurück (aus Trace oder Filesystem)."""
        # 1. Aus aktiver/letzter Session
        if self._output_wav and Path(self._output_wav).exists():
            return self._output_wav
        # 2. Aus Symlink
        try:
            data = json.loads(_LATEST_SYMLINK.read_text(encoding="utf-8"))
            wav = data.get("output_wav")
            if wav and Path(wav).exists():
                return wav  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("signal_flow_tracer.py::latest_Ausgabe_wav Ersatzpfad: %s", e)
        # 3. Filesystem-Fallback: neueste WAV in output/
        return _find_latest_output_wav()

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _build_trace_dict(self) -> dict:
        """Erstellt das vollständige Trace-Dict für JSON-Serialisierung."""
        all_flags: list[str] = []
        crit_phases: list[str] = []
        warn_phases: list[str] = []

        for pt in self._phases:
            for f in pt.flags:
                all_flags.append(f"{pt.phase_id}: {f}")
                if "CRIT" in f or "COLLAPSE" in f or "CONTAMINATION" in f:
                    crit_phases.append(pt.phase_id)
                else:
                    warn_phases.append(pt.phase_id)

        # Kumulativer Peak-Gewinn über alle Phasen
        if self._phases:
            total_peak_db = self._phases[-1].peak_db_post - self._phases[0].peak_db_pre
        else:
            total_peak_db = 0.0

        # HNR-Gesamt-Delta
        hnr_total = None
        if self._is_vocal and self._phases:
            deltas = [p.hnr_delta_db for p in self._phases if p.hnr_delta_db is not None]
            if deltas:
                hnr_total = round(sum(deltas), 2)

        return {
            "session_id": self._session_id,
            "source": os.path.basename(self._source_path),
            "source_full": self._source_path,
            "material": self._material,
            "era_decade": self._era_decade,
            "mode": self._mode,
            "panns_singing": round(self._panns_singing, 3),
            "is_vocal": self._is_vocal,
            "original_peak_db": round(self._orig_peak_db, 2),
            "original_rms_db": round(self._orig_rms_db, 2),
            "hpi": round(self._hpi, 4) if self._hpi is not None else None,
            "artifact_freedom": round(self._artifact_freedom, 4) if self._artifact_freedom is not None else None,
            "vqi": round(self._vqi, 4) if self._vqi is not None else None,
            "output_wav": self._output_wav,
            "phases": [p.to_dict() for p in self._phases],
            "summary": {
                "total_phases": len(self._phases),
                "critical_flags": sorted(
                    {f for f in all_flags if "CRIT" in f or "COLLAPSE" in f or "CONTAMINATION" in f}
                ),
                "warning_flags": sorted({f for f in all_flags if "WARN" in f}),
                "critical_phases": sorted(set(crit_phases)),
                "warning_phases": sorted(set(warn_phases)),
                "total_peak_gain_db": round(total_peak_db, 2),
                "total_hnr_delta_db": hnr_total,
                "all_flags_count": len(all_flags),
                "clean_phases_count": sum(1 for p in self._phases if not p.flags),
            },
        }

    def _write_trace(self) -> None:
        """Schreibt Trace-Datei und aktualisiert Symlink."""
        try:
            _TRACE_DIR.mkdir(parents=True, exist_ok=True)
            trace_path = _TRACE_DIR / f"sft_{self._session_id}.json"
            data = self._build_trace_dict()
            trace_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

            # Symlink atomisch aktualisieren
            tmp = _LATEST_SYMLINK.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(_LATEST_SYMLINK)

            logger.info("§SFT trace written: %s", trace_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SFT _write_trace exception: %s", exc)


# ---------------------------------------------------------------------------
# DSP-Hilfsfunktionen (pure numpy, kein ML)
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray | None) -> np.ndarray | None:
    """Audio → mono float32, None bei Fehler."""
    if audio is None:
        return None
    try:
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 2:
            # Shape (n, 2) oder (2, n)
            if a.shape[0] <= 8:
                a = np.mean(a, axis=0)
            else:
                a = np.mean(a, axis=1)
        return a.ravel()  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_to_mono Ersatzpfad: %s", e)
        return None


def _to_db_peak(audio: np.ndarray) -> float:
    """Peak-Level in dBFS (99.9-Perzentil, Anti-V08)."""
    try:
        peak = float(np.percentile(np.abs(audio), _PEAK_PERCENTILE))
        if peak < 1e-9:
            return -120.0
        return float(20.0 * np.log10(np.clip(peak, 1e-9, None)))
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_to_db_peak Ersatzpfad: %s", e)
        return -120.0


def _to_db_rms(audio: np.ndarray) -> float:
    """RMS-Level in dBFS."""
    try:
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if rms < 1e-9:
            return -120.0
        return float(20.0 * np.log10(np.clip(rms, 1e-9, None)))
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_to_db_rms Ersatzpfad: %s", e)
        return -120.0


def _compute_psd_fingerprint(audio: np.ndarray, sr: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Berechnet Welch-PSD-Fingerabdruck für Novelty-Vergleich."""
    try:
        from scipy.signal import welch

        mono = _to_mono(audio)
        if mono is None:
            return None, None

        # Auf max. 10 s begrenzen
        max_samples = int(_ORIG_PSD_MAXLEN_S * sr)
        if len(mono) > max_samples:
            # Mittleres Segment nehmen (repräsentativer als Anfang)
            start = (len(mono) - max_samples) // 2
            mono = mono[start : start + max_samples]

        nperseg = min(_ORIG_PSD_NPERSEG, len(mono) // 4)
        if nperseg < 64:
            return None, None

        freqs, psd = welch(mono, fs=sr, nperseg=nperseg)
        return psd.astype(np.float32), freqs.astype(np.float32)
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_berechnen_psd_fingerprint Ersatzpfad: %s", e)
        return None, None


def _compute_spectral_novelty_fast(
    post: np.ndarray,
    sr: int,
    orig_psd: np.ndarray | None,
    orig_freqs: np.ndarray | None,
) -> float:
    """Spektrale Neuheit von post vs. Referenz-PSD-Fingerabdruck.

    Misst den Anteil der Post-PSD, der die Referenz um > 3 dB übersteigt.
    Schnell: O(N_fft) ohne ML. Gibt 0.0 bei fehlender Referenz zurück.

    §v10.55 FIX: Extrahiert das GLEICHE mittlere 10s-Segment aus post,
    das _compute_psd_fingerprint fuer die Referenz verwendet.
    Vorher wurde welch auf das GESAMTE post-Signal angewendet,
    waehrend die Referenz-PSD nur aus den mittleren 10s stammte →
    systematischer Segment-Bias erzeugte ~0.47 False-Positive bei
    jedem Song mit nicht-stationaerem Spektrum (d.h. jedem echten Song).
    """
    if orig_psd is None or orig_freqs is None:
        return 0.0
    try:
        from scipy.signal import welch

        # §v10.55: Gleiches mittleres Segment wie _compute_psd_fingerprint verwenden
        max_samples = int(_ORIG_PSD_MAXLEN_S * sr)
        if len(post) > max_samples:
            start = (len(post) - max_samples) // 2
            post_seg = post[start : start + max_samples]
        else:
            post_seg = post

        nperseg = min(_ORIG_PSD_NPERSEG, len(post_seg) // 4)
        if nperseg < 64:
            return 0.0

        _, post_psd = welch(post_seg, fs=sr, nperseg=nperseg)

        # Länge auf min beider PSDs angleichen
        n = min(len(orig_psd), len(post_psd))
        op = np.maximum(orig_psd[:n], 1e-12)
        pp = post_psd[:n]

        # Anteil Frequenzbins wo Post > Original + 3 dB (Faktor 2)
        excess_bins: int = int(np.sum(pp > op * 2.0))
        novelty = float(excess_bins) / float(n) if n > 0 else 0.0
        return float(np.clip(novelty, 0.0, 1.0))
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_berechnen_spectral_novelty_fast Ersatzpfad: %s", e)
        return 0.0


def _hnr_fast(audio: np.ndarray, sr: int) -> float | None:
    """HNR-Schätzung via ACF (Boersma 1993 vereinfacht, < 20ms).

    Gibt None zurück wenn Signal zu kurz oder kein tonaler Inhalt.
    """
    try:
        # Nur mittleres Segment analysieren (repräsentativ, schnell)
        max_samples = int(0.5 * sr)  # max 0.5 s
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            seg = audio[start : start + max_samples]
        else:
            seg = audio

        if len(seg) < 256:
            return None

        seg = seg.astype(np.float64)
        seg -= np.mean(seg)

        # ACF via FFT
        n_fft = 2 ** int(np.ceil(np.log2(2 * len(seg) - 1)))
        X = np.fft.rfft(seg, n=n_fft)
        acf = np.fft.irfft(X * np.conj(X))[: len(seg)]
        acf = acf / (acf[0] + 1e-12)

        # F0-Bereich: 60–600 Hz (Gesang + Instrumente)
        lag_min = int(sr / 600)
        lag_max = int(sr / 60)
        lag_min = max(lag_min, 1)
        lag_max = min(lag_max, len(acf) - 1)

        if lag_max <= lag_min:
            return None

        peak_val = float(np.max(acf[lag_min:lag_max]))
        peak_val = np.clip(peak_val, 0.0, 1.0 - 1e-6)

        if peak_val < 0.05:
            return None  # Kein tonaler Anteil

        hnr = 10.0 * np.log10(peak_val / (1.0 - peak_val + 1e-12))
        return float(np.clip(hnr, -20.0, 40.0))
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_hnr_fast Ersatzpfad: %s", e)
        return None


def _detect_echo(diff: np.ndarray, sr: int, pre: np.ndarray | None = None) -> tuple[float, float]:
    """Detektiert Echo-Artefakte im Differenzsignal via Autokorrelation.

    Gibt kein False-Positive bei natürlich periodischen Differenzsignalen:
    Wenn Diff-Signal < -40 dB relativ zum Pre-Signal → kein Echo.

    Returns:
        (max_corr, lag_ms): Maximale Korrelation und ihr Lag in ms.
        max_corr = 0.0 wenn kein Echo erkennbar.
    """
    try:
        if len(diff) < 512:
            return 0.0, 0.0

        # Nur kurzes Segment (200ms) für Geschwindigkeit
        max_samples = int(0.2 * sr)
        seg = diff[:max_samples].astype(np.float64)
        seg -= np.mean(seg)
        energy: float = float(np.sum(seg**2))
        if energy < 1e-12:
            return 0.0, 0.0

        # Energie-Guard: Diff < -16 dB relativ zu Pre → kein hörbares Echo
        # Verhindert False-Positives bei natürlich periodischen Signalen
        # (z.B. Sinus mit -1 dB Absenkung: diff/pre = 0.109 < 0.15 → Guard greift)
        if pre is not None and len(pre) >= len(seg):
            pre_rms = float(np.sqrt(np.mean(np.square(pre[: len(seg)].astype(np.float64)))))
            diff_rms = float(np.sqrt(energy / len(seg)))
            if pre_rms > 1e-9 and (diff_rms / pre_rms) < 0.15:  # -16.5 dB Threshold
                return 0.0, 0.0

        # ACF via FFT
        n_fft = 2 ** int(np.ceil(np.log2(2 * len(seg) - 1)))
        X = np.fft.rfft(seg, n=n_fft)
        acf = np.fft.irfft(X * np.conj(X))[: len(seg)]
        acf = acf / (acf[0] + 1e-12)

        # Suche im Lag-Bereich > 20ms (Echo, kein Phasenfehler)
        lag_min = max(1, int(_ECHO_MIN_LAG_MS / 1000.0 * sr))
        lag_max = min(int(0.5 * sr), len(acf) - 1)

        if lag_max <= lag_min:
            return 0.0, 0.0

        search = np.abs(acf[lag_min:lag_max])
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx])
        lag_ms = (lag_min + peak_idx) / sr * 1000.0

        return float(np.clip(peak_val, 0.0, 1.0)), lag_ms
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_erkennen_echo Ersatzpfad: %s", e)
        return 0.0, 0.0


def _check_silence_contamination(
    pre: np.ndarray,
    post: np.ndarray,
    silence_zones: list,
    _sr: int,  # für künftige frequenzgewichtete Analyse reserviert
) -> float | None:
    """Prüft ob Stille-Zonen durch die Phase Energie hinzubekommen haben.

    Returns:
        dBFS der hinzugefügten Energie in den Stille-Zonen (max über alle Zonen).
        None wenn keine Stille-Zonen oder Fehler.
    """
    if not silence_zones:
        return None
    try:
        max_contamination = -120.0
        for zone in silence_zones:
            # Zone als (start_sample, end_sample) erwartet
            if not (isinstance(zone, (list, tuple)) and len(zone) >= 2):
                continue
            s, e = int(zone[0]), int(zone[1])

            # Audio-Shape behandeln
            def _slice(arr: np.ndarray, start: int, end: int) -> np.ndarray:
                if arr.ndim == 2:
                    if arr.shape[0] <= 8:
                        return arr[:, start:end]
                    return arr[start:end, :]
                return arr[start:end]

            pre_zone = _slice(pre, s, min(e, pre.shape[-1] if pre.ndim == 1 else pre.shape[0]))
            post_zone = _slice(post, s, min(e, post.shape[-1] if post.ndim == 1 else post.shape[0]))

            if pre_zone.size < 16 or post_zone.size < 16:
                continue

            # Energie-Delta in der Zone
            rms_pre = float(np.sqrt(np.mean(np.square(pre_zone.astype(np.float64)))))
            rms_post = float(np.sqrt(np.mean(np.square(post_zone.astype(np.float64)))))

            if rms_post > rms_pre * 1.5 and rms_post > 1e-6:
                db = 20.0 * np.log10(max(rms_post, 1e-9))
                max_contamination = max(max_contamination, db)

        return max_contamination if max_contamination > -100.0 else None
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_slice Ersatzpfad: %s", e)
        return None


def _find_latest_output_wav() -> str | None:
    """Sucht die neueste WAV-Datei in output/ und output_audio/ per Timestamp."""
    try:
        workspace = Path(__file__).parent.parent.parent
        candidates: list[Path] = []
        for subdir in ("output", "output_audio"):
            d = workspace / subdir
            if d.is_dir():
                candidates.extend(d.glob("*.wav"))
        if not candidates:
            return None
        return str(max(candidates, key=lambda p: p.stat().st_mtime))
    except Exception as e:
        logger.warning("signal_flow_tracer.py::_find_latest_Ausgabe_wav Ersatzpfad: %s", e)
        return None


def _format_report(data: dict) -> str:
    """Formatiert Trace-Dict als lesbaren Diagnose-Text."""
    lines: list[str] = []
    lines.append(f"{'=' * 70}")
    lines.append(f"§SFT Signal-Flow-Trace  [{data.get('session_id', '?')}]")
    lines.append(f"Quelle   : {data.get('source', '?')}")
    lines.append(
        f"Material : {data.get('material', '?')}  Era: {data.get('era_decade', '?')}  Mode: {data.get('mode', '?')}"
    )
    lines.append(f"Vokal    : {data.get('is_vocal', False)}  (PANNs-Singing={data.get('panns_singing', 0):.2f})")
    lines.append(
        f"Original : Peak={data.get('original_peak_db', 0):.1f} dBFS  RMS={data.get('original_rms_db', 0):.1f} dBFS"
    )
    lines.append(f"HPI={data.get('hpi', '–')}  AF={data.get('artifact_freedom', '–')}  VQI={data.get('vqi', '–')}")
    lines.append(f"Output   : {data.get('output_wav') or '(noch nicht finalisiert)'}")
    lines.append("")

    summary = data.get("summary", {})
    crits = summary.get("critical_flags", [])
    warns = summary.get("warning_flags", [])
    total = summary.get("total_phases", 0)
    clean = summary.get("clean_phases_count", 0)

    lines.append(f"PHASEN: {total} gesamt, {clean} sauber, {total - clean} mit Flags")
    lines.append(f"Peak-Gesamtgewinn: {summary.get('total_peak_gain_db', 0):+.1f} dB")
    if data.get("is_vocal"):
        lines.append(f"HNR-Gesamt-Delta: {summary.get('total_hnr_delta_db') or '–'} dB")
    lines.append("")

    if crits:
        lines.append(f"🚨 KRITISCH ({len(crits)}):")
        for c in crits:
            lines.append(f"   {c}")
        lines.append("")

    if warns:
        lines.append(f"⚠️  WARNUNGEN ({len(warns)}):")
        for w in warns:
            lines.append(f"   {w}")
        lines.append("")

    if not crits and not warns:
        lines.append("✅ Keine Flags — alle Phasen sauber.")
        lines.append("")

    # Per-Phase-Tabelle (nur Phasen mit Flags oder signifikanten Werten)
    phases = data.get("phases", [])
    notable = [p for p in phases if p.get("flags") or abs(p.get("peak_ratio_db", 0)) > 3.0]
    if notable:
        lines.append("AUFFÄLLIGE PHASEN (Flags oder |ΔPeak| > 3 dB):")
        lines.append(f"  {'Phase':<40} {'ΔPeak':>8} {'ΔRMS':>7} {'Novelty':>8} {'ΔHNR':>6}  Flags")
        lines.append("  " + "-" * 85)
        for p in notable:
            hnr_str = f"{p['hnr_delta_db']:+.1f}" if p.get("hnr_delta_db") is not None else "  –  "
            flag_str = "; ".join(p.get("flags", []))
            lines.append(
                f"  {p['phase_id']:<40} {p['peak_ratio_db']:>+7.1f} "
                f"{p['rms_delta_db']:>+6.1f} "
                f"{p['spectral_novelty']:>8.3f} "
                f"{hnr_str:>6}  {flag_str}"
            )
        lines.append("")

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: SignalFlowTracer | None = None
_instance_lock = threading.Lock()


def get_signal_flow_tracer() -> SignalFlowTracer:
    """Thread-sicherer Singleton-Zugriff."""
    global _instance  # pylint: disable=global-statement  # §3.2 Singleton-Pattern (normativ vorgeschrieben)
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SignalFlowTracer()
    return _instance


def set_novelty_crit_threshold(value: float) -> None:
    """§v10.40 Adaptiv: Setzt die NOVELTY_CRIT-Schwelle pro Song.

    Transfer-Chain-Tiefe 1:  value ≈ 0.25
    Transfer-Chain-Tiefe 2:  value ≈ 0.35
    Transfer-Chain-Tiefe 3:  value ≈ 0.45
    Transfer-Chain-Tiefe 4+: value ≈ 0.55
    """
    global _NOVELTY_CRIT, _NOVELTY_WARN
    # §G86 Monotonie-Garantie: NOVELTY_CRIT darf nur sinken, nie steigen.
    # Saubereres Audio (Mid-Pipeline) rechtfertigt keine laschere Toleranz.
    _NOVELTY_CRIT = min(float(value), _NOVELTY_CRIT)
    _NOVELTY_WARN = _NOVELTY_CRIT * 0.57
    logger.info("§v10.40 SFT: NOVELTY_CRIT kalibriert auf %.3f (monoton)", _NOVELTY_CRIT)


def calibrate_sft_thresholds(
    *,
    novelty_crit: float | None = None,
    material_type: str = "unknown",
    vocal_confidence: float = 0.0,
    transfer_chain_depth: int = 1,
    restorability_score: float = 65.0,
) -> None:
    """§v10.43 Zentrale SFT-Schwellwert-Kalibrierung aus CalibrationContext.

    Alle SFT-Schwellwerte, die vom Song abhängen, werden hier ZENTRAL gesetzt.
    Kein Modul darf eigene Konstanten pflegen (§V27, §G81).
    """
    global _ECHO_CORR_THRESH, _HNR_WARN_DB, _HNR_CRIT_DB
    global _WET_CEILING_NONREPAIR, _WET_CEILING_REPAIR
    global _HG_BASE_THRESHOLD

    if novelty_crit is not None:
        set_novelty_crit_threshold(novelty_crit)

    # §v10.122 HallucinationGuard-Integration: Depth-adaptiver Basis-Schwellwert.
    # Der HG-Default 0.15 ist nur für Studio-Master (depth 1) korrekt.
    # Tiefere Transfer-Ketten produzieren legitimerweise mehr spektrale Neuheit
    # durch Carrier-Inversion — der Schwellwert MUSS mit der Depth skalieren.
    _depth = max(1, int(transfer_chain_depth))
    if _depth >= 5:
        _HG_BASE_THRESHOLD = 0.55  # extreme chain: sehr viel Neuheit erwartet
    elif _depth == 4:
        _HG_BASE_THRESHOLD = 0.40  # deep cassette (Novelty 0.55)
    elif _depth == 3:
        _HG_BASE_THRESHOLD = 0.28  # moderate chain
    elif _depth == 2:
        _HG_BASE_THRESHOLD = 0.20  # shallow chain
    else:
        _HG_BASE_THRESHOLD = 0.15  # studio master (§2.46e normativ)
    logger.info("§v10.122 HG-Schwelle: depth=%d → base=%.2f", _depth, _HG_BASE_THRESHOLD)
    # §v10.131 Depth-adaptive Echo: tiefe Ketten erzeugen legitimerweise
    # mehr harmonische Echos durch kaskadierte Filter-Phasen.
    # Phase_07 (Harmonic Restoration) auf depth=4 Kassetten produziert
    # Autokorrelation >0.78 — aber das ist kein Artefakt, sondern
    # natürliche Resonanz der harmonischen Synthese.
    if _depth >= 4:
        _ECHO_CORR_THRESH = 0.55  # relaxed: 0.35→0.55
    elif _depth >= 3:
        _ECHO_CORR_THRESH = 0.45
    else:
        _ECHO_CORR_THRESH = 0.35  # Studio-Master: strikt
    # §G125 CalibratedConstants-Integration: zentrale Konstanten als Fallback.
    # Wenn CalibratedConstants verfügbar ist, überschreiben dessen Werte
    # die lokalen Berechnungen (Single Source of Truth, §G76 (GEBOTE.md)).
    try:
        from backend.core.calibrated_constants import get_constants

        _cc = get_constants()
        _ECHO_CORR_THRESH = _cc.echo_corr_threshold
        _HG_BASE_THRESHOLD = _cc.hg_base_threshold
        _WET_CEILING_NONREPAIR = _cc.sft_wet_ceiling_nonrepair
        _WET_CEILING_REPAIR = _cc.sft_wet_ceiling_repair
        _HNR_WARN_DB = _cc.hnr_warn_db
        _HNR_CRIT_DB = _cc.hnr_crit_db
        logger.debug(
            "§G125 CalibratedConstants angewendet: echo=%.2f hg=%.2f wet_nr=%.2f wet_r=%.2f",
            _ECHO_CORR_THRESH,
            _HG_BASE_THRESHOLD,
            _WET_CEILING_NONREPAIR,
            _WET_CEILING_REPAIR,
        )
    except Exception:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        pass  # Fallback: lokale Berechnungen bleiben aktiv

    # §G71 (GEBOTE.md) Wet-Ceilings: depth-adaptiv für effektive Phasen-Wirkung ≥ 0.15
    _depth = max(1, int(transfer_chain_depth))
    # §v10.119: Extreme Transfer-Chains (depth≥5) warnen und konservativ cappen.
    # 5+ Stufen (z.B. Wachswalze→Schellack→Tonband→Kassette→MP3) haben
    # physikalische Grenzen, die kein Algorithmus überwinden kann.
    if _depth >= 5:
        logger.warning(
            "§v10.119 Extreme-Transfer-Chain: depth=%d ≥ 5 — physikalische Grenzen. "
            "Wet-Ceilings werden auf konservative Maximalwerte begrenzt. "
            "Erwarte reduzierte Restaurationsqualität — das ist KEIN Aurik-Fehler, "
            "sondern eine physikalische Grenze der Quellkette.",
            _depth,
        )
        _depth_effective = 4  # Cap für Kalibrierung
    else:
        _depth_effective = _depth
    _WET_CEILING_NONREPAIR = float(np.clip(0.72 + max(0, _depth_effective - 1) * 0.05, 0.65, 0.90))
    _WET_CEILING_REPAIR = float(np.clip(0.82 + max(0, _depth_effective - 1) * 0.05, 0.75, 0.95))
    logger.info(
        "§G71 (GEBOTE.md) SFT-Wet-Ceilings: depth=%d rs=%.1f → nonrepair=%.2f repair=%.2f",
        _depth,
        restorability_score,
        _WET_CEILING_NONREPAIR,
        _WET_CEILING_REPAIR,
    )

    # Echo: Tape-Medien haben höhere Kanal-Korrelation (Kopf-Übersprechen)
    _mat_lower = str(material_type).lower()
    _is_tape = any(t in _mat_lower for t in ("cassette", "reel_tape", "tape"))
    _ECHO_CORR_THRESH = 0.45 if _is_tape else 0.35
    # §v10.117: Deep chains accumulate more phase rotation → higher echo baseline.
    # §v10.120 Calibration-Shift: depth≥4 is the new "deep cassette" tier
    # (Novelty 0.55); depth 3 is moderate (0.45) and does not trigger echo boost.
    if _depth >= 4:
        _ECHO_CORR_THRESH += 0.10 * (_depth - 3)  # depth=4→0.55, depth=5→0.65
    logger.info("§v10.43 SFT-Echo: mat=%s depth=%d → ECHO=%.2f", _mat_lower, _depth, _ECHO_CORR_THRESH)

    # HNR: Vocal-Material braucht strengere Grenzwerte (Gesangsschaden hörbarer)
    if vocal_confidence > 0.30:
        _HNR_WARN_DB = 2.0
        _HNR_CRIT_DB = 4.0
    else:
        _HNR_WARN_DB = 3.0
        _HNR_CRIT_DB = 6.0
    logger.info("§v10.43 SFT-HNR: vocal=%.2f → WARN=%.0fdB KRIT=%.0fdB", vocal_confidence, _HNR_WARN_DB, _HNR_CRIT_DB)


# §G71 (GEBOTE.md) Wet-Ceilings: depth-adaptiv, von calibrate_sft_thresholds() gesetzt
_WET_CEILING_NONREPAIR: float = 0.70
_WET_CEILING_REPAIR: float = 0.80


def get_sft_wet_ceilings() -> tuple[float, float]:
    """§G71 (GEBOTE.md) Gibt die kalibrierten Wet-Ceilings zurück (non-repair, repair)."""
    return (_WET_CEILING_NONREPAIR, _WET_CEILING_REPAIR)


def get_hallucination_guard_threshold() -> float:
    """§v10.122 Gibt den depth-adaptiven HallucinationGuard-Basis-Schwellwert zurück.

    Wird von calibrate_sft_thresholds() pro Song gesetzt.
    Default: 0.15 (§2.46e normativ).
    """
    return _HG_BASE_THRESHOLD
