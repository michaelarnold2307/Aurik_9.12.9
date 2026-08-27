import importlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import scipy.signal as _sig
except ImportError:
    _sig = None

try:
    import soxr as _soxr_rs
except ImportError:
    _soxr_rs = None

# Ensure repository root is on sys.path when CLI is executed as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_bridge = importlib.import_module("backend.api.bridge")
build_export_quality_gate_payload = _bridge.build_export_quality_gate_payload
export_guard = _bridge.export_guard
get_audio_exporter_class = _bridge.get_audio_exporter_class
get_aurik_denker_instance = _bridge.get_aurik_denker_instance
get_load_audio_fn = _bridge.get_load_audio_fn
normalize_user_mode = _bridge.normalize_user_mode
run_pre_analysis = _bridge.run_pre_analysis
validate_export_quality = _bridge.validate_export_quality

_TARGET_SR = 48_000
_VALID_MODES = {"Restoration", "Studio 2026"}


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Lädt audio file via canonical bridge import cascade."""
    try:
        load_audio_file = get_load_audio_fn()
        loaded = load_audio_file(path, target_sr=None, mono=False, do_carrier_analysis=False)
        if not isinstance(loaded, dict) or loaded.get("audio") is None or loaded.get("sr") is None:
            raise RuntimeError(str((loaded or {}).get("error") or "Unbekannter Ladefehler"))
        audio = np.asarray(loaded["audio"], dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        return np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), int(loaded["sr"])
    except Exception as exc:
        raise RuntimeError(f"Audio konnte nicht geladen werden: {exc}") from exc


def _normalize_mode(mode: str) -> str:
    return normalize_user_mode(mode)  # type: ignore[no-any-return]


def _normalize_phase_strength_oracle_rollout(mode: str | None) -> str | None:
    """Normalisiert den optionalen Strength-Oracle-Rollout-Modus (off/pilot/all)."""
    if mode is None:
        return None
    raw = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "0": "off",
        "false": "off",
        "disabled": "off",
        "none": "off",
        "pilot": "pilot",
        "1": "all",
        "true": "all",
        "enabled": "all",
        "full": "all",
        "all": "all",
    }
    return aliases.get(raw)


def _rms_dbfs(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        # Normalize to (N, ch): channels-first (ch, N) → transpose
        if arr.shape[0] <= 2 and arr.shape[1] > 2:
            arr = arr.T
        arr = arr.mean(axis=1)  # downmix to mono

    # §2.45a-I: Gated RMS (frame-based, silence ignored).
    # Global RMS penalizes successful denoise/dereverb because removed noise floor
    # lowers average energy although musical loudness is preserved.
    arr64 = arr.astype(np.float64, copy=False)
    if arr64.size == 0:
        return -120.0

    frame = 2048
    n = int(arr64.shape[0])
    n_frames = max(1, n // frame)
    gated_chunks: list[np.ndarray] = []

    for i in range(n_frames):
        s = i * frame
        e = min(s + frame, n)
        ch = arr64[s:e]
        if ch.size == 0:
            continue
        ch_rms = float(np.sqrt(np.mean(ch * ch) + 1e-12))
        ch_db = 20.0 * np.log10(max(ch_rms, 1e-12))
        if ch_db > -50.0:
            gated_chunks.append(ch)

    # Tail frame
    tail_s = n_frames * frame
    if tail_s < n:
        ch = arr64[tail_s:]
        if ch.size > 0:
            ch_rms = float(np.sqrt(np.mean(ch * ch) + 1e-12))
            ch_db = 20.0 * np.log10(max(ch_rms, 1e-12))
            if ch_db > -50.0:
                gated_chunks.append(ch)

    if gated_chunks:
        gated = np.concatenate(gated_chunks)
        rms = float(np.sqrt(np.mean(gated * gated) + 1e-12))
    else:
        # Fallback for near-silent clips
        rms = float(np.sqrt(np.mean(arr64 * arr64) + 1e-12))

    return float(20.0 * np.log10(max(rms, 1e-12)))


def _compute_export_signal_signature(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Berechnet leichte Signal-Features für exportseitige Gate-Entscheidungen."""
    arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        if arr.shape[0] <= 2 and arr.shape[1] > 2:
            arr = arr.mean(axis=0)
        elif arr.shape[1] <= 2 and arr.shape[0] > 2:
            arr = arr.mean(axis=1)
        else:
            arr = arr.mean(axis=-1)
    elif arr.ndim != 1:
        arr = np.ravel(arr)

    if arr.size < 128:
        return {"crest_db": 0.0, "hf_ratio": 0.0, "transient_ratio": 0.0, "micro_dynamic_db": 0.0}

    arr64 = arr.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(arr64)) + 1e-12)
    rms = float(np.sqrt(np.mean(arr64 * arr64) + 1e-12))
    crest_db = float(20.0 * np.log10(max(peak / max(rms, 1e-8), 1e-8)))

    n_fft = int(min(16384, arr64.size))
    if n_fft >= 512:
        frame = arr64[:n_fft]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft))) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sr, 1))
        total_energy = float(np.sum(spectrum) + 1e-12)
        hf_energy = float(np.sum(spectrum[freqs >= 6000.0]))
        hf_ratio = float(np.clip(hf_energy / total_energy, 0.0, 1.0))
    else:
        hf_ratio = 0.0

    diff = np.abs(np.diff(arr64, prepend=arr64[0]))
    transient_thr = max(float(np.percentile(diff, 99.0)), 1e-6)
    transient_ratio = float(np.mean(diff > transient_thr))

    frame_len = 2048
    frame_rms_db: list[float] = []
    if arr64.size >= frame_len:
        for start in range(0, arr64.size - frame_len + 1, frame_len):
            chunk = arr64[start : start + frame_len]
            chunk_rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
            frame_rms_db.append(float(20.0 * np.log10(max(chunk_rms, 1e-12))))
    micro_dynamic_db = (
        float(max(0.0, float(np.percentile(frame_rms_db, 95)) - float(np.percentile(frame_rms_db, 5))))
        if frame_rms_db
        else 0.0
    )

    return {
        "crest_db": float(np.clip(crest_db, 0.0, 40.0)),
        "hf_ratio": hf_ratio,
        "transient_ratio": float(np.clip(transient_ratio, 0.0, 1.0)),
        "micro_dynamic_db": float(np.clip(micro_dynamic_db, 0.0, 60.0)),
    }


def _compute_export_gate_adjustment(
    material_key: str,
    signal_signature: dict[str, float],
    result_obj: object,
) -> tuple[float, float, str]:
    """Leitet adaptive Gate-Offsets ab: (qe_delta, pegel_delta_db, reason)."""
    historical = {
        "wax_cylinder",
        "shellac",
        "wire_recording",
        "lacquer_disc",
        "vinyl",
        "tape",
        "reel_tape",
        "cassette",
    }
    modern = {"cd_digital", "dat", "lossless", "aac", "mp3_high", "streaming"}

    crest_db = float(signal_signature.get("crest_db", 0.0))
    hf_ratio = float(signal_signature.get("hf_ratio", 0.0))
    transient_ratio = float(signal_signature.get("transient_ratio", 0.0))
    micro_dynamic_db = float(signal_signature.get("micro_dynamic_db", 0.0))

    risk = 0.0
    if material_key in historical:
        risk += 0.24
    if crest_db >= 18.0:
        risk += 0.12
    if transient_ratio >= 0.01:
        risk += 0.12
    if micro_dynamic_db >= 14.0:
        risk += 0.08
    if hf_ratio <= 0.02 and material_key in historical:
        risk += 0.06

    # Nutze Denker-Telemetrie wenn vorhanden (gleiche Intelligenzlinie).
    stage_notes = getattr(result_obj, "stage_notes", None)
    if isinstance(stage_notes, dict):
        profile = stage_notes.get("exzellenz_recovery_profile")
        if isinstance(profile, dict):
            risk += float(profile.get("preserve_signal", 0.0)) * 0.20

    risk = float(np.clip(risk, 0.0, 1.0))
    if material_key in modern and risk <= 0.15:
        return (0.02, -0.4, f"modern_stable(risk={risk:.2f})")
    if risk >= 0.45:
        return (-0.03, +0.8, f"fragile_or_transient_risk(risk={risk:.2f})")
    return (0.0, 0.0, f"neutral(risk={risk:.2f})")


def _resample_to_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 48 kHz with frontend-parity path (soxr HQ, fallback scipy)."""
    if sr == _TARGET_SR:
        return audio
    if _soxr_rs is not None:
        try:
            # Frontend parity: soxr HQ for deterministic quality alignment.
            out = _soxr_rs.resample(audio, sr, _TARGET_SR, quality="HQ")
            return np.asarray(out, dtype=np.float32)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("aurik_cli.py::_resample_to_48k fallback", exc_info=True)
    if _sig is not None:
        try:
            int(round(audio.shape[0] * _TARGET_SR / sr))
            out = _sig.resample_poly(audio, _TARGET_SR, sr, axis=0)
            return np.asarray(out, dtype=np.float32)  # type: ignore[no-any-return]
        except Exception as exc2:
            raise RuntimeError(
                "Interne 48-kHz-Normierung fehlgeschlagen. Ursache: Resampling konnte nicht ausgefuehrt werden. "
                "Loesung: soxr/scipy im Bundle sicherstellen oder Eingabedatei vorab auf 48 kHz konvertieren."
            ) from exc2
    raise RuntimeError(
        "Interne 48-kHz-Normierung fehlgeschlagen. Ursache: kein Resampler im Bundle. "
        "Loesung: soxr/scipy im Bundle sicherstellen oder Eingabedatei vorab auf 48 kHz konvertieren."
    )


def _as_samples_channels(audio: np.ndarray) -> np.ndarray:
    """Normalisiert Aurik-Audio auf soundfile-/Exporter-Layout ``(samples, channels)``."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] <= 2 and arr.shape[1] > 2:
        arr = arr.T
    return np.ascontiguousarray(arr)  # type: ignore[no-any-return]


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resampelt Audio von src_sr auf dst_sr mit soxr HQ (Fallback: scipy)."""
    if src_sr == dst_sr:
        return audio
    if _soxr_rs is not None:
        try:
            _out: np.ndarray = np.asarray(_soxr_rs.resample(audio, src_sr, dst_sr, quality="HQ"), dtype=np.float32)
            return _out
        except Exception:
            logger.warning("aurik_cli.py::_resample_audio fallback", exc_info=True)
    if _sig is not None:
        n_out = int(round(audio.shape[0] * dst_sr / src_sr))
        if audio.ndim == 1:
            _out1: np.ndarray = np.asarray(_sig.resample(audio, n_out), dtype=np.float32)
            return _out1
        _out2: np.ndarray = np.stack(
            [np.asarray(_sig.resample(audio[:, ch], n_out), dtype=np.float32) for ch in range(audio.shape[1])],
            axis=1,
        ).astype(np.float32)
        return _out2
    raise RuntimeError("soxr und scipy nicht verfügbar — Output-Resampling nicht möglich.")


def _export_audio_frontend_parity(
    result: object,
    output_path: str,
    restored_audio: np.ndarray,
    reference_audio: np.ndarray,
    logger: logging.Logger,
    bit_depth: int = 24,
    output_sr: int = _TARGET_SR,
) -> tuple[bool, list[str], dict[str, object]]:
    """Exportiert mit demselben Guard-/Gate-Vertrag wie das Frontend."""
    write_audio = export_guard(_as_samples_channels(restored_audio))
    reference_for_export = export_guard(_as_samples_channels(reference_audio))

    eq_passed, eq_warnings = validate_export_quality(result)
    eq_payload = build_export_quality_gate_payload(result)
    _ml_payload = eq_payload.get("musiclover", {}) if isinstance(eq_payload, dict) else {}
    _ml_stereo = _ml_payload.get("stereo_integrity", {}) if isinstance(_ml_payload, dict) else {}
    _ml_vocal = _ml_payload.get("vocal_integrity", {}) if isinstance(_ml_payload, dict) else {}
    _ml_temporal = _ml_payload.get("temporal_risk", {}) if isinstance(_ml_payload, dict) else {}
    _ml_goals = _ml_payload.get("musical_goals", {}) if isinstance(_ml_payload, dict) else {}
    _ml_decision = _ml_payload.get("decision_trace", {}) if isinstance(_ml_payload, dict) else {}
    _wcs_payload = (
        eq_payload.get("worldclass_composite_gate", {})
        if isinstance(eq_payload.get("worldclass_composite_gate", {}), dict)
        else {}
    )
    _result_metadata = getattr(result, "metadata", None)
    _hybrid_engineer_vector = (
        _result_metadata.get("hybrid_engineer_vector", {}) if isinstance(_result_metadata, dict) else {}
    )
    _hybrid_engineer_vector_json = json.dumps(
        {str(k): float(v) for k, v in _hybrid_engineer_vector.items()}
        if isinstance(_hybrid_engineer_vector, dict)
        else {},
        sort_keys=True,
        ensure_ascii=True,
    )
    _evidence_payload = (
        eq_payload.get("threshold_evidence", {}) if isinstance(eq_payload.get("threshold_evidence", {}), dict) else {}
    )
    _wcs_evidence = (
        _evidence_payload.get("worldclass_composite_gate", {})
        if isinstance(_evidence_payload.get("worldclass_composite_gate", {}), dict)
        else {}
    )
    _ml_mono_softened = False

    # Music-Lover Exportschutz: leichte Mid/Side-Softening-Korrektur bei Mono-Risikoflag.
    if (
        bool(_ml_stereo.get("mono_compatibility_warning", False))
        and isinstance(write_audio, np.ndarray)
        and write_audio.ndim == 2
        and write_audio.shape[1] >= 2
    ):
        _left = write_audio[:, 0].astype(np.float32)
        _right = write_audio[:, 1].astype(np.float32)
        _mid = 0.5 * (_left + _right)
        _side = 0.5 * (_left - _right)
        _side *= 0.92
        write_audio[:, 0] = np.clip(_mid + _side, -1.0, 1.0)
        write_audio[:, 1] = np.clip(_mid - _side, -1.0, 1.0)
        _ml_mono_softened = True
        logger.info("CLI Export-MonoGuard: leichte Stereo-Softening-Korrektur aktiv")

    if eq_warnings:
        for warning in eq_warnings:
            logger.warning("Export-Quality: %s", warning)
    if not eq_passed:
        # Diagnose-/A-B-Modus: expliziter, dokumentierter Override — das
        # Schutz-Gate bleibt im Normalbetrieb voll aktiv.
        if os.getenv("AURIK_EXPORT_OVERRIDE", "0") == "1":
            logger.warning(
                "Export-Quality-Gate mit AURIK_EXPORT_OVERRIDE=1 übersprungen "
                "(Diagnose/A-B-Modus — Ergebnis NICHT für Produktion): %s",
                "; ".join(eq_warnings) if eq_warnings else "unbekannt",
            )
        else:
            metadata = getattr(result, "metadata", None)
            if isinstance(metadata, dict):
                metadata["export_quality_gate_failed"] = True
                metadata["export_quality_gate_warnings"] = list(eq_warnings)
                metadata["export_blocked_by_quality_gate"] = True
            raise RuntimeError(
                "Export blockiert: Export-Quality-Gate nicht bestanden"
                + (f" ({'; '.join(eq_warnings)})" if eq_warnings else "")
            )

    export_metadata = {
        "quality_gate_passed": str(bool(eq_payload.get("passed", eq_passed))),
        "quality_gate_degradation_status": str(eq_payload.get("degradation_status", "ok")),
        "quality_gate_fail_reason": str(eq_payload.get("fail_reason", "")),
        "quality_gate_recovery_attempted": str(bool(eq_payload.get("recovery_attempted", False))),
        "quality_gate_best_possible_reached": str(bool(eq_payload.get("best_possible_reached", False))),
        "fallback_quality_floor_status": str(
            (eq_payload.get("fallback_quality_floor", {}) or {}).get("status", "passed")
        ),
        "quality_gate_profile": str(eq_payload.get("profile", "")),
        "quality_gate_material": str(eq_payload.get("material", "")),
        "quality_gate_preserve_signal": str(float(eq_payload.get("preserve_signal", 0.0) or 0.0)),
        "quality_gate_threshold_qe": str(
            float((eq_payload.get("thresholds", {}) or {}).get("quality_estimate", 0.0) or 0.0)
        ),
        "quality_gate_threshold_level_drop_db": str(
            float((eq_payload.get("thresholds", {}) or {}).get("level_drop_db", 0.0) or 0.0)
        ),
        "quality_gate_worldclass_score": str(float(_wcs_payload.get("wcs", 0.0) or 0.0)),
        "quality_gate_worldclass_threshold": str(float(_wcs_payload.get("threshold", 0.0) or 0.0)),
        "quality_gate_worldclass_passed": str(bool(_wcs_payload.get("passed", False))),
        "quality_gate_worldclass_profile": str(_wcs_payload.get("profile", "") or ""),
        "quality_gate_worldclass_artifact_veto": str(bool(_wcs_payload.get("artifact_veto", False))),
        "quality_gate_hybrid_engineer_vector": _hybrid_engineer_vector_json,
        "quality_gate_evidence_worldclass_source_class": str(_wcs_evidence.get("source_class", "") or ""),
        "quality_gate_evidence_worldclass_revalidate_by": str(_wcs_evidence.get("revalidate_by", "") or ""),
        "quality_gate_musiclover_vqi": str(float(_ml_vocal.get("vqi", 0.0) or 0.0)),
        "quality_gate_musiclover_sid": str(float(_ml_vocal.get("singer_identity_cosine", 0.0) or 0.0)),
        "quality_gate_musiclover_temporal_hotspots": str(int(_ml_temporal.get("hotspot_count", 0) or 0)),
        "quality_gate_musiclover_mono_warning": str(bool(_ml_stereo.get("mono_compatibility_warning", False))),
        "quality_gate_musiclover_remaining_goals": str(
            int((_ml_goals.get("remaining_count", 0) if isinstance(_ml_goals, dict) else 0) or 0)
        ),
        "quality_gate_musiclover_mono_softened": str(bool(_ml_mono_softened)),
        "quality_gate_musiclover_all_sota_real": str(bool(_ml_decision.get("all_sota_real", True))),
        "quality_gate_musiclover_sota_reason": str(_ml_decision.get("vocal_restoration_capability_status", "") or ""),
    }

    out_path = Path(output_path)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Output-Resampling wenn abweichende Ziel-Sample-Rate gewünscht (z.B. 44100 Hz für CD)
    write_sr = _TARGET_SR
    if output_sr != _TARGET_SR:
        try:
            write_audio = _resample_audio(write_audio, _TARGET_SR, output_sr)
            if reference_for_export is not None:
                reference_for_export = _resample_audio(reference_for_export, _TARGET_SR, output_sr)
            write_sr = output_sr
            logger.info("Output-Resampling: %d Hz → %d Hz (soxr HQ)", _TARGET_SR, output_sr)
        except Exception as _rs_exc:
            logger.warning("Output-Resampling fehlgeschlagen, exportiere mit %d Hz: %s", _TARGET_SR, _rs_exc)

    _SUBTYPE_MAP = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    _sf_subtype = _SUBTYPE_MAP.get(bit_depth, "PCM_24")

    audio_exporter_cls = get_audio_exporter_class()
    if audio_exporter_cls is not None and out_path.suffix.lower() in audio_exporter_cls.FORMATS:
        exporter = audio_exporter_cls()
        exporter.export(
            write_audio,
            write_sr,
            out_path,
            bit_depth=bit_depth,
            quality="veryhigh",
            metadata=export_metadata,
            normalize=False,
            reference_audio=reference_for_export,
        )
    else:
        tmp_path = str(out_path) + ".wav.tmp"
        try:
            sf.write(tmp_path, write_audio, write_sr, format="WAV", subtype=_sf_subtype)
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.debug("Temporäre Exportdatei konnte nicht entfernt werden: %s", tmp_path)

    if str(out_path) != output_path:
        logger.info("Ausgabepfad ohne Dateiendung wurde als WAV geschrieben: %s", out_path)
    return True, [str(w) for w in eq_warnings], eq_payload


def process_audio(
    input_path: str,
    output_path: str,
    verbose: bool = True,
    mode: str = "Restoration",
    phase_strength_oracle_rollout: str | None = None,
    bit_depth: int = 24,
    output_sr: int = _TARGET_SR,
    json_mode: bool = False,
    abx_mode: bool = False,
    dry_run: bool = False,
) -> object:
    """Verarbeitet eine Audiodatei über denselben Denker-/Exportpfad wie das Frontend."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("aurik_cli")

    mode = _normalize_mode(mode)
    if mode not in _VALID_MODES:
        logger.warning("Unbekannter Modus '%s' — verwende 'Restoration'.", mode)
        mode = "Restoration"

    rollout_mode = _normalize_phase_strength_oracle_rollout(phase_strength_oracle_rollout)
    if rollout_mode is None:
        rollout_mode = _normalize_phase_strength_oracle_rollout(os.getenv("AURIK_PHASE_STRENGTH_ORACLE_ROLLOUT"))
    if phase_strength_oracle_rollout is not None and rollout_mode is None:
        logger.warning(
            "Ungültiger Strength-Oracle-Rollout '%s' — verwende Pipeline-Default.",
            phase_strength_oracle_rollout,
        )

    if not os.path.exists(input_path):
        logger.error("Input-Datei nicht gefunden: %s", input_path)
        sys.exit(2)

    # ── 1. Audio laden ────────────────────────────────────────────────────────
    try:
        audio_raw, sr_raw = _load_audio(input_path)
    except RuntimeError as exc:
        logger.error("Fehler beim Laden der Datei: %s", exc)
        sys.exit(3)

    file_mb = os.path.getsize(input_path) / 1024 / 1024
    if verbose:
        logger.info("Datei: %s  (%.2f MB, %d Hz, %d Kanäle)", input_path, file_mb, sr_raw, audio_raw.shape[1])

    # ── 2. Auf 48 kHz resamplen (Aurik-kanonische SR) ─────────────────────────
    try:
        audio_48k = _resample_to_48k(audio_raw, sr_raw)
    except RuntimeError as exc:
        logger.error("Fehler bei der SR-Normierung: %s", exc)
        sys.exit(6)

    try:
        pre = run_pre_analysis(
            audio_native=audio_raw,
            sr_native=sr_raw,
            audio_48k=audio_48k,
            file_path=input_path,
            store_in_bridge_cache=True,
        )
    except Exception as exc:
        logger.error("Fehler in der Voranalyse: %s", exc)
        sys.exit(10)

    if verbose:
        logger.info("🔧 Starte AurikDenker — Modus: %s", mode)

    # ── 3. Kanonischer Einstiegspunkt: AurikDenker.denke() (Spec §2.2) ────────
    try:
        denker = get_aurik_denker_instance()
        # Quality-first policy: prefer full-quality execution over RT budget cuts.
        result = denker.denke(
            audio_48k,
            sr=_TARGET_SR,
            mode=mode,
            no_rt_limit=True,
            input_path=input_path,
            output_path=output_path,
            pre_analysis_result=pre,
            phase_strength_oracle_rollout=rollout_mode,
        )
    except Exception as exc:
        logger.error("Fehler in der Restaurierungspipeline: %s", exc)
        sys.exit(4)

    if verbose:
        logger.info(
            "✅ Verarbeitung abgeschlossen  ·  Material: %s  ·  Qualität: %.3f  ·  RT-Faktor: %.2f×",
            result.material,
            result.quality_estimate,
            result.rt_factor,
        )
        if result.warnings:
            for w in result.warnings:
                logger.warning("⚠ %s", w)
        if result.processing_note:
            logger.info("ℹ %s", result.processing_note)
        logger.info(
            "🎯 Musical Goals: %d/14 bestanden  ·  Phasen: %d",
            result.goals_passed,
            len(result.phases_executed),
        )
        # §G63: CD-Rauschprofil für Vorschau — Preview klingt jetzt wie Export
        try:
            from backend.api.bridge import inject_cd_noise_profile

            result.audio = inject_cd_noise_profile(result.audio, _TARGET_SR, bit_depth=bit_depth)
            logger.info("💿 CD-Rauschprofil für Vorschau angewendet — Vorschau = Export")
        except Exception:
            logger.debug("CD-Rauschprofil für Vorschau übersprungen")

    # ── 4. Export-Quality-Gate (§8.1 + §0c RELEASE_MUST) ───────────────────────
    # §0c: Bei fehlgeschlagenem End-Gate MUSS Aurik das bestmögliche sichere Ergebnis
    # exportieren. Hardstop ohne Ausgabedatei ist normativ unzulässig.
    # Status bleibt transparent (degraded). Kein stiller Abbruch.
    _export_degraded = False
    _export_degraded_reasons: list[str] = []

    # §2.54 Material-adaptive quality_estimate threshold: Physikalische Qualitätsdecke je
    # Material-Klasse — multi-generationale mp3_low-Ketten können strukturell keine CD-Qualität
    # erreichen. Hardcoded 0.55 erzeugt für legitime Restaurierungen immer 'degraded'-Export.
    _mat_str = str(getattr(result, "material", "") or "").lower()
    _mat_str = _mat_str.rsplit(".", maxsplit=1)[-1]  # Enum-Value (.mp3_low → mp3_low)
    _QE_THRESHOLD: dict[str, float] = {
        "mp3_low": 0.33,
        "mp3_high": 0.40,
        "aac": 0.38,
        "streaming": 0.38,
        "shellac": 0.35,
        "wax_cylinder": 0.32,
        "wire_recording": 0.34,
        "cassette": 0.38,
        "tape": 0.42,
        "reel_tape": 0.42,
        "vinyl": 0.45,
        "lacquer_disc": 0.42,
        "minidisc": 0.45,
        "cd_digital": 0.55,
        "dat": 0.55,
        "lossless": 0.55,
    }
    _qe_threshold = _QE_THRESHOLD.get(_mat_str, 0.45)
    _sig = _compute_export_signal_signature(result.audio, _TARGET_SR)
    _qe_delta, _pegel_delta, _gate_profile = _compute_export_gate_adjustment(_mat_str, _sig, result)
    _qe_threshold = float(np.clip(_qe_threshold + _qe_delta, 0.25, 0.70))
    _qe = getattr(result, "quality_estimate", None)
    if _qe is not None and _qe < _qe_threshold:
        _export_degraded = True
        _export_degraded_reasons.append(f"quality_estimate={_qe:.3f}<{_qe_threshold:.2f} (§8.1)")
        logger.warning(
            "§0c: quality_estimate=%.3f < %.2f (%s-Schwelle) — Export mit Status 'degraded'.",
            _qe,
            _qe_threshold,
            _mat_str or "default",
        )
    logger.info(
        "Export-Gate-Profil: %s · qe_threshold=%.2f · crest=%.1f dB · hf=%.3f · transient=%.4f",
        _gate_profile,
        _qe_threshold,
        _sig.get("crest_db", 0.0),
        _sig.get("hf_ratio", 0.0),
        _sig.get("transient_ratio", 0.0),
    )

    # P1/P2 Musical Goals — normativ erstrebenswert, kein Hard-Export-Stop (§0c)
    _P1_P2_THRESHOLDS = {
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "tonal_center": 0.95,
        "timbre_authentizitaet": 0.87,
        "artikulation": 0.85,
    }
    _goals = getattr(result, "musical_goals_scores", None) or {}
    _failed_goals = [f"{g}={_goals[g]:.3f}<{t}" for g, t in _P1_P2_THRESHOLDS.items() if g in _goals and _goals[g] < t]
    if _failed_goals:
        _export_degraded = True
        _export_degraded_reasons.append(f"P1/P2-Goals: {', '.join(_failed_goals)}")
        logger.warning(
            "§0c: P1/P2-Goals verfehlt (%s) — Export mit Status 'degraded'.",
            ", ".join(_failed_goals),
        )

    # ── 5. Ergebnis speichern ─────────────────────────────────────────────────
    restored = _as_samples_channels(result.audio)
    _in_db = _rms_dbfs(audio_48k)
    _out_db = _rms_dbfs(restored)
    _drop_db = _in_db - _out_db
    # §2.54 Material-adaptive Pegelabfall-Schwelle: Subtraktive Phasen (Denoise, Dereverb)
    # entfernen auf verlustbehafteten Trägern legitim mehr Rauschen als auf linearem Material.
    # Hardcoded 2.5 dB führt bei mp3_low/shellac immer zum False-Positive 'degraded'.
    _PEGEL_THRESHOLD: dict[str, float] = {
        "mp3_low": 5.0,
        "mp3_high": 4.0,
        "aac": 4.0,
        "streaming": 4.0,
        "shellac": 5.5,
        "wax_cylinder": 6.0,
        "wire_recording": 6.0,
        "cassette": 4.5,
        "tape": 4.5,
        "reel_tape": 4.0,
        "vinyl": 3.5,
        "lacquer_disc": 4.0,
        "cd_digital": 2.5,
        "dat": 2.5,
        "lossless": 2.5,
        "minidisc": 3.5,
    }
    _pegel_threshold = _PEGEL_THRESHOLD.get(_mat_str, 4.0)
    _pegel_threshold = float(np.clip(_pegel_threshold + _pegel_delta, 1.5, 8.5))
    _meta_obj = getattr(result, "metadata", None)
    _meta_dict = dict(_meta_obj) if isinstance(_meta_obj, dict) else {}
    _stage_notes = getattr(result, "stage_notes", None)
    _xp_profile = _stage_notes.get("exzellenz_recovery_profile", {}) if isinstance(_stage_notes, dict) else {}
    _preserve_signal = float(np.clip(float((_xp_profile or {}).get("preserve_signal", 0.0) or 0.0), 0.0, 1.0))
    _meta_dict.update(
        {
            "export_gate_profile": str(_gate_profile),
            "export_gate_material": str(_mat_str),
            "export_gate_preserve_signal": _preserve_signal,
            "export_gate_signal_signature": {
                "crest_db": float(_sig.get("crest_db", 0.0) or 0.0),
                "hf_ratio": float(_sig.get("hf_ratio", 0.0) or 0.0),
                "transient_ratio": float(_sig.get("transient_ratio", 0.0) or 0.0),
                "micro_dynamic_db": float(_sig.get("micro_dynamic_db", 0.0) or 0.0),
            },
            "export_gate_thresholds": {
                "quality_estimate": float(_qe_threshold),
                "level_drop_db": float(_pegel_threshold),
            },
        }
    )
    result.metadata = _meta_dict
    if _drop_db > _pegel_threshold:
        _export_degraded = True
        _export_degraded_reasons.append(f"Pegelabfall={_drop_db:.2f}dB>{_pegel_threshold:.1f}dB")
        logger.warning(
            "§0c: Pegelabfall %.2f dB > %.1f dB (%s-Schwelle) — Export mit Status 'degraded'.",
            _drop_db,
            _pegel_threshold,
            _mat_str or "default",
        )

    try:
        _eq_passed, _eq_warnings, _eq_payload = _export_audio_frontend_parity(
            result,
            output_path,
            restored,
            audio_48k,
            logger,
            bit_depth=bit_depth,
            output_sr=output_sr,
        )
    except Exception as exc:
        logger.error("Fehler beim Speichern der Audiodatei: %s", exc)
        sys.exit(5)

    if verbose:
        logger.info("📈 Pegel-Drift: in=%.2f dBFS out=%.2f dBFS delta=%.2f dB", _in_db, _out_db, -_drop_db)
        if _export_degraded or not _eq_passed:
            if not _export_degraded_reasons and _eq_warnings:
                _export_degraded_reasons.extend(_eq_warnings)
            elif not _export_degraded_reasons and isinstance(_eq_payload, dict):
                _export_degraded_reasons.append(str(_eq_payload.get("fail_reason", "Export-Quality-Gate")))
            logger.warning(
                "🟡 Export abgeschlossen (DEGRADED): %s — Grund: %s",
                output_path,
                " | ".join(_export_degraded_reasons),
            )
            logger.info("💾 Gespeichert (degraded): %s", output_path)
        else:
            logger.info("💾 Gespeichert: %s", output_path)

    # ── §T5.2: ABX Blindtest-Generator ──
    if abx_mode and result is not None:
        try:
            pass

            import soundfile as _abx_sf

            # Load original and restored audio
            orig_audio, orig_sr = _abx_sf.read(str(input_path))
            restored_audio, restored_sr = _abx_sf.read(str(output_path))

            # Match lengths and sample rates
            min_len = min(len(orig_audio), len(restored_audio))
            sr_use = min(orig_sr, restored_sr)

            # Generate 3 ABX snippets (10s each, random positions)
            snippet_dur = int(10 * sr_use)
            n_snippets = 3
            abx_data = []

            rng = np.random.RandomState(hash(str(input_path)) % (2**31))

            for idx in range(n_snippets):
                max_start = max(0, min_len - snippet_dur - int(2 * sr_use))
                start = rng.randint(int(2 * sr_use), max_start) if max_start > 0 else 0
                end = start + snippet_dur

                # X is randomly A (original) or B (restored)
                x_is_a = rng.random() > 0.5

                a_path = Path(output_path).parent / f"{Path(output_path).stem}_abx{idx + 1}_A.wav"
                b_path = Path(output_path).parent / f"{Path(output_path).stem}_abx{idx + 1}_B.wav"
                x_path = Path(output_path).parent / f"{Path(output_path).stem}_abx{idx + 1}_X.wav"

                _abx_sf.write(str(a_path), orig_audio[start:end], sr_use)
                _abx_sf.write(str(b_path), restored_audio[start:end], sr_use)
                _abx_sf.write(str(x_path), orig_audio[start:end] if x_is_a else restored_audio[start:end], sr_use)

                abx_data.append(
                    {
                        "snippet": idx + 1,
                        "start_s": float(start) / sr_use,
                        "duration_s": 10.0,
                        "x_is": "A" if x_is_a else "B",
                        "files": {
                            "A": str(a_path),
                            "B": str(b_path),
                            "X": str(x_path),
                        },
                    }
                )

            # Write ABX mapping (user can check after listening)
            mapping_path = Path(output_path).parent / f"{Path(output_path).stem}_abx_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump({"abx_snippets": abx_data, "note": "X=A (Original) oder X=B (Restauriert)?"}, f, indent=2)

            if verbose:
                logger.info("🔬 ABX-Blindtest: %d Snippets generiert → %s", n_snippets, mapping_path)
        except Exception as _abx_err:
            logger.warning("ABX-Generator: %s", _abx_err)

    return result


def print_usage():
    """Gibt die CLI-Hilfe aus."""
    print(
        "\nVerwendung: aurik_cli [--input PATH] [--output PATH] [--mode MODUS] [--bit-depth N] [--output-sr HZ] [--abx] [--dry-run] [--json] [-q] [-h]"
    )
    print("\nOptionen:")
    print("  --input, --input_audio PATH  Eingabe-Audiodatei")
    print("  --output, --output_audio PATH Ausgabe-Audiodatei")
    print("  --mode MODUS                 Restaurierungsmodus: 'Restoration' (Standard) oder 'Studio 2026'")
    print("  --bit-depth N                Bit-Tiefe: 16, 24 (Standard) oder 32 (float)")
    print("  --output-sr HZ               Ausgabe-Sample-Rate: 44100 oder 48000 (Standard)")
    print("  --dry-run                    Nur Pre-Analyse + Phasen-Plan, keine DSP-Verarbeitung")
    print("  --json                       Maschinenlesbare JSON-Ausgabe")
    print("  --abx                        A/B/X-Blindtest-Dateien nach Export generieren")
    print("  --progress                   Zeige Fortschrittsbalken während der Verarbeitung")
    print("  -q, --quiet                  Keine Fortschritts-Ausgaben")
    print("  -h, --help                   Diese Hilfe anzeigen")
    print()


def main():
    """Parst CLI-Argumente und startet die Aurik-Verarbeitung."""
    args = sys.argv[1:]
    verbose = True
    if "-q" in args or "--quiet" in args:
        verbose = False
        args = [a for a in args if a not in ["-q", "--quiet"]]

    if "-h" in args or "--help" in args:
        print_usage()
        sys.exit(0)

    input_file = None
    output_file = None
    mode = "Restoration"
    phase_strength_oracle_rollout: str | None = None
    bit_depth = 24
    output_sr = _TARGET_SR
    dry_run = False
    json_mode = False
    abx_mode = False
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--input_audio", "--input"):
            if i + 1 < len(args):
                input_file = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] in ("--input_audio", "--input"):
            input_file = arg.split("=", 1)[1]
        elif arg in ("--output_audio", "--output"):
            if i + 1 < len(args):
                output_file = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] in ("--output_audio", "--output"):
            output_file = arg.split("=", 1)[1]
        elif arg in ("--dry-run", "--json", "--abx"):
            if arg == "--dry-run":
                dry_run = True
            elif arg == "--json":
                json_mode = True
            elif arg == "--abx":
                abx_mode = True
            elif arg == "--progress" or arg == "--resume":
                pass
        elif arg == "--album":
            if i + 1 < len(args):
                args[i + 1]
                skip_next = True
        elif arg == "--mode":
            if i + 1 < len(args):
                mode = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] == "--mode":
            mode = arg.split("=", 1)[1]
        elif arg == "--bit-depth":
            if i + 1 < len(args):
                try:
                    _bd = int(args[i + 1])
                    if _bd in (16, 24, 32):
                        bit_depth = _bd
                    else:
                        print(f"❌ Ungültige Bit-Tiefe '{args[i + 1]}' — erlaubt: 16, 24, 32")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Ungültige Bit-Tiefe '{args[i + 1]}' — erlaubt: 16, 24, 32")
                    sys.exit(1)
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] == "--bit-depth":
            _bd_str = arg.split("=", 1)[1]
            try:
                _bd = int(_bd_str)
                if _bd in (16, 24, 32):
                    bit_depth = _bd
                else:
                    print(f"❌ Ungültige Bit-Tiefe '{_bd_str}' — erlaubt: 16, 24, 32")
                    sys.exit(1)
            except ValueError:
                print(f"❌ Ungültige Bit-Tiefe '{_bd_str}' — erlaubt: 16, 24, 32")
                sys.exit(1)
        elif arg == "--output-sr":
            if i + 1 < len(args):
                try:
                    _sr = int(args[i + 1])
                    if _sr in (44_100, 48_000):
                        output_sr = _sr
                    else:
                        print(f"❌ Ungültige Sample-Rate '{args[i + 1]}' — erlaubt: 44100, 48000")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Ungültige Sample-Rate '{args[i + 1]}' — erlaubt: 44100, 48000")
                    sys.exit(1)
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] == "--output-sr":
            _sr_str = arg.split("=", 1)[1]
            try:
                _sr = int(_sr_str)
                if _sr in (44_100, 48_000):
                    output_sr = _sr
                else:
                    print(f"❌ Ungültige Sample-Rate '{_sr_str}' — erlaubt: 44100, 48000")
                    sys.exit(1)
            except ValueError:
                print(f"❌ Ungültige Sample-Rate '{_sr_str}' — erlaubt: 44100, 48000")
                sys.exit(1)
    # Positional Fallback: nur Nicht-Flag-Argumente verwenden
    positional = [a for a in args if not a.startswith("-")]
    if input_file is None and len(positional) >= 1:
        input_file = positional[0]
    if output_file is None and len(positional) >= 2:
        output_file = positional[1]

    if not input_file or not output_file:
        print("❌ Fehler: Zu wenig oder ungültige Argumente\n")
        print_usage()
        sys.exit(1)

    # ── §QW2: Dry-Run Modus ──
    if dry_run:
        if not json_mode:
            print("🔍 DRY-RUN: Pre-Analyse + Phasen-Plan ohne DSP-Verarbeitung")
        # Nur Pre-Analyse durchführen
        try:
            from backend.api.bridge import get_defect_scanner

            audio_dry, sr_dry = sf.read(str(input_file))
            ScannerClass = get_defect_scanner()
            scanner = ScannerClass()
            analysis = scanner.scan(audio_dry, sr_dry)
            n_defects = sum(
                1
                for d in (analysis.scores.values() if isinstance(analysis.scores, dict) else [])
                if getattr(d, "severity", 0) > 0.5
            )
            duration_s = len(audio_dry) / sr_dry
            if json_mode:
                print(
                    json.dumps(
                        {
                            "status": "dry_run_complete",
                            "mode": mode,
                            "input": str(input_file),
                            "output": str(output_file),
                            "detected_material": getattr(analysis, "material_type", "unknown"),
                            "defect_count": n_defects,
                            "duration_s": duration_s,
                        },
                        indent=2,
                        default=str,
                    )
                )
            else:
                print(f"✅ Dry-Run abgeschlossen: {duration_s:.1f}s Audio, {n_defects} Defekte erkannt")
        except Exception as e:
            if json_mode:
                print(json.dumps({"status": "dry_run_failed", "error": str(e)}))
            else:
                print(f"❌ Dry-Run fehlgeschlagen: {e}")
            sys.exit(4)
        return

    # ── §QW2: JSON Mode ──
    if input_file and not os.path.exists(input_file):
        print(f'❌ Die Datei "{input_file}" wurde nicht gefunden.')
        print("   Bitte überprüfe den Pfad und versuche es erneut.")
        sys.exit(2)

    process_audio(
        input_file,
        output_file,
        verbose=verbose,
        mode=mode,
        phase_strength_oracle_rollout=phase_strength_oracle_rollout,
        bit_depth=bit_depth,
        output_sr=output_sr,
        json_mode=json_mode,
        abx_mode=abx_mode,
        dry_run=dry_run,
    )


# ── §v10 V6: Checkpoint & Resume ──
def save_pipeline_checkpoint(audio, phase_id, output_path, metadata=None):
    """Speichert Checkpoint nach jeder 5. Phase."""
    import pickle

    ckpt_dir = Path(output_path).parent / ".aurik_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{Path(output_path).stem}_{phase_id}.ckpt"
    try:
        data = {"audio": audio, "phase": phase_id, "metadata": metadata or {}}
        with open(ckpt_path, "wb") as f:
            pickle.dump(data, f, protocol=5)
        logger.debug("Checkpoint: %s", ckpt_path)
    except Exception as e:
        logger.debug("Checkpoint failed: %s", e)


def load_latest_checkpoint(output_path):
    """Lädt den neuesten Checkpoint für Resume."""
    import glob
    import os
    import pickle

    ckpt_dir = Path(output_path).parent / ".aurik_checkpoints"
    if not ckpt_dir.exists():
        return None
    files = sorted(glob.glob(str(ckpt_dir / "*.ckpt")), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            continue
    return None


if __name__ == "__main__":
    main()
