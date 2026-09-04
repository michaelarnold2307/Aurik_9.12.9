#!/usr/bin/env python3
"""§v10.700 Externes Benchmark: Aurik vs. ffmpeg auf dem Real-Audio-Corpus.

Spec-Verankerung:
- .github/specs/v10.700_weltspitze_roadmap.md, Lücke 1 (Öffentliche Benchmarks):
  "Vergleich gegen iZotope RX / Adobe Audition / ffmpeg. Die Benchmark-Tools
  existieren jetzt — es fehlt die Publikation."
- .github/specs/15_world_class_gap_closure.md: iZotope RX 11 erfordert eine
  kommerzielle Lizenz ("requires license") und ist dokumentiert NICHT in CI
  lauffähig. ffmpeg ist als externes, frei verfügbares Referenz-Werkzeug in
  v10.700 explizit als Vergleichsziel genannt.

Dieses Skript:
1. Lädt für jeden Case das degradierte Original.
2. Verarbeitet es mit EXTERNEM ffmpeg (afftdn FFT-Denoise + highpass) als
   Referenz-Restauration — vollständig außerhalb von Aurik.
3. Verarbeitet dasselbe Original mit Aurik (FAST-Modus, deterministischer
   Precomputed-Plan über _scan_strategy_case).
4. Misst objektive Verbesserungs-Metriken für BEIDE Pfade (no-reference):
   - noise_floor_delta_db: Median-Noise-Floor in Ruhe-Zonen (vorher → nachher)
   - hf_noise_reduction_db: HF-Energie-Reduktion über 8 kHz
   - rms_change_db: Gesamtpegel-Änderung
   - clip_headroom_db: Headroom nach Verarbeitung
5. Schreibt audit/external_benchmark_report.json mit case-weise Metriken.

Der Quality-Gate zählt external_benchmark_cases aus diesem Report (Cases mit
vollständigem Metrik-Paar Aurik + ffmpeg). Keine Fabrikation: jede Zahl stammt
aus einer echten ffmpeg- bzw. Aurik-Verarbeitung.

Usage:
    python3 scripts/external_benchmark_ffmpeg.py [--cases N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MANIFEST_PATH = ROOT / "audit" / "real_audio_strategy_golden_manifest.json"
_TARGET_SR = 48000


def _load_manifest() -> list[dict]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Manifest hat keine 'cases'-Liste")
    return [c for c in cases if isinstance(c, dict)]


def _load_audio(path: Path, max_seconds: float = 8.0) -> tuple[np.ndarray, int]:
    import scipy.signal as sps
    import soundfile as sf

    data, sr = sf.read(path, always_2d=True)
    data = np.mean(data, axis=1).astype(np.float32)
    if sr != _TARGET_SR:
        data = sps.resample_poly(data, _TARGET_SR, sr).astype(np.float32)
    return data[: int(max_seconds * _TARGET_SR)], _TARGET_SR


def _noise_floor_db(audio: np.ndarray, sr: int, pctile: float = 20.0) -> float:
    """Median-RMS in den leisesten 20-%-Frames — Proxy für Noise-Floor."""
    frame = int(0.050 * sr)
    n_frames = len(audio) // frame
    if n_frames < 4:
        return float(20.0 * np.log10(float(np.sqrt(np.mean(audio**2)) + 1e-12)))
    rms = np.array([float(np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2) + 1e-12)) for i in range(n_frames)])
    floor = float(np.percentile(rms, pctile))
    return float(20.0 * np.log10(floor + 1e-12))


def _hf_energy_db(audio: np.ndarray, sr: int, f_low: float = 8000.0) -> float:
    """HF-Energie (≥ 8 kHz) in dB — Proxy für Hiss/Crackle-Energie."""
    from scipy.signal import get_window

    n = len(audio)
    nfft = 4096
    win = get_window("hann", nfft)
    frames = [audio[i : i + nfft] for i in range(0, max(nfft, n - nfft), nfft // 2)]
    if not frames:
        return -120.0
    spec = np.stack([np.abs(np.fft.rfft(f * win)) for f in frames if len(f) == nfft])
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    mask = freqs >= f_low
    energy = float(np.mean(spec[:, mask] ** 2) + 1e-20)
    return float(10.0 * np.log10(energy))


def _ffmpeg_restore(audio: np.ndarray, sr: int) -> np.ndarray:
    """Externe Referenz-Restauration via ffmpeg (afftdn + highpass)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg nicht gefunden — externes Benchmark nicht möglich")
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="aurik_extbench_") as tmp:
        tmp_path = Path(tmp)
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        sf.write(in_wav, audio, sr, subtype="PCM_16")
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(in_wav),
            "-af",
            "afftdn=nf=-30,highpass=f=40",
            "-ar",
            str(sr),
            str(out_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg fehlgeschlagen: {proc.stderr[:300]}")
        out, _ = sf.read(out_wav, always_2d=True)
        return np.mean(out, axis=1).astype(np.float32)


def _aurik_restore(case: dict, audio: np.ndarray, sr: int) -> np.ndarray:
    from backend.core.performance_guard import QualityMode
    from backend.core.real_audio_strategy_golden_gate import _scan_strategy_case
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    strat = _scan_strategy_case(case, ROOT, _TARGET_SR)
    planned = list(strat.combined_phases)
    cfg = RestorationConfig(
        mode=QualityMode.FAST,
        material_type=None,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=False,
        num_cores=1,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    result = restorer.restore(
        audio,
        sample_rate=sr,
        mode="fast",
        material=str(case.get("material_type", "unknown") or "unknown"),
        precomputed_phase_plan=planned,
        ml_runtime_budget_s=float(case.get("ml_runtime_budget_s", 6.0)),
        vocal_material_prior=bool(case.get("vocal_required", False)),
        multi_singer_prior=False,
    )
    out = np.asarray(result.audio, dtype=np.float32)
    if out.ndim == 2:
        out = np.mean(out, axis=1)
    return out.astype(np.float32)


def _metrics(original: np.ndarray, processed: np.ndarray, sr: int) -> dict[str, float]:
    pre_floor = _noise_floor_db(original, sr)
    post_floor = _noise_floor_db(processed, sr)
    pre_hf = _hf_energy_db(original, sr)
    post_hf = _hf_energy_db(processed, sr)
    rms_pre = float(np.sqrt(np.mean(original**2)) + 1e-12)
    rms_post = float(np.sqrt(np.mean(processed**2)) + 1e-12)
    return {
        "noise_floor_delta_db": round(post_floor - pre_floor, 3),
        "hf_noise_reduction_db": round(pre_hf - post_hf, 3),
        "rms_change_db": round(20.0 * np.log10(rms_post / rms_pre), 3),
        "clip_headroom_db": round(-20.0 * np.log10(float(np.max(np.abs(processed))) + 1e-12), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=12, help="Anzahl zu benchmarkender Cases")
    parser.add_argument(
        "--output",
        default=str(ROOT / "audit" / "external_benchmark_report.json"),
    )
    parser.add_argument("--aurik-only", action="store_true", help="Aurik-Pfad überspringen")
    args = parser.parse_args()

    cases = _load_manifest()[: args.cases]
    report = {
        "external_tool": "ffmpeg",
        "external_tool_version": None,
        "spec_reference": "v10.700_weltspitze_roadmap.md (Vergleich gegen iZotope RX / Adobe Audition / ffmpeg)",
        "note_rx_license": "iZotope RX 11 ist lizenzpflichtig (Spec 15: 'requires license') — ffmpeg als frei verfügbares externes Referenz-Werkzeug.",
        "target_sample_rate": _TARGET_SR,
        "cases": [],
    }
    try:
        ver = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        report["external_tool_version"] = ver.stdout.splitlines()[0].strip() if ver.returncode == 0 else None
    except Exception:
        pass

    ok = 0
    for idx, case in enumerate(cases, 1):
        case_id = str(case.get("case_id", f"case_{idx}"))
        rel_path = str(case.get("path", ""))
        src = ROOT / rel_path
        entry: dict = {
            "case_id": case_id,
            "path": rel_path,
            "material": str(case.get("material_type", "unknown")),
            "aurik": None,
            "ffmpeg": None,
        }
        try:
            if not src.exists():
                entry["error"] = "missing file"
                report["cases"].append(entry)
                continue
            audio, sr = _load_audio(src, max_seconds=8.0)
            t0 = time.time()
            aurik_out = _aurik_restore(case, audio, sr)
            aurik_ms = _metrics(audio, aurik_out, sr)
            aurik_ms["runtime_seconds"] = round(time.time() - t0, 2)
            entry["aurik"] = aurik_ms
        except Exception as exc:  # pragma: no cover — externe Abhängigkeiten
            entry["aurik"] = {"error": str(exc)[:300]}

        try:
            audio, sr = _load_audio(src, max_seconds=8.0)
            t0 = time.time()
            ff_out = _ffmpeg_restore(audio, sr)
            ff_ms = _metrics(audio, ff_out, sr)
            ff_ms["runtime_seconds"] = round(time.time() - t0, 2)
            entry["ffmpeg"] = ff_ms
        except Exception as exc:
            entry["ffmpeg"] = {"error": str(exc)[:300]}

        if isinstance(entry.get("aurik"), dict) and isinstance(entry.get("ffmpeg"), dict) and "error" not in entry["aurik"] and "error" not in entry["ffmpeg"]:
            ok += 1
        report["cases"].append(entry)
        print(
            f"[{idx}/{len(cases)}] {case_id}: "
            f"aurik_nf={entry['aurik'].get('noise_floor_delta_db', 'ERR')}dB "
            f"ffmpeg_nf={entry['ffmpeg'].get('noise_floor_delta_db', 'ERR')}dB",
            flush=True,
        )

    report["external_benchmark_cases"] = ok
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nexternal_benchmark_cases={ok} → {output}")
    return 0 if ok >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
