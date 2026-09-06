#!/usr/bin/env python3
"""§v10.8xx benchmark_effizienz_matrix — Effizienz-Profil der UV3-Restaurierung.

Misst Wandzeit, RT-Faktor und Qualitäts-Kennzahlen derselben Audio-Sequenz
über eine Matrix aus Qualitätsmodi und Ebenen-Schaltern:

  Zellen (Defaults):
    fast       — QualityMode.FAST (8×-RT-Pfad)
    balanced   — QualityMode.BALANCED (32×-RT-Pfad, GUI-Default)
    maximum    — QualityMode.MAXIMUM (32×-RT-Pfad, volle Phasenmenge)
  Ebenen-Abschaltungen (nur Messung, kein Produktions-Eingriff):
    balanced_nopmg    — BALANCED ohne PMGG (§2.29, enable_phase_gate=False)
    balanced_adaptive — BALANCED mit RT-basiertem adaptiven Skipping
                        (enable_adaptive_skipping=True, opt-in)

Pro Zelle werden aufgezeichnet:
  * Wandzeit je Stufe via progress_callback (phase, pct) + Log-Zeitstempel
  * RSS-Verlauf (psutil oder /proc-Fallback, 2-s-Raster)
  * Ergebnis-Kennzahlen: rt_factor, quality_estimate, phases_executed/-skipped,
    PQS-MOS, HPI, Detected-Material
  * restauriertes Audio als WAV (PCM_24) für spätere Hör-/Qualitätsprüfung

Nutzung (aus dem Repo-Root, ROCm-Venv):
  python scripts/benchmark_effizienz_matrix.py --cells fast
  python scripts/benchmark_effizienz_matrix.py            # alle Zellen
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_effizienz")

OUT_ROOT = ROOT / "output_audio" / "benchmark_effizienz"


# --------------------------------------------------------------------------
# Zellendefinitionen
# --------------------------------------------------------------------------
@dataclass
class Cell:
    id: str
    quality_mode: str
    enable_phase_gate: bool = True
    enable_adaptive_skipping: bool = False


CELLS: list[Cell] = [
    Cell(id="fast", quality_mode="fast"),
    Cell(id="balanced", quality_mode="balanced"),
    Cell(id="maximum", quality_mode="maximum"),
    Cell(id="balanced_nopmg", quality_mode="balanced", enable_phase_gate=False),
    Cell(id="balanced_adaptive", quality_mode="balanced", enable_adaptive_skipping=True),
]

CELL_BY_ID = {c.id: c for c in CELLS}


# --------------------------------------------------------------------------
# RSS-Sampler (eigener Thread, 2-s-Raster)
# --------------------------------------------------------------------------
def _read_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


class RssSampler:
    def __init__(self, interval_s: float = 2.0) -> None:
        self._interval = interval_s
        self.samples: list[tuple[float, float | None]] = []
        self._stop = False

    def start(self) -> None:
        import threading

        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop:
            self.samples.append((time.monotonic() - t0, _read_rss_mb()))
            time.sleep(self._interval)

    def stop(self) -> None:
        self._stop = True


# --------------------------------------------------------------------------
# Eine Zelle ausführen
# --------------------------------------------------------------------------
def run_cell(
    cell: Cell,
    audio: np.ndarray,
    sr: int,
    log_path: Path,
    out_root: Path,
    save_wav: bool,
) -> dict[str, Any]:
    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    # Zellen-Log: INFO des Restorers wird hierhin gespiegelt (spätere
    # Phasen-Zeitanalyse aus den ▶/✅-Zeilen möglich).
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    try:
        cfg = RestorationConfig(
            mode=QualityMode(cell.quality_mode),
            enable_phase_gate=cell.enable_phase_gate,
            enable_adaptive_skipping=cell.enable_adaptive_skipping,
        )
        engine = UnifiedRestorerV3(cfg)

        progress: list[dict[str, float | str]] = []
        last_pct: dict[str, Any] = {"pct": -1}

        def _cb(pct: float, phase: str, elapsed_s: float) -> None:
            # Nur relevante Übergänge speichern (kein Spam bei identischem pct).
            if int(pct) != int(last_pct["pct"]) or phase != last_pct.get("phase"):
                progress.append(
                    {
                        "t": round(time.monotonic(), 3),
                        "pct": float(pct),
                        "phase": str(phase),
                        "elapsed_s": round(float(elapsed_s), 3),
                    }
                )
                last_pct["pct"] = pct
                last_pct["phase"] = phase

        sampler = RssSampler()
        sampler.start()
        t_wall0 = time.monotonic()
        try:
            result = engine.restore(
                audio,
                sample_rate=sr,
                progress_callback=_cb,
            )
        finally:
            sampler.stop()
        t_wall = time.monotonic() - t_wall0

        wav_path: str | None = None
        if save_wav:
            try:
                import soundfile as sf

                _out_wav = out_root / f"restored_{cell.id}.wav"
                _arr_out = np.asarray(result.audio, dtype=np.float32)
                if _arr_out.ndim == 2 and _arr_out.shape[0] == 2 and _arr_out.shape[1] != 2:
                    _arr_out = _arr_out.T  # (2,N)-intern → (N,2) für soundfile
                # Benchmark-Artefakt (kein Produkt-Export): PCM_24 ohne Dither-Pflicht,
                # da kein bit_depth < 32-Quantisierungsziel fürs menschliche Ohr.
                sf.write(str(_out_wav), _arr_out, sr, format="WAV", subtype="PCM_24")
                wav_path = str(_out_wav)
            except Exception as _wav_exc:
                logger.warning("WAV-Save für Zelle %s fehlgeschlagen: %s", cell.id, _wav_exc)

        pqs_mos: float | None = None
        try:
            pqs_mos = float(getattr(result.pqs_result, "mos", None) or 0.0) or None
        except Exception:
            pqs_mos = None

        meta = dict(result.metadata or {})
        # Vollständige Vocal-/Hör-Invarianten-Telemetrie sichern (nicht nur 60 Keys).
        _vocal_meta: dict[str, str] = {}
        for _mk, _mv in meta.items():
            _mk_l = _mk.lower()
            if any(
                _s in _mk_l
                for _s in ("vocal", "drive", "level_1", "einladung", "vqi", "sing", "breath")
            ):
                try:
                    _vocal_meta[_mk] = json.dumps(_mv, ensure_ascii=False, default=str)[:400]
                except Exception:
                    _vocal_meta[_mk] = str(_mv)[:400]
        return {
            "cell": cell.id,
            "quality_mode": cell.quality_mode,
            "enable_phase_gate": cell.enable_phase_gate,
            "enable_adaptive_skipping": cell.enable_adaptive_skipping,
            "wall_s": round(t_wall, 2),
            "audio_duration_s": round(float(len(audio) / sr), 3),
            "rt_factor": round(t_wall / max(float(len(audio) / sr), 1e-9), 2),
            "engine_total_s": round(float(getattr(result, "total_time_seconds", 0.0) or 0.0), 2),
            "engine_rt_factor": round(float(getattr(result, "rt_factor", 0.0) or 0.0), 2),
            "quality_estimate": round(float(result.quality_estimate or 0.0), 4),
            "material_detected": str(getattr(result, "material_type", "")),
            "phases_executed": int(len(result.phases_executed or [])),
            "phases_skipped": int(len(result.phases_skipped or [])),
            "deferred_phases": list(result.deferred_phases or []),
            "pqs_mos": pqs_mos,
            "hpi": meta.get("hpi", None),
            "artifact_freedom": meta.get("artifact_freedom", None),
            "n_progress_events": len(progress),
            "progress": progress,
            "rss_samples": sampler.samples,
            "meta_keys": sorted(meta.keys())[:60],
            "vocal_meta": _vocal_meta,
            "warnings": list(getattr(result, "warnings", []) or [])[:10],
            "error": None,
            "wav_path": wav_path,
        }
    except Exception as exc:  # Zellenfehler protokollieren statt Matrix abzubrechen
        logger.exception("Zelle %s fehlgeschlagen", cell.id)
        return {"cell": cell.id, "quality_mode": cell.quality_mode, "error": repr(exc)}
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def load_clip(path: Path, seconds: float) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2 and audio.shape[1] == 2:
        pass  # Stereo belassen
    elif audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    # Kein Resample nötig: die Engine konvertiert intern auf 48 kHz.
    n = int(seconds * sr)
    if len(audio) > n:
        audio = audio[:n]
    return audio, sr


def main() -> None:
    ap = argparse.ArgumentParser(description="UV3-Effizienz-Matrix-Benchmark")
    ap.add_argument(
        "--input",
        default=str(ROOT / "test_audio" / "_elke_60s_excerpt.wav"),
        help="Eingabe-WAV (degradiert, real)",
    )
    ap.add_argument("--seconds", type=float, default=30.0, help="zu verarbeitende Länge in s")
    ap.add_argument(
        "--cells",
        nargs="+",
        default=[c.id for c in CELLS],
        help="Zellen: " + ", ".join(CELL_BY_ID),
    )
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--no-wav", action="store_true", help="kein Ergebnis-WAV speichern")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [CELL_BY_ID[cid] for cid in args.cells if cid in CELL_BY_ID]
    if not cells:
        sys.exit(f"Keine gültige Zelle. Verfügbar: {', '.join(CELL_BY_ID)}")

    clip_path = Path(args.input)
    if not clip_path.exists():
        sys.exit(f"Input existiert nicht: {clip_path}")

    audio, sr = load_clip(clip_path, args.seconds)
    logger.warning("Clip: %s | %.1f s | %d Hz | shape=%s", clip_path.name, len(audio) / sr, sr, audio.shape)

    results: list[dict[str, Any]] = []
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    for cell in cells:
        logger.warning("=== Zelle %s (%s) gestartet: %s ===", cell.id, cell.quality_mode, time.strftime("%H:%M:%S"))
        log_path = out_root / f"cell_{cell.id}_{run_tag}.log"
        entry = run_cell(cell, audio, sr, log_path, out_root, not args.no_wav)
        entry["clip"] = clip_path.name
        entry["seconds"] = float(args.seconds)
        entry["run_tag"] = run_tag
        results.append(entry)
        if entry.get("error"):
            logger.warning("Zelle %s FEHLER: %s", cell.id, entry["error"])
        else:
            logger.warning(
                "Zelle %s fertig: wall=%.1f s | RT=%.1f× | quality=%.3f | MOS=%s | Phasen=%d(+%d skip)",
                cell.id,
                entry["wall_s"],
                entry["rt_factor"],
                entry["quality_estimate"],
                entry.get("pqs_mos"),
                entry.get("phases_executed", 0),
                entry.get("phases_skipped", 0),
            )
    out_file = out_root / f"results_{run_tag}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.warning("Ergebnisse: %s", out_file)
    # Kurzfassung auf stdout
    for r in results:
        if r.get("error"):
            print(f"{r['cell']:22s} FEHLER {r['error'][:80]}")
        else:
            print(
                f"{r['cell']:22s} wall={r['wall_s']:8.1f}s  RT={r['rt_factor']:6.1f}×  "
                f"qual={r['quality_estimate']:.3f}  MOS={r.get('pqs_mos')}  "
                f"phases={r['phases_executed']}(skip {r['phases_skipped']})"
            )


if __name__ == "__main__":
    main()
