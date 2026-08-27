#!/usr/bin/env python3
"""Flacher Medium-/Depth-Klassifikator — trainiert auf dem deklarierten Corpus.

Empfehlung 9 (Ohr-Messprogramm): Die tiefen Modelle bleiben eingefroren; der
einzige legitime „Trainings“-Hebel ist das schnelle Nach-Training flacher
Schätzer auf kuratierten Labels. Die Labels kommen aus den Corpus-Manifesten
(§15.2, deklarierte `chain`-Felder) und dem goldenen Hör-Set-Manifest
(audit/golden_listening_set.json: material + depth sind kuriert/verifiziert).

Deterministisch (§G5 (copilot-instructions.md)): fixe Seeds, keine Zufalls-Features. Kein tiefer
Trainings-Lauf — Laufzeit im Minutenbereich.

Erzeugt:
    models/medium_shallow_v1.joblib   — Material- UND Depth-Klassifikator + Metadaten
    models/medium_shallow_v1_report.json — CV-Report + Vergleich gegen MediumDetector

Usage:
    python scripts/train_medium_classifier.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "audit" / "golden_listening_set.json"
ARTIFACT = ROOT / "models" / "medium_shallow_v1.joblib"
REPORT = ROOT / "models" / "medium_shallow_v1_report.json"
_SEED = 42

DEPTH_CLASSES = ("1", "2", "3", "4+")


def load_audio_mono(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 48000:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 48000, sr).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def extract_features(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Deterministische DSP-Features (§G5 (copilot-instructions.md)): Zeitbereich + Spektrum + Bänder."""
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    frame_len, hop = 2048, 512
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n_frames = (len(audio) - frame_len) // hop + 1
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    fr = audio[idx].astype(np.float64)

    feats: list[float] = []
    rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
    feats += [np.percentile(rms, 10), np.percentile(rms, 50), np.percentile(rms, 90), rms.mean(), rms.std()]
    crest = np.max(np.abs(fr), axis=1) / (rms + 1e-12)
    feats += [crest.mean(), crest.max()]
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)), axis=1) / 2.0
    feats.append(float(zcr.mean()))

    win = np.hanning(frame_len)
    spec = np.abs(np.fft.rfft(fr * win, axis=1))
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    total = spec.sum(axis=1) + 1e-12
    centroid = (spec * freqs).sum(axis=1) / total
    feats += [float(centroid.mean()), float(centroid.std())]
    cum = np.cumsum(spec, axis=1)
    rolloff = np.array([freqs[np.argmax(cum[i] >= 0.85 * total[i])] for i in range(n_frames)])
    feats.append(float(rolloff.mean()))
    flat = np.exp(np.mean(np.log(spec + 1e-12), axis=1)) / (total / spec.shape[1])
    feats.append(float(flat.mean()))

    bands = [
        (20, 200),
        (200, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 12000),
        (12000, 20000),
    ]
    band_mask = np.array([(freqs >= lo) & (freqs < hi) for lo, hi in bands])  # (8, n_bins)
    band_e = spec @ band_mask.T + 1e-12  # (frames, 8)
    ratios = band_e / (band_e.sum(axis=1, keepdims=True) + 1e-12)
    feats += list(ratios.mean(axis=0))
    hf = band_e[:, -2:].sum(axis=1) / (band_e.sum(axis=1) + 1e-12)
    feats.append(float(hf.mean()))
    # Defekt-Signaturen (Ketten-/Material-Cues)
    feats.append(float(np.mean(np.abs(fr) ** 4) / (np.mean(np.abs(fr) ** 2) ** 2 + 1e-12)))  # Kurtosis-Prox
    env = np.abs(fr)  # Hüllkurve je Frame
    mod = np.abs(np.fft.rfft(env - env.mean(axis=0), axis=0))  # Modulation über die Zeit
    mod_freqs = np.fft.rfftfreq(n_frames, hop / sr)
    wow_band = (
        mod[(mod_freqs >= 0.1) & (mod_freqs <= 2.0)].mean() if np.any((mod_freqs >= 0.1) & (mod_freqs <= 2.0)) else 0.0
    )
    feats.append(float(wow_band))
    slope = np.polyfit(np.log(freqs[1:] + 1e-9), np.log(spec.mean(axis=0)[1:] + 1e-12), 1)[0]
    feats.append(float(slope))
    hf12 = spec[:, freqs >= 12000.0].sum(axis=1) / (total + 1e-12)
    feats.append(float(hf12.mean()))
    hf_flat = np.exp(np.mean(np.log(spec[:, freqs >= 8000.0] + 1e-12), axis=1)) / (
        spec[:, freqs >= 8000.0].sum(axis=1) / max(1, int((freqs >= 8000.0).sum())) + 1e-12
    )
    feats.append(float(hf_flat.mean()))
    # Crackle-Impulsdichte (Peaks je Sekunde, Refraktär-Abstand)
    _m = np.median(np.abs(audio)) + 1e-12
    _peaks = np.where(np.abs(audio) > 8.0 * _m)[0]
    _count = 0
    _last = -(10**9)
    for _p in _peaks:
        if _p - _last > int(0.005 * sr):
            _count += 1
            _last = _p
    feats.append(float(_count / max(1e-6, len(audio) / sr)))
    # Noise-Floor (10. Perzentil des mittleren Frame-Spektrums)
    feats.append(float(np.percentile(spec.mean(axis=0), 10)))
    # Band-Zeitvarianz (Hiss/Modulation vs. stabiler Ton)
    feats += list(np.std(band_e, axis=0) / (band_e.mean(axis=0) + 1e-12))
    return np.asarray(feats, dtype=np.float32)


def _cv_evaluate(model: Any, X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEED)
    pred = cross_val_predict(model, X, y, cv=cv)
    report: dict[str, Any] = {
        "accuracy": float(np.mean(pred == y)),
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
        "labels": labels,
        "report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
    }
    return report


def main() -> int:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = golden["items"]
    X_list: list[np.ndarray] = []
    y_mat: list[str] = []
    y_dep: list[str] = []
    for it in items:
        audio = load_audio_mono(it["path"])
        f = extract_features(audio)
        # Kuratierter Kontext als Feature (§v10.14.1: genau der Kanal, den die
        # Produktion dem MediumDetector als Prior durchreicht).
        era = float(it.get("era_year") or 0.0)
        f = np.concatenate([f, np.asarray([era, era // 10.0 * 10.0], dtype=np.float32)])
        X_list.append(f)
        y_mat.append(str(it["material"]))
        y_dep.append(str(it["depth"]))
    X = np.vstack(X_list)
    y_mat_np = np.asarray(y_mat)
    y_dep_np = np.asarray(y_dep)

    mat_labels = sorted(set(y_mat))
    dep_labels = sorted(set(y_dep), key=lambda d: (d == "4+", int(d) if d != "4+" else 4))

    mat_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=_SEED)
    dep_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=_SEED)

    mat_cv = _cv_evaluate(mat_model, X, y_mat_np, mat_labels)
    dep_cv = _cv_evaluate(dep_model, X, y_dep_np, dep_labels)

    # Fit auf allen Daten für das finale Artefakt
    mat_model.fit(X, y_mat_np)
    dep_model.fit(X, y_dep_np)

    # Baseline: MediumDetector auf demselben kuratierten Corpus
    baseline_mat_agree = sum(1 for i in items if i.get("detected_material") == i.get("material"))
    baseline_dep_agree = sum(1 for i in items if str(i.get("detected_depth")) == str(i.get("depth")))
    baseline = {
        "material_accuracy": round(baseline_mat_agree / len(items), 4),
        "depth_accuracy": round(baseline_dep_agree / len(items), 4),
        "n": len(items),
    }

    artifact = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": "dsp_v1",
        "seed": _SEED,
        "material": {"model": mat_model, "labels": mat_labels, "cv": mat_cv},
        "depth": {"model": dep_model, "labels": dep_labels, "cv": dep_cv},
        "baseline_medium_detector": baseline,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT)

    report = {
        "n_items": len(items),
        "material": {"labels": mat_labels, "cv": mat_cv},
        "depth": {"labels": dep_labels, "cv": dep_cv},
        "baseline_medium_detector": baseline,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Material-CV-Accuracy: {mat_cv['accuracy'] * 100:.1f}% (Baseline MediumDetector: {baseline['material_accuracy'] * 100:.1f}%)"
    )
    print(
        f"Depth-CV-Accuracy:    {dep_cv['accuracy'] * 100:.1f}% (Baseline MediumDetector: {baseline['depth_accuracy'] * 100:.1f}%)"
    )
    print(f"Artefakt: {ARTIFACT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
