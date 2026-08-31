#!/usr/bin/env python3
"""Hörordnungs-Kalibrierungs-Harness — psychoakustische Invarianten gegen
synthetische Referenz-Signale (Hörordnung §8; Vorstufe echter Panel-Tests).

Prüft die Kalibrierungs-RICHTUNGEN der Hörordnungs-Module mit Signalen, deren
psychoakustische Eigenschaften bekannt sind:

  1. Roughness (Zwicker): AM-Tiefe (70 Hz) ↑ ⇒ asper ↑
  2. Sharpness (Bismarck-Näherung): LPF-Cutoff ↓ ⇒ acum ↓; helles Rauschen > Sinus
  3. Residuum-Masking: lauter Kontext maskiert; stiller Kontext exponiert
  4. Residuum-Monotonie: Click-SNR ↑ ⇒ Salience ↑
  5. Einladungs-Gate: raues AM-Signal > glatter Sinus (max_asper)
  6. Hörstufen-Konsistenz: Natürlichkeit < Wärme < Klarheit < Brillanz

Ausgabe: Bericht; Exit-Code 1 bei verletzter Invariante (Kalibrierungs-Drift).

Verwendung:  python3 -B scripts/horordnung_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo-Root in den Pfad (Skript lebt in scripts/, backend-Importe brauchen Root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _noise(sr: int, dur_s: float, rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sr * dur_s))
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12) * rms).astype(np.float64)


def _lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, cutoff_hz / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x).astype(np.float64)


def _click_at(audio: np.ndarray, sr: int, t: float, amp: float) -> np.ndarray:
    idx = int(t * sr)
    out = audio.copy()
    n = max(1, int(0.002 * sr))
    out[idx : idx + n] += amp
    return out


def main() -> int:
    sr = 48000
    failures: list[str] = []

    # 1) Roughness: AM-Tiefe ↑ ⇒ asper ↑
    try:
        from backend.core.dsp.zwicker_metrics import compute_roughness_asper

        t = np.arange(sr * 6) / sr
        carrier = np.sin(2 * np.pi * 440 * t)
        am_shallow = (0.3 * (1 + 0.3 * np.sin(2 * np.pi * 70 * t)) * carrier).astype(np.float32)
        am_deep = (0.3 * (1 + 0.9 * np.sin(2 * np.pi * 70 * t)) * carrier).astype(np.float32)
        a1 = float(compute_roughness_asper(am_shallow, sr))
        a2 = float(compute_roughness_asper(am_deep, sr))
        if not (a2 > a1):
            failures.append(f"Roughness-Monotonie: AM-tief {a2:.3f} !> AM-flach {a1:.3f}")
        print(f"1) Roughness-Monotonie: {a1:.3f} → {a2:.3f} asper  {'OK' if a2 > a1 else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"Roughness nicht verfügbar: {exc}")

    # 2) Sharpness: LPF ↓ ⇒ acum ↓; Rauschen > Sinus
    try:
        from backend.core.inviting_sound_gate import compute_sharpness_acum

        noise = _noise(sr, 6.0, 0.2, seed=5)
        bright = compute_sharpness_acum(noise, sr)
        dull = compute_sharpness_acum(_lowpass(noise, sr, 1500.0), sr)
        sine = compute_sharpness_acum((0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 6) / sr)).astype(np.float32), sr)
        if not (dull < bright):
            failures.append(f"Sharpness-Monotonie: LPF {dull:.3f} !< breit {bright:.3f}")
        if not (bright > sine):
            failures.append(f"Sharpness-Diskrimination: Rauschen {bright:.3f} !> Sinus {sine:.3f}")
        print(
            f"2) Sharpness: breit={bright:.3f} dumpf={dull:.3f} sinus={sine:.3f}  {'OK' if dull < bright and bright > sine else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Sharpness nicht verfügbar: {exc}")

    # 3) Residuum-Masking: lauter Kontext maskiert, stiller exponiert
    try:
        from backend.core.residuum_masking import estimate_residuum_salience

        loud_ctx = _click_at(_noise(sr, 6.0, 0.2, seed=6), sr, 3.0, amp=0.05)
        silent_ctx = _click_at(np.zeros(sr * 6, dtype=np.float64), sr, 3.0, amp=0.05)
        s_loud = estimate_residuum_salience(loud_ctx, sr, 2.99, 3.01).salience
        s_silent = estimate_residuum_salience(silent_ctx, sr, 2.99, 3.01).salience
        if not (s_silent >= s_loud):
            failures.append(f"Residuum-Maskierung: still {s_silent:.3f} !>= laut {s_loud:.3f}")
        print(
            f"3) Residuum-Maskierung: lauter Kontext={s_loud:.3f}, still={s_silent:.3f}  {'OK' if s_silent >= s_loud else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Residuum nicht verfügbar: {exc}")

    # 4) Residuum-Monotonie: Click-Amplitude ↑ ⇒ Salience ↑
    try:
        from backend.core.residuum_masking import estimate_residuum_salience as _rs

        base = _noise(sr, 6.0, 0.05, seed=7)  # leiser Kontext: Diskrimination möglich
        q = _rs(_click_at(base, sr, 3.0, amp=0.1), sr, 2.99, 3.01).salience
        l = _rs(_click_at(base, sr, 3.0, amp=0.8), sr, 2.99, 3.01).salience
        if not (l >= q):
            failures.append(f"Residuum-Monotonie: laut {l:.3f} !>= leise {q:.3f}")
        print(f"4) Residuum-Monotonie: {q:.3f} → {l:.3f}  {'OK' if l >= q else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"Residuum-Monotonie nicht verfügbar: {exc}")

    # 5) Einladungs-Gate: AM-rau > Sinus glatt (max_asper)
    try:
        from backend.core.inviting_sound_gate import check_inviting_gate

        t6 = np.arange(sr * 12) / sr
        sine12 = (0.3 * np.sin(2 * np.pi * 440 * t6)).astype(np.float32)
        am12 = (0.3 * (1 + 0.9 * np.sin(2 * np.pi * 70 * t6)) * np.sin(2 * np.pi * 440 * t6)).astype(np.float32)
        r_sine = check_inviting_gate(sine12, sr, fatigue_index=0.0)
        r_am = check_inviting_gate(am12, sr, fatigue_index=0.0)
        if not (r_am.max_asper_in_voice > r_sine.max_asper_in_voice):
            failures.append(
                f"Einladungs-Gate-Richtung: AM {r_am.max_asper_in_voice:.3f} !> Sinus {r_sine.max_asper_in_voice:.3f}"
            )
        print(
            f"5) Einladungs-Gate: sinus={r_sine.max_asper_in_voice:.3f}, AM70={r_am.max_asper_in_voice:.3f} asper  "
            f"{'OK' if r_am.max_asper_in_voice > r_sine.max_asper_in_voice else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Einladungs-Gate nicht verfügbar: {exc}")

    # 6) Hörstufen-Konsistenz
    try:
        from backend.core.goal_priority_protocol import GoalPriorityProtocol

        gpp = GoalPriorityProtocol()
        order_ok = (
            gpp.hearing_tier("natuerlichkeit")
            < gpp.hearing_tier("waerme")
            < gpp.hearing_tier("transparenz")
            < gpp.hearing_tier("brillanz")
        )
        if not order_ok:
            failures.append("Hörstufen-Konsistenz verletzt")
        print(f"6) Hörstufen-Konsistenz: Natürlichkeit<Wärme<Klarheit<Brillanz  {'OK' if order_ok else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"GoalPriorityProtocol nicht verfügbar: {exc}")

    print()
    if failures:
        print(f"KALIBRIERUNG: {len(failures)} INVARIANTE(N) VERLETZT")
        for f in failures:
            print(f"  ❌ {f}")
        return 1
    print("KALIBRIERUNG: alle Invarianten erfüllt — Schwellwerte konsistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
