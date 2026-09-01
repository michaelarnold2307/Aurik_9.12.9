"""StemRemixBalancer — LUFS-korrekter, phasenkohärenter Stem-Re-Mix (§1.4/§2.8).

Spec 02 (02_pipeline_architecture.md) und phase_42_vocal_enhancement referenzieren
diese Klasse seit v10.0.0 — sie existierte aber nie im Code, phase_42 fiel auf einen
pauschalen Direkt-Mix („(enhanced_vocals + instr_stem) * 0.5“) zurück. Implementiert
Rev. 2026-08-17 mit den Lessons Learned der Tiefenanalyse:

    1. LUFS-Ziel = QUELL-LUFS (nie ein fixes Ziel): Der Re-Mix wird auf die
       Integrated-Loudness der Original-Referenz gezogen — kein Pumpen durch
       Zielwerte-Drift (Befund: LUFS-Δ −3.7 LU in alten Läufen).
    2. Summen-Invariante: mix = vocals·w + instrumental·(2−w) → bei w=1.0 exakt
       vocals+instrumental (kein impliziter 0.5-Faktor).
    3. Soft-Knee-Peak-Cap statt Hard-Clamp (§III copilot-instructions.md).
    4. Fail-Safe: NaN/RMS-Kollaps/Stille-Referenz → Original-Referenz wird
       unverändert zurückgegeben (Primum-non-nocere, kein Export-Block-Risiko).
    5. LUFS-Messung via export_quality_gate._measure_lufs — EINE kanonische
       Implementierung statt einer vierten Kopie (Drift-Lektion).

Wissenschaftliche Verankerung: BS.1770-konforme Gated-Loudness (vereinfacht),
ITU-R BS.1770-4 Ziel-Lautheit = Referenz, Audio-EQ-Cookbook-Prinzip fürs Knie.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_MAX_GAIN_DB = 6.0  # Kein Rausch-Boost: max. +6 dB Loudness-Korrektur.
_SOFT_KNEE_START = 0.95  # Soft-Knee beginnt bei 95 % Full-Scale.
_KNEE_WIDTH = 0.035


class StemRemixBalancer:
    """Balanciert Stem-Re-Mixe auf Quell-LUFS mit Soft-Knee-Peak-Schutz."""

    def balance_remix(
        self,
        vocals: np.ndarray,
        instrumental: np.ndarray,
        original_reference: np.ndarray,
        sr: int,
        vocal_weight: float | None = 1.0,
    ) -> np.ndarray:
        """LUFS-korrekter Re-Mix zweier Stems gegen die Original-Referenz.

        Args:
            vocals: Vokal-Stem (enhanced), mono/stereo.
            instrumental: Begleitungs-Stem, mono/stereo.
            original_reference: Original-Mix VOR der Stem-Verarbeitung
                (Lautheits-Referenz UND Fail-Safe-Quelle).
            sr: Sample-Rate.
            vocal_weight: 1.0 = natürliche Summe; <1.0 reduziert den
                Vokal-Anteil (Energie-Erhalt: instrumental bekommt 2−w).

        Returns:
            Re-Mix mit Referenz-LUFS und Soft-Knee-Peak-Schutz.
            Bei NaN/Kollaps: original_reference unverändert.
            Bei stiller Referenz: Mix mit Peak-Schutz (kein LUFS-Zug).
        """
        try:
            ref = np.asarray(original_reference, dtype=np.float32)
            voc = np.asarray(vocals, dtype=np.float32)
            ins = np.asarray(instrumental, dtype=np.float32)

            # Shape-Invariante: alle Signale auf gemeinsame Länge trimmen.
            _n = min(ref.shape[0], voc.shape[0], ins.shape[0])
            if _n < 256:
                return cast(np.ndarray, ref.copy())
            ref = ref[:_n]
            voc = voc[:_n]
            ins = ins[:_n]
            if ref.ndim != voc.ndim or ref.ndim != ins.ndim:
                # Layout-Konflikt (mono vs. stereo): Referenz gewinnt.
                voc = self._coerce_layout(voc, ref)
                ins = self._coerce_layout(ins, ref)

            # Summen-Invariante: w=1.0 → exakt voc + ins.
            _w = float(np.clip(vocal_weight if vocal_weight is not None else 1.0, 0.0, 2.0))
            mix = voc * _w + ins * (2.0 - _w)

            if not np.isfinite(mix).all():
                logger.warning("StemRemixBalancer: NaN/Inf im Mix — Ursprungs-Referenz zurück")
                return cast(np.ndarray, ref.copy())

            # 1) Loudness-Ausgleich auf QUELL-LUFS (BS.1770-vereinfacht, eine
            #    kanonische Messung aus export_quality_gate).
            try:
                from backend.core.export_quality_gate import (  # pylint: disable=import-outside-toplevel
                    ExportQualityGate,
                )

                _lufs_ref = float(ExportQualityGate._measure_lufs(ref, sr))
                _lufs_mix = float(ExportQualityGate._measure_lufs(mix, sr))
            except Exception as _lufs_exc:
                logger.debug("StemRemixBalancer: LUFS-Messung nicht verfügbar (%s) — ohne Ausgleich", _lufs_exc)
                _lufs_ref = None
                _lufs_mix = None

            _gain_db = 0.0
            if _lufs_ref is not None and _lufs_mix is not None and _lufs_ref > -69.0 and _lufs_mix > -69.0:
                _gain_db = float(np.clip(_lufs_ref - _lufs_mix, -_MAX_GAIN_DB, _MAX_GAIN_DB))
                if abs(_gain_db) > 0.01:
                    mix = mix * float(10.0 ** (_gain_db / 20.0))

            # 2) Fail-Safe: RMS-Kollaps → Referenz (kein stilles Ergebnis).
            _rms_mix = float(np.sqrt(np.mean(np.square(mix.astype(np.float64))) + 1e-12))
            if _rms_mix < 1e-4:
                logger.warning("StemRemixBalancer: RMS-Kollaps (%.2e) — Ursprungs-Referenz zurück", _rms_mix)
                return cast(np.ndarray, ref.copy())

            # 3) Soft-Knee-Peak-Cap statt Hard-Clamp (§III): sanftes Knie über 95 % FS.
            _peak = float(np.max(np.abs(mix)))
            if _peak > _SOFT_KNEE_START:
                _knee = mix / _peak  # FS-Normalisierung, Peak = 1.0
                _over = np.abs(_knee) > _SOFT_KNEE_START
                if np.any(_over):
                    _soft = np.where(
                        _over,
                        np.sign(_knee)
                        * (_SOFT_KNEE_START + _KNEE_WIDTH * np.tanh((np.abs(_knee) - _SOFT_KNEE_START) / _KNEE_WIDTH)),
                        _knee,
                    )
                    # KEINE Rückskalierung mit _peak: das Knie wirkt im FS-Bereich;
                    # Rückskalierung würde die Schwelle auf 0.95·peak verschieben und
                    # der finale Clip ein hartes Plateau erzeugen (Befund 2026-08-17).
                    mix = _soft.astype(np.float32)

            mix = np.clip(mix, -1.0, 1.0).astype(np.float32)
            if _gain_db != 0.0 or _peak > _SOFT_KNEE_START:
                logger.debug(
                    "StemRemixBalancer: gain=%+.1f dB (ref %.1f → mix %.1f LUFS), peak=%.3f",
                    _gain_db,
                    float(_lufs_ref) if _lufs_ref is not None else float("nan"),
                    float(_lufs_mix) if _lufs_mix is not None else float("nan"),
                    float(_peak),
                )
            return cast(np.ndarray, mix)
        except Exception as _exc:
            # §V6 (copilot-instructions.md): kein Silent-Failure, aber auch kein Phasen-Crash.
            logger.warning("StemRemixBalancer fehlgeschlagen (%s) — Ursprungs-Referenz zurück", _exc)
            try:
                return cast(np.ndarray, (np.asarray(original_reference, dtype=np.float32).copy()))
            except Exception:
                return cast(np.ndarray, np.asarray(original_reference).copy())

    @staticmethod
    def _coerce_layout(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Gleicht mono/stereo-Layout an die Referenz an (kanalweise)."""
        if signal.ndim == reference.ndim:
            return signal
        if reference.ndim == 2 and signal.ndim == 1:
            return cast(np.ndarray, (np.stack([signal, signal], axis=1)))
        if reference.ndim == 1 and signal.ndim == 2:
            return cast(np.ndarray, signal.mean(axis=1))
        return signal


def get_stem_remix_balancer() -> StemRemixBalancer:
    """Singleton (leichtgewichtig — keine Modelle)."""
    return _SINGLETON


_SINGLETON = StemRemixBalancer()
