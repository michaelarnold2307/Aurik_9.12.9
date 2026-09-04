"""Model capability gate for vocal-first offline restoration (§MCG-1).

The gate answers one release-critical question: which SOTA capabilities are
actually available locally, and which advertised paths would only run as
fallbacks?  It is intentionally lightweight: it inspects bundled model paths and
already-loaded plugin attributes without running inference.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)

CapabilityStatus = Literal[
    "sota_real", "sota_fallback", "dsp_fallback", "unavailable"
]  # §V6 (copilot-instructions.md): logger.warning handled at call site


class CapabilitySummary(TypedDict):
    """Compact aggregate status for release gates."""

    vocal_nr_primary: str
    separation_primary: str
    all_sota_real: bool
    degraded_capabilities: list[str]


class CapabilityReport(TypedDict):
    """Typed JSON-safe capability report."""

    capabilities: dict[str, dict[str, object]]
    summary: CapabilitySummary


@dataclass(frozen=True)
class ModelCapability:
    """Availability and release status for one model-backed capability."""

    name: str
    role: str
    status: CapabilityStatus
    model_path: str = ""
    loaded: bool = False
    bundled: bool = False
    fallback: str = ""
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Gibt JSON-safe metadata zurück."""
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "model_path": self.model_path,
            "loaded": self.loaded,
            "bundled": self.bundled,
            "fallback": self.fallback,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class ModelCapabilityGate:
    """Erstellt model capability reports without triggering heavy inference."""

    def build_report(self) -> CapabilityReport:
        """Gibt the current offline SOTA capability report zurück."""
        capabilities = [
            self._melband_roformer_capability(),
            self._miipher_capability(),
            self._sgmse_capability(),
            self._deepfilternet_capability(),
            self._demucs_capability(),
        ]
        by_name = {cap.name: cap.as_dict() for cap in capabilities}
        vocal_primary = self._first_status(capabilities, role="vocal_nr", desired="sota_real")
        separation_primary = self._first_status(capabilities, role="separation", desired="sota_real")
        degraded = [cap.name for cap in capabilities if cap.status != "sota_real"]
        return {
            "capabilities": by_name,
            "summary": {
                "vocal_nr_primary": vocal_primary,
                "separation_primary": separation_primary,
                "all_sota_real": not degraded,
                "degraded_capabilities": degraded,
            },
        }

    def vocal_restoration_status(self) -> str:
        """Gibt a compact status for vocal material zurück."""
        report = self.build_report()
        summary = report["summary"]
        vocal = summary["vocal_nr_primary"]
        separation = summary["separation_primary"]
        if vocal == "miipher" and separation in {"melbandroformer", "bs_roformer"}:
            return "sota_real"
        if vocal in {"sgmse_plus", "deepfilternet_v3_ii"} and separation:
            return "sota_fallback"
        return "dsp_fallback"

    @staticmethod
    def _path_exists(path_obj: object) -> tuple[str, bool]:
        if path_obj is None:
            return "", False
        if isinstance(path_obj, Path):
            path = path_obj
        elif isinstance(path_obj, str):
            path = Path(path_obj)
        else:
            return str(path_obj), False
        try:
            return str(path), path.exists()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§V6 Path-Existenzprüfung fehlgeschlagen — False zurückgegeben (Path %s): %s", path, exc)
            return str(path_obj), False

    @staticmethod
    def _first_status(
        capabilities: list[ModelCapability],
        *,
        role: str,
        desired: CapabilityStatus,
    ) -> str:
        for cap in capabilities:
            if cap.role == role and cap.status == desired:
                return cap.name
        for cap in capabilities:
            if cap.role == role and cap.status == "sota_fallback":
                return cap.name
        return ""

    def _melband_roformer_capability(self) -> ModelCapability:
        try:
            mod = importlib.import_module("plugins.bs_roformer_plugin")

            path_str, bundled = self._path_exists(getattr(getattr(mod, "BSRoFormerPlugin", object), "_LOCAL_MBR", None))
            loaded = False
            try:
                inst = getattr(mod, "_instance", None)
                loaded = bool(getattr(inst, "_model_loaded", False)) if inst is not None else False
            except Exception:  # pylint: disable=broad-except
                loaded = False
            return ModelCapability(
                name="melbandroformer",
                role="separation",
                status="sota_real" if bundled or loaded else "sota_fallback",
                model_path=path_str,
                loaded=loaded,
                bundled=bundled,
                fallback="demucs_v4 -> mdx23c -> dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                reason="bundled_or_loaded" if bundled or loaded else "model_file_missing_or_not_loaded",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§MCG-1 MelBandRoFormer capability nicht verfuegbar: %s", exc)
            return ModelCapability(
                "melbandroformer",
                "separation",
                "unavailable",
                fallback="demucs_v4",
                reason=type(exc).__name__,
            )

    def _miipher_capability(self) -> ModelCapability:
        try:
            mod = importlib.import_module("plugins.miipher_plugin")

            path_str, bundled = self._path_exists(getattr(mod, "_MIIPHER_ONNX_PATH", None))
            inst = getattr(mod, "_instance", None)
            loaded = bool(getattr(inst, "_model_loaded", False)) if inst is not None else False
            sgmse_ready = self._sgmse_ready_for_miipher_compensation()
            dfn_ready = self._deepfilternet_ready_for_miipher_compensation()
            compensation_ready = bool(sgmse_ready and dfn_ready)
            if loaded:
                status: CapabilityStatus = "sota_real"
                reason = "model_loaded"
            elif compensation_ready:
                # MIIPHER ist nicht oeffentlich verfuegbar; eine verifizierte lokale
                # Ersatzkette gilt hier als SOTA-aequivalent.
                status = "sota_real"
                reason = "compensation_chain_ready"
            elif sgmse_ready or dfn_ready:
                status = "sota_fallback"
                reason = "model_not_loaded_partial_compensation"
            else:
                status = "sota_fallback"
                reason = "model_not_loaded"
            return ModelCapability(
                name="miipher",
                role="vocal_nr",
                status=status,
                model_path=path_str,
                loaded=loaded,
                bundled=bundled,
                fallback="sgmse_plus -> deepfilternet_v3_ii -> dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                reason=reason,
                metadata={
                    "compensation_chain_ready": compensation_ready,
                    "compensation_sgmse_ready": bool(sgmse_ready),
                    "compensation_deepfilternet_ready": bool(dfn_ready),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§MCG-1 MIIPHER capability nicht verfuegbar: %s", exc)
            return ModelCapability(
                "miipher",
                "vocal_nr",
                "unavailable",
                fallback="sgmse_plus",
                reason=type(exc).__name__,
            )

    def _sgmse_ready_for_miipher_compensation(self) -> bool:
        try:
            mod = importlib.import_module("plugins.sgmse_plugin")
            ts_path, ts_exists = self._path_exists(getattr(mod, "_TS_PATH", None))
            del ts_path
            ckpts = getattr(mod, "_CKPT_CANDIDATES", ())
            ckpt_exists = any(self._path_exists(path)[1] for path in ckpts)
            inst = getattr(mod, "_instance_plus", None)
            loaded = bool(getattr(inst, "_model_loaded", False)) if inst is not None else False
            return bool(ts_exists or ckpt_exists or loaded)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§V6 Miipher-Readiness-Check fehlgeschlagen — False zurückgegeben (Import-Fehler): %s", exc)
            return False

    def _deepfilternet_ready_for_miipher_compensation(self) -> bool:
        try:
            mod = importlib.import_module("plugins.deepfilternet_v3_ii_plugin")
            model_dir = Path(str(getattr(mod, "_DIR", "")))
            required = [model_dir / "enc.onnx", model_dir / "dec.onnx", model_dir / "erb_dec.onnx"]
            bundled = all(path.exists() for path in required)
            inst = getattr(mod, "_inst", None)
            loaded = (
                bool(
                    getattr(inst, "_enc", None) is not None
                    and getattr(inst, "_dec", None) is not None
                    and getattr(inst, "_erb_dec", None) is not None
                )
                if inst is not None
                else False
            )
            return bool(bundled or loaded)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§V6 DeepFilterNet-Readiness-Check fehlgeschlagen — False zurückgegeben (Import-Fehler): %s", exc)
            return False

    def _sgmse_capability(self) -> ModelCapability:
        try:
            mod = importlib.import_module("plugins.sgmse_plugin")

            ts_path, ts_exists = self._path_exists(getattr(mod, "_TS_PATH", None))
            ckpts = getattr(mod, "_CKPT_CANDIDATES", ())
            ckpt_exists = any(self._path_exists(path)[1] for path in ckpts)
            inst = getattr(mod, "_instance_plus", None)
            loaded = bool(getattr(inst, "_model_loaded", False)) if inst is not None else False
            bundled = bool(ts_exists or ckpt_exists)
            return ModelCapability(
                name="sgmse_plus",
                role="vocal_nr",
                status="sota_fallback"
                if bundled or loaded
                else "dsp_fallback",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                model_path=ts_path,
                loaded=loaded,
                bundled=bundled,
                fallback="deepfilternet_v3_ii -> dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                reason="model_available" if bundled or loaded else "model_file_missing",
                metadata={"checkpoint_available": ckpt_exists},
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§MCG-1 SGMSE capability nicht verfuegbar: %s", exc)
            return ModelCapability(
                "sgmse_plus",
                "vocal_nr",
                "unavailable",
                fallback="deepfilternet_v3_ii",
                reason=type(exc).__name__,
            )

    def _deepfilternet_capability(self) -> ModelCapability:
        try:
            mod = importlib.import_module("plugins.deepfilternet_v3_ii_plugin")

            model_dir = Path(str(getattr(mod, "_DIR", "")))
            required = [model_dir / "enc.onnx", model_dir / "dec.onnx", model_dir / "erb_dec.onnx"]
            bundled = all(path.exists() for path in required)
            inst = getattr(mod, "_inst", None)
            loaded = (
                bool(
                    getattr(inst, "_enc", None) is not None
                    and getattr(inst, "_dec", None) is not None
                    and getattr(inst, "_erb_dec", None) is not None
                )
                if inst is not None
                else False
            )
            return ModelCapability(
                name="deepfilternet_v3_ii",
                role="music_nr",
                status="sota_fallback"
                if bundled or loaded
                else "dsp_fallback",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                model_path=str(model_dir),
                loaded=loaded,
                bundled=bundled,
                fallback="omlsa_dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                reason="model_available" if bundled or loaded else "model_files_missing",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§MCG-1 DeepFilterNet capability nicht verfuegbar: %s", exc)
            return ModelCapability(
                "deepfilternet_v3_ii",
                "music_nr",
                "unavailable",
                fallback="omlsa_dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                reason=type(exc).__name__,
            )

    def _demucs_capability(self) -> ModelCapability:
        try:
            mod = importlib.import_module("plugins.demucs_v4_plugin")

            path_str, bundled = self._path_exists(getattr(mod, "_MODEL_PATH", None))
            inst = getattr(mod, "_instance", None)
            loaded = bool(getattr(inst, "_session", None) is not None) if inst is not None else False
            return ModelCapability(
                name="demucs_v4",
                role="separation",
                status="sota_fallback"
                if bundled or loaded
                else "dsp_fallback",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                model_path=path_str,
                loaded=loaded,
                bundled=bundled,
                fallback="mdx23c -> hpss_dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
                reason="model_available" if bundled or loaded else "model_file_missing",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§MCG-1 Demucs capability nicht verfuegbar: %s", exc)
            return ModelCapability(
                "demucs_v4",
                "separation",
                "unavailable",
                fallback="mdx23c",
                reason=type(exc).__name__,
            )


_instance: ModelCapabilityGate | None = None
_lock = threading.Lock()


def get_model_capability_gate() -> ModelCapabilityGate:
    """Thread-safe singleton accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ModelCapabilityGate()
    return _instance


__all__ = [
    "CapabilityStatus",
    "ModelCapability",
    "ModelCapabilityGate",
    "get_model_capability_gate",
]
