"""
Gaussian-Process Parameter Optimizer — Aurik 10.0.0
==================================================
Adaptiver Parameter-Optimierer auf Basis Gaussianischer Prozesse (GP).
Lernt materialspezifische optimale Restaurierungsparameter aus vergangenen
Sitzungen und schlägt via Upper-Confidence-Bound (UCB) Akquisitionsfunktion
neue Parameter-Kandidaten vor.

GP-Formel (posterior predictive):
    μ(x*) = K(x*,X) · [K(X,X) + σ²_n·I]⁻¹ · y
    σ²(x*) = K(x*,x*) - K(x*,X) · [K(X,X) + σ²_n·I]⁻¹ · K(X,x*)

Kernel: Squared Exponential (RBF)
    K(x,x') = σ²_f · exp(−½ · ||x−x'||² / l²)

Akquisition: UCB (Upper Confidence Bound)
    α(x) = μ(x) + κ · σ(x),  κ = 2.0

Gedächtnis: JSON-Dateien in ~/.aurik/gp_memory/<material>.json
Jeder Eintrag: {"params": [...], "score": float}

Referenzen:
    - Rasmussen & Williams, Gaussian Processes for Machine Learning (2006)
    - Snoek et al., Practical Bayesian Optimization of ML Algorithms (NeurIPS 2012)
    - Brochu et al., A Tutorial on Bayesian Optimization (2010)
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_MEMORY_DIR = Path.home() / ".aurik" / "gp_memory"
_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Warm-Start-Seed: vortrainierte Priors beim ersten Start kopieren
# ---------------------------------------------------------------------------
# data/gp_warmstart/<material>.json wird einmalig nach ~/.aurik/gp_memory/
# kopiert, wenn dort noch keine Datei existiert.  Bestehende echte Lerndata
# (n ≥ 1 Eintrag) werden NIEMALS überschrieben.
_WARMSTART_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gp_warmstart"


def _seed_warmstart_memory() -> None:
    """Kopiert vortrainierte GP-Memory-Dateien einmalig in ~/.aurik/gp_memory/.

    Wird beim Modul-Import ausgeführt. Jede Datei wird nur kopiert wenn
    ~/.aurik/gp_memory/<material>.json noch nicht existiert — echte Lerndaten
    werden nie überschrieben.
    """
    if not _WARMSTART_DIR.exists():
        return
    try:
        for src in _WARMSTART_DIR.glob("*.json"):
            dst = _MEMORY_DIR / src.name
            if not dst.exists():
                import shutil

                shutil.copy2(src, dst)
    except Exception as _ws_exc:
        logger.debug("GP Warm-Start-Seed fehlgeschlagen: %s", _ws_exc)


_seed_warmstart_memory()

_MAX_MEMORY_ENTRIES = 200  # Maximale Einträge pro Material
_UCB_KAPPA = 2.0  # Explorations-Gewicht
_NOISE_VAR = 1e-3  # Beobachtungsrauschen σ²_n
_KERNEL_LENGTH = 1.0  # GP-Kernel Längenscale l
_KERNEL_AMPLITUDE = 1.0  # GP-Kernel Amplitude σ_f
_N_RANDOM_CANDIDATES = 512  # Zufällige Kandidaten für UCB-Suche

# ---------------------------------------------------------------------------
# Pareto-Objectives (§2.5 Spec 03) — 15 Musical Goals
# ---------------------------------------------------------------------------

PARETO_OBJECTIVES: list[str] = [
    "brillanz",
    "waerme",
    "natuerlichkeit",
    "authentizitaet",
    "emotionalitaet",
    "transparenz",
    "bass_kraft",
    "groove",
    "spatial_depth",
    "tonal_center",
    "micro_dynamics",
    "timbre_authentizitaet",
    "separation_fidelity",
    "artikulation",
    "transient_energie",
]


# ---------------------------------------------------------------------------
# Parameter-Räume pro Domäne
# ---------------------------------------------------------------------------

# Format: {name: (low, high, mode)}
# mode: "float" | "int" | "log" (log-uniform sampling)

PARAMETER_SPACE: dict[str, tuple[float, float, str]] = {
    "noise_reduction_strength": (0.0, 1.0, "float"),  # full range: 0=bypass, 1=max NR
    "harmonic_boost_db": (0.0, 8.0, "float"),  # §2.5: extended for studio quality
    "ola_crossfade_ms": (5.0, 60.0, "float"),
    "compression_ratio": (1.05, 5.0, "log"),
    "eq_high_shelf_db": (-6.0, 6.0, "float"),
    "ar_order": (16.0, 200.0, "int"),  # extended for high-SR LPC analysis
    "click_threshold_sigma": (2.5, 8.0, "float"),  # lower floor for extreme crackle
    "hpf_cutoff_hz": (10.0, 120.0, "log"),
    "nr_smoothing_ms": (20.0, 200.0, "log"),
    "declip_threshold": (0.90, 0.99, "float"),
}

# Material-spezifische Default-Parameter (Initialstartpunkt)
MATERIAL_DEFAULTS: dict[str, dict[str, float]] = {
    "tape": {
        "noise_reduction_strength": 0.60,
        "harmonic_boost_db": 1.0,
        "ola_crossfade_ms": 20.0,
        "ar_order": 64,
        "nr_smoothing_ms": 80.0,
    },
    "vinyl": {
        "noise_reduction_strength": 0.45,
        "harmonic_boost_db": 0.5,
        "click_threshold_sigma": 4.5,
        "ar_order": 48,
        "nr_smoothing_ms": 60.0,
    },
    "shellac": {
        "noise_reduction_strength": 0.70,
        "harmonic_boost_db": 0.0,
        "click_threshold_sigma": 5.0,
        "ar_order": 32,
        "nr_smoothing_ms": 100.0,
    },
    "digital": {
        "noise_reduction_strength": 0.20,
        "compression_ratio": 1.5,
        "declip_threshold": 0.97,
        "eq_high_shelf_db": 0.5,
    },
    "unknown": {
        "noise_reduction_strength": 0.40,
        "harmonic_boost_db": 0.5,
        "ola_crossfade_ms": 15.0,
        "ar_order": 32,
    },
}

# ---------------------------------------------------------------------------
# §2.47 Material-Ähnlichkeitsmatrix (9×9) — Cross-Material GP-Wissenstransfer
# ---------------------------------------------------------------------------
# Spec 02 §2.47: Bei < 10 Beobachtungen für ein Material werden ähnliche
# Materialien als gewichtete Prior-Quellen hinzugezogen.
# Schlüssel entsprechen den cannonical Material-Namen der Carrier-Chain.

_MATERIAL_SIMILARITY_KEYS: list[str] = [
    "shellac",
    "wax_cyl",
    "vinyl_78",
    "vinyl_std",
    "tape_std",
    "tape_stu",
    "cassette",
    "digital",
    "mp3_lossy",
]

# Alias-Mapping: kurze Bezeichnungen → normalized keys
_MATERIAL_ALIAS: dict[str, str] = {
    "tape": "tape_std",
    "vinyl": "vinyl_std",
    "shellac": "shellac",
    "digital": "digital",
    "cassette": "cassette",
    "mp3": "mp3_lossy",
    "wax": "wax_cyl",
    "wax_cyl": "wax_cyl",
    "vinyl_78": "vinyl_78",
    "vinyl_std": "vinyl_std",
    "tape_std": "tape_std",
    "tape_stu": "tape_stu",
    "mp3_lossy": "mp3_lossy",
    "unknown": "digital",  # digital als neutraler Fallback
}

# Symmetrische Ähnlichkeitsmatrix (Spec 02 §2.47)
_MATERIAL_SIMILARITY_MATRIX: list[list[float]] = [
    # shl    wax    v78    vst    tst    tsu    cas    dig    mp3
    [1.00, 0.85, 0.75, 0.40, 0.15, 0.10, 0.10, 0.05, 0.05],  # shellac
    [0.85, 1.00, 0.70, 0.35, 0.10, 0.10, 0.08, 0.05, 0.05],  # wax_cyl
    [0.75, 0.70, 1.00, 0.65, 0.20, 0.15, 0.15, 0.08, 0.08],  # vinyl_78
    [0.40, 0.35, 0.65, 1.00, 0.45, 0.40, 0.35, 0.15, 0.12],  # vinyl_std
    [0.15, 0.10, 0.20, 0.45, 1.00, 0.85, 0.70, 0.25, 0.20],  # tape_std
    [0.10, 0.10, 0.15, 0.40, 0.85, 1.00, 0.60, 0.35, 0.25],  # tape_stu
    [0.10, 0.08, 0.15, 0.35, 0.70, 0.60, 1.00, 0.20, 0.18],  # cassette
    [0.05, 0.05, 0.08, 0.15, 0.25, 0.35, 0.20, 1.00, 0.55],  # digital
    [0.05, 0.05, 0.08, 0.12, 0.20, 0.25, 0.18, 0.55, 1.00],  # mp3_lossy
]

# Minimale Ähnlichkeit für Cross-Material-Transfer
_CROSS_MATERIAL_MIN_SIM: float = 0.30


def _apply_context_priors(
    params: dict[str, Any],
    space: dict[str, tuple[float, float, str]],
    chain_hint: dict[str, Any] | None = None,
    memory_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Blend chain and RestorationMemory priors into bounded DSP parameters."""
    adjusted = dict(params)

    if isinstance(memory_prior, dict):
        prior_params = memory_prior.get("phase_params", {})
        hpi_prev = float(memory_prior.get("hpi_achieved", 0.0) or 0.0)
        flat_prior: dict[str, float] = {}
        causal_conf = 0.0
        coalition_factor = 1.0
        if isinstance(prior_params, dict):
            with contextlib.suppress(Exception):
                causal_conf = float(prior_params.get("_causal_credit_confidence", 0.0) or 0.0)
            with contextlib.suppress(Exception):
                coalition_factor = float(prior_params.get("_coalition_learning_factor", 1.0) or 1.0)
            for key, value in prior_params.items():
                if key in space:
                    with contextlib.suppress(Exception):
                        flat_prior[key] = float(value)
                elif isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if nested_key in space:
                            with contextlib.suppress(Exception):
                                flat_prior[nested_key] = float(nested_value)
        prior_weight = float(np.clip(0.15 + 0.25 * hpi_prev, 0.15, 0.55)) if flat_prior else 0.0
        if prior_weight > 0.0 and causal_conf > 0.0:
            prior_weight = float(np.clip(prior_weight * (1.0 + 0.50 * np.clip(causal_conf, 0.0, 1.0)), 0.15, 0.65))
        if prior_weight > 0.0 and coalition_factor > 1.0:
            prior_weight = float(np.clip(prior_weight * np.clip(coalition_factor, 1.0, 1.25), 0.15, 0.65))
        for key, prior_value in flat_prior.items():
            lo, hi, mode = space[key]
            base_value = float(adjusted.get(key, prior_value))
            blended = (1.0 - prior_weight) * base_value + prior_weight * prior_value
            if mode == "int":
                blended = float(round(blended))
            adjusted[key] = float(np.clip(blended, lo, hi))

    if isinstance(chain_hint, dict):
        strength_scale = float(np.clip(chain_hint.get("strength_scale", 1.0), 0.25, 1.0))
        if strength_scale < 0.999:
            scalable_tokens = ("strength", "boost", "ratio")
            for key, value in list(adjusted.items()):
                if key not in space or not any(token in key for token in scalable_tokens):
                    continue
                lo, hi, mode = space[key]
                try:
                    scaled = float(value) * strength_scale
                except (TypeError, ValueError):
                    continue
                if mode == "int":
                    scaled = float(round(scaled))
                adjusted[key] = float(np.clip(scaled, lo, hi))

    return adjusted


def _material_similarity(m1: str, m2: str) -> float:
    """§2.47 Ähnlichkeit zwischen zwei Material-Bezeichnungen [0, 1].

    Verwendet Alias-Mapping + MATERIAL_SIMILARITY_MATRIX.
    Liefert 0.0 wenn kein Eintrag in der Matrix.
    """
    n1 = _MATERIAL_ALIAS.get(m1)
    n2 = _MATERIAL_ALIAS.get(m2)
    if n1 is None or n2 is None:
        return 0.0
    try:
        i = _MATERIAL_SIMILARITY_KEYS.index(n1)
        j = _MATERIAL_SIMILARITY_KEYS.index(n2)
        return _MATERIAL_SIMILARITY_MATRIX[i][j]
    except ValueError as exc:
        logger.debug("§V6 Material-Similaritäts-Key nicht gefunden — 0.0 zurückgegeben (%s, %s): %s", n1, n2, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class ParameterProposal:
    """Vorschlag des GP-Optimierers."""

    parameters: dict[str, Any]  # {param_name: wert}
    expected_quality: float  # GP-Posterior μ(x*)
    uncertainty: float  # GP-Posterior σ(x*)
    ucb_value: float  # μ + κ·σ
    from_memory: bool = False  # Aus Material-Gedächtnis
    iteration: int = 0  # Optimierungsiteration
    param_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Vorschlag als JSON-kompatibles Dict."""
        return {
            "parameters": self.parameters,
            "expected_quality": round(self.expected_quality, 4),
            "uncertainty": round(self.uncertainty, 4),
            "ucb_value": round(self.ucb_value, 4),
            "from_memory": self.from_memory,
            "iteration": self.iteration,
        }


@dataclass
class MemoryEntry:
    """Ein Eintrag im GP-Gedächtnis."""

    params_normalized: list[float]  # normierter Parametervektor [0,1]^d
    score: float  # Qualitätsscore [0,1] oder MOS o.ä.
    material: str
    timestamp: float = field(default_factory=time.time)
    goal_scores: dict[str, float] = field(default_factory=dict)  # 15 Musical Goals (§2.5)


# ---------------------------------------------------------------------------
# RBF-Kernel
# ---------------------------------------------------------------------------


def _rbf_kernel(
    X: np.ndarray, Y: np.ndarray, length: float = _KERNEL_LENGTH, amplitude: float = _KERNEL_AMPLITUDE
) -> np.ndarray:
    """
    Squared-Exponential Kernel: K_{ij} = σ²_f · exp(−½ ||x_i − y_j||² / l²)

    Args:
        X: (n, d)
        Y: (m, d)
    Returns:
        (n, m) Kernel-Matrix
    """
    # ||x_i - y_j||^2 via broadcasting
    X2 = np.sum(X**2, axis=1, keepdims=True)  # (n,1)
    Y2 = np.sum(Y**2, axis=1, keepdims=True)  # (m,1)
    XY = X @ Y.T  # (n,m)
    dist_sq = np.maximum(X2 + Y2.T - 2.0 * XY, 0.0)
    kernel: np.ndarray[Any, Any] = np.asarray(
        (amplitude**2) * np.exp(-0.5 * dist_sq / (length**2 + 1e-12)),
        dtype=np.float64,
    )
    return kernel


# ---------------------------------------------------------------------------
# Gaussian-Process (Posterior)
# ---------------------------------------------------------------------------


class GaussianProcess:
    """
    Leichtgewichtiger GP mit RBF-Kernel, geschlossene Lösung.
    Unterstützt inkrementelles Update und UCB-Akquisition.
    """

    def __init__(
        self,
        noise_var: float = _NOISE_VAR,
        length: float = _KERNEL_LENGTH,
        amplitude: float = _KERNEL_AMPLITUDE,
    ):
        self.noise_var = noise_var
        self.length = length
        self.amplitude = amplitude
        self._X: np.ndarray | None = None  # (n, d)
        self._y: np.ndarray | None = None  # (n,)
        self._L: np.ndarray | None = None  # Cholesky-Faktor
        self._alpha: np.ndarray | None = None  # L^{-T} L^{-1} y

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """GP an Beobachtungen (X, y) anpassen."""
        # NaN/Inf-Bereinigung
        valid = np.isfinite(y)
        if not np.any(valid):
            return
        X = X[valid]
        y = y[valid]
        self._X = X.copy()
        self._y = y.copy()
        K = _rbf_kernel(X, X, self.length, self.amplitude)
        K += self.noise_var * np.eye(len(X))
        try:
            L = scipy.linalg.cholesky(K, lower=True)
        except scipy.linalg.LinAlgError:
            # Fallback: erhöhe Rauschen
            K += 1e-4 * np.eye(len(X))
            L = scipy.linalg.cholesky(K, lower=True)
        self._L = L
        self._alpha = scipy.linalg.cho_solve((L, True), y)

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Posteriore Vorhersage für X_star.

        Returns:
            (mu, sigma) — jeweils (m,) Arrays
        """
        if self._X is None:
            return np.zeros(len(X_star)), np.ones(len(X_star))

        K_star = _rbf_kernel(X_star, self._X, self.length, self.amplitude)  # (m,n)
        mu = K_star @ self._alpha  # (m,)

        # Varianz: k(x*,x*) - k(x*,X)·(K+σ²I)^{-1}·k(X,x*)
        K_ss = _rbf_kernel(X_star, X_star, self.length, self.amplitude)  # (m,m)
        V = scipy.linalg.solve_triangular(self._L, K_star.T, lower=True)
        var = np.diag(K_ss) - np.sum(V**2, axis=0)
        sigma = np.sqrt(np.maximum(var, 1e-12))

        return mu, sigma

    def ucb(self, X_star: np.ndarray, kappa: float = _UCB_KAPPA) -> np.ndarray:
        """UCB-Akquisitionswerte für Kandidaten X_star."""
        mu, sigma = self.predict(X_star)
        ucb_vals: np.ndarray[Any, Any] = np.asarray(mu + kappa * sigma, dtype=np.float64)
        return ucb_vals

    @property
    def n_observations(self) -> int:
        """Anzahl der aktuell gefitteten Beobachtungen."""
        return len(self._y) if self._y is not None else 0


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Parameter-Normierung
# ---------------------------------------------------------------------------


def _param_names_sorted(space: dict[str, tuple[float, float, str]] | None = None) -> list[str]:
    if space is None:
        space = PARAMETER_SPACE
    return sorted(space.keys())


def _normalize_params(
    params: dict[str, float],
    space: dict[str, tuple[float, float, str]] | None = None,
) -> np.ndarray:
    """Parameter-Dict → normierter Vektor [0,1]^d."""
    if space is None:
        space = PARAMETER_SPACE
    names = _param_names_sorted(space)
    v = []
    for name in names:
        if name in params:
            lo, hi, mode = space[name]
            raw = float(params[name])
            if mode == "log":
                lo_l, hi_l = math.log(lo + 1e-9), math.log(hi)
                val_l = math.log(max(raw, lo + 1e-9))
                v.append((val_l - lo_l) / (hi_l - lo_l + 1e-12))
            else:
                v.append((raw - lo) / (hi - lo + 1e-12))
        else:
            v.append(0.5)  # Mittelpunkt als Default
    return np.clip(np.array(v, dtype=np.float64), 0.0, 1.0)  # type: ignore[no-any-return]


def _denormalize_params(
    vec: np.ndarray,
    space: dict[str, tuple[float, float, str]] | None = None,
) -> dict[str, Any]:
    """Normierter Vektor → Parameter-Dict."""
    if space is None:
        space = PARAMETER_SPACE
    names = _param_names_sorted(space)
    result: dict[str, Any] = {}
    for i, name in enumerate(names):
        lo, hi, mode = space[name]
        x = float(np.clip(vec[i], 0.0, 1.0))
        if mode == "log":
            lo_l, hi_l = math.log(lo + 1e-9), math.log(hi)
            raw = math.exp(lo_l + x * (hi_l - lo_l))
        else:
            raw = lo + x * (hi - lo)
        if mode == "int":
            result[name] = int(round(raw))
        else:
            result[name] = round(raw, 4)
    return result


def _sample_random_candidates(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n Zufalls-Kandidaten im d-dimensionalen Einheitswürfel."""
    # Latin-Hypercube-Näherung: besser verteilt als rein zufällig
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(0, 1, size=n)) / n
    return samples  # type: ignore[no-any-return]


def _safe_normalize_targets(y: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Bereinigt and normalize objective vectors for stable GP training.

    Returns:
        (y_clean, y_min, y_range, y_norm)
    """
    y_clean = np.asarray(y, dtype=np.float64)
    y_clean = np.nan_to_num(y_clean, nan=0.0, posinf=1e6, neginf=-1e6)
    y_clean = np.clip(y_clean, -1e6, 1e6)

    if y_clean.size == 0:
        return y_clean, 0.0, 1.0, y_clean

    y_min = float(np.min(y_clean))
    y_max = float(np.max(y_clean))
    y_range = float(y_max - y_min)
    if (not math.isfinite(y_range)) or y_range < 1e-6:
        y_range = 1e-6

    y_norm = np.divide(
        y_clean - y_min,
        y_range,
        out=np.zeros_like(y_clean),
        where=np.isfinite(y_clean),
    )
    y_norm = np.nan_to_num(y_norm, nan=0.0, posinf=1.0, neginf=0.0)
    return y_clean, y_min, y_range, y_norm


# ---------------------------------------------------------------------------
# Gedächtnis-I/O
# ---------------------------------------------------------------------------


def _memory_path(material: str) -> Path:
    return _MEMORY_DIR / f"{material}.json"


def _load_memory(material: str) -> list[MemoryEntry]:
    path = _memory_path(material)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # Support both list format (current) and legacy dict formats
        if isinstance(raw, dict):
            raw = raw.get("observations", [])
        if not isinstance(raw, list):
            return []
        entries = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            params = item.get("params")
            # Legacy: params as dict → skip (parameter ordering unknown)
            if not isinstance(params, list):
                continue
            try:
                entries.append(
                    MemoryEntry(
                        params_normalized=params,
                        score=float(item["score"]),
                        material=material,
                        timestamp=float(item.get("ts", 0.0)),
                        goal_scores=dict(item.get("goal_scores", {})),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return entries
    except Exception as exc:
        logger.warning("GP-Gedächtnis konnte nicht geladen werden: %s", exc)
        return []


def _save_memory(material: str, entries: list[MemoryEntry]) -> None:
    path = _memory_path(material)
    try:
        raw = [
            {
                "params": e.params_normalized,
                "score": e.score,
                "ts": e.timestamp,
                "goal_scores": e.goal_scores,
            }
            for e in entries[-_MAX_MEMORY_ENTRIES:]
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(raw, f)
    except Exception as exc:
        logger.warning("GP-Gedächtnis konnte nicht gespeichert werden: %s", exc)


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class GPParameterOptimizer:
    """
    Pro-Aufnahme adaptiver Parameter-Optimierer via Gaussianischen Prozessen.

    Workflow::

        opt = GPParameterOptimizer()

        # Vorschlag holen (mit material-spezifischem Gedächtnis)
        proposal = opt.propose(material="tape")

        # Parameter anwenden und Qualität messen
        score = apply_and_score(proposal.parameters)

        # GP mit neuem Datenpunkt aktualisieren
        opt.update(proposal.parameters, score, material="tape")

        # Nächster Vorschlag (informed by previous round)
        proposal2 = opt.propose(material="tape")
    """

    def __init__(
        self,
        parameter_space: dict | None = None,
        noise_var: float = _NOISE_VAR,
        kappa: float = _UCB_KAPPA,
        rng_seed: int | None = None,
    ):
        self._space = parameter_space or PARAMETER_SPACE
        self._gp = GaussianProcess(noise_var=noise_var)
        self._kappa = kappa
        self._rng = np.random.default_rng(rng_seed)
        self._session_X: list[np.ndarray] = []
        self._session_y: list[float] = []
        self._dim = len(_param_names_sorted(self._space))
        self._iterations: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def propose(
        self,
        material: str = "unknown",
        n_init: int = 5,
        embedding_vec: np.ndarray | None = None,
        era_warmstart: dict[str, float] | None = None,
        chain_hint: dict[str, Any] | None = None,
        memory_prior: dict[str, Any] | None = None,
    ) -> ParameterProposal:
        """
        Schlägt die nächsten zu testenden Parameter vor.

        Args:
            material:      Trägermaterial, bestimmt Prior und Gedächtnis
            n_init:        Mindestanzahl Beobachtungen bevor GP genutzt wird
            embedding:     Optionaler 256-dim Perceptual-Embedding-Vektor
                           (wird derzeit als Kontextmerkmal gelogt, aber
                           nicht in den GP-Eingaberaum eingebaut — zukünftige
                           Erweiterung via Input Warping)
            era_warmstart: Optionaler Ära-Prior aus EraClassifier.get_gp_warmstart()
                           (§2.14). Beim Cold-Start (n_obs < n_init) überschreibt
                           dieser Prior die gematerialten Default-Parameter für
                           epochenspezifische Initialisierung (z.B.
                           noise_reduction_strength ≈ 0.90 für Aufnahmen vor 1940).
                           Nur gültige Bounds-Werte werden übernommen.
        Returns:
            ParameterProposal
        """
        it = self._iterations.get(material, 0)

        # Embedding-Vektor als Kontextmerkmal loggen (zukünftige GP-Input-Warping-Erweiterung)
        if embedding_vec is not None:
            logger.debug(
                "propose: embedding_vec shape=%s (Eingabe-Warping reserviert, nicht im GP-Raum)",
                getattr(embedding_vec, "shape", type(embedding_vec)),
            )

        # Gedächtnis laden
        memory = _load_memory(material)
        all_X: list[np.ndarray] = []
        all_y: list[float] = []

        for entry in memory:
            if len(entry.params_normalized) == self._dim:
                all_X.append(np.array(entry.params_normalized))
                all_y.append(entry.score)

        # Note: _session_X/_session_y are intentionally NOT merged here.
        # update() persists every entry immediately via _save_memory(), so
        # _load_memory() above already contains all session observations.
        # Merging _session_X/_session_y would double-count every entry that
        # was added in the current session.

        n_obs = len(all_X)

        # §2.47 Cross-Material-Transfer: bei < 10 Beobachtungen ähnliche Materialien einbeziehen
        if n_obs < 10:
            all_X, all_y = self._augment_with_cross_material(material, all_X, all_y)

        if n_obs < n_init:
            # Zufällige Exploration oder Defaults
            params, mu_scalar, sig_scalar = self._random_or_default(material, n_obs)
            # Era-Warmstart: Ära-Prior überschreibt Default-Parameter beim Cold-Start (§2.14)
            # Gültige Bounds-geclampte Werte aus EraClassifier.get_gp_warmstart() werden
            # direkt in den Parametervorschlag eingebracht, bevor GP-Daten vorliegen.
            if era_warmstart:
                for _era_k, _era_v in era_warmstart.items():
                    try:
                        _era_v_f = float(_era_v)
                        if _era_k in self._space and math.isfinite(_era_v_f):
                            _lo, _hi, _mode = self._space[_era_k]
                            params[_era_k] = float(np.clip(_era_v_f, _lo, _hi))
                    except (TypeError, ValueError) as _exc:
                        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            params = _apply_context_priors(params, self._space, chain_hint=chain_hint, memory_prior=memory_prior)
            self._iterations[material] = it + 1
            return ParameterProposal(
                parameters=params,
                expected_quality=mu_scalar,
                uncertainty=sig_scalar,
                ucb_value=mu_scalar + self._kappa * sig_scalar,
                from_memory=(n_obs > 0),
                iteration=it,
                param_names=_param_names_sorted(self._space),
            )

        # GP-Vorhersage + UCB-Optimierung
        X_obs = np.stack(all_X)  # (n, d)
        y_obs = np.array(all_y)  # (n,)

        # Normierung der Zielwerte auf [0, 1]
        y_obs, y_min, y_range, y_norm = _safe_normalize_targets(y_obs)

        self._gp.fit(X_obs, y_norm)

        # Kandidaten-Suche: Random + Perturbation der besten Beobachtung
        candidates = _sample_random_candidates(_N_RANDOM_CANDIDATES, self._dim, self._rng)
        best_idx = int(np.argmax(y_obs))
        best_x = X_obs[best_idx]
        # Perturbation um besten Punkt (Exploitation-Bias)
        perturbed = best_x + self._rng.normal(0, 0.05, size=(64, self._dim))
        perturbed = np.clip(perturbed, 0.0, 1.0)
        candidates = np.vstack([candidates, perturbed])

        ucb_vals = self._gp.ucb(candidates, kappa=self._kappa)
        best_cand = int(np.argmax(ucb_vals))
        x_star = candidates[best_cand]

        mu_arr, sigma_arr = self._gp.predict(x_star[None, :])
        mu_real = float(y_min + mu_arr[0] * y_range)
        sigma_real = float(sigma_arr[0] * y_range)

        params = _denormalize_params(x_star, self._space)
        # Material-Defaults überlappen mit GP nur für nicht optimierte Params
        defaults = MATERIAL_DEFAULTS.get(material, {})
        for param, val in defaults.items():
            if param not in params:
                params[param] = val
        params = _apply_context_priors(params, self._space, chain_hint=chain_hint, memory_prior=memory_prior)

        self._iterations[material] = it + 1
        return ParameterProposal(
            parameters=params,
            expected_quality=mu_real,
            uncertainty=sigma_real,
            ucb_value=float(ucb_vals[best_cand]),
            from_memory=True,
            iteration=it,
            param_names=_param_names_sorted(self._space),
        )

    def propose_pareto(
        self,
        material: str = "unknown",
        n_candidates: int = 5,
        n_init: int = 5,
        era_warmstart: dict[str, float] | None = None,
        chain_hint: dict[str, Any] | None = None,
        memory_prior: dict[str, Any] | None = None,
    ) -> list[ParameterProposal]:
        """True multi-objective Pareto-front proposals over 15 Musical Goals (§2.5).

        Uses one GP per Musical Goal objective. Samples *_N_RANDOM_CANDIDATES*
        candidates in the DSP parameter space, predicts all 15 objective
        posteriors, computes the non-dominated (Pareto) front via dominance
        check, and selects up to *n_candidates* diverse representatives via
        crowding-distance selection.

        Falls back to UCB-diversity sampling when goal_scores data are
        insufficient (< n_init entries with populated goal_scores).

        Args:
            material:      Trägermaterial (bestimmt Prior und Gedächtnis)
            n_candidates:  Maximale Anzahl Pareto-Kandidaten (≤ 5)
            n_init:        Mindestanzahl Beobachtungen mit goal_scores, bevor
                           MOO genutzt wird (UCB-Fallback unter diesem Wert)
            era_warmstart: Optionaler Ära-Prior aus EraClassifier (§2.14)

        Returns:
            Liste von ParameterProposal-Objekten (Pareto-Front, max. n_candidates).
            Enthält mindestens einen Eintrag (Fallback auf propose()).
        """
        n_c = max(1, min(n_candidates, 5))
        memory = _load_memory(material)

        # ── Einträge mit vollständigen goal_scores ermitteln ──────────────
        moo_entries = [
            e for e in memory if len(e.params_normalized) == self._dim and len(e.goal_scores) >= len(PARETO_OBJECTIVES)
        ]

        if len(moo_entries) < n_init:
            # ── Fallback: UCB-Diversitätssampling (bisheriges Verhalten) ──
            logger.debug(
                "propose_pareto: zu wenig MOO-Daten (%d < %d), UCB-Ersatzpfad für '%s'",
                len(moo_entries),
                n_init,
                material,
            )
            return self._pareto_ucb_fallback(
                material=material,
                n_c=n_c,
                n_init=n_init,
                memory=memory,
                era_warmstart=era_warmstart,
                chain_hint=chain_hint,
                memory_prior=memory_prior,
            )

        # ── One separate GP per objective ─────────────────────────────────
        X_obs = np.array([e.params_normalized for e in moo_entries])  # (n, d)

        # Objective-Matrix: (n, len(PARETO_OBJECTIVES))
        obj_matrix = np.zeros((len(moo_entries), len(PARETO_OBJECTIVES)), dtype=np.float64)
        for row_i, entry in enumerate(moo_entries):
            for col_j, obj in enumerate(PARETO_OBJECTIVES):
                val = entry.goal_scores.get(obj, np.nan)
                obj_matrix[row_i, col_j] = val if math.isfinite(val) else 0.0

        # Kandidaten im normierten Parameterraum samplen
        candidates = _sample_random_candidates(_N_RANDOM_CANDIDATES, self._dim, self._rng)

        # Posterior-Means für alle Objectives vorhersagen: (n_cands, n_objectives)
        pred_means = np.zeros((len(candidates), len(PARETO_OBJECTIVES)), dtype=np.float64)
        pred_sigma = np.zeros((len(candidates), len(PARETO_OBJECTIVES)), dtype=np.float64)
        for col_j in range(len(PARETO_OBJECTIVES)):
            y_obj = obj_matrix[:, col_j]
            _, y_min, y_range, y_norm = _safe_normalize_targets(y_obj)
            gp_obj = GaussianProcess(noise_var=_NOISE_VAR)
            gp_obj.fit(X_obs, y_norm)
            mu_n, sig_n = gp_obj.predict(candidates)
            pred_means[:, col_j] = np.nan_to_num(y_min + mu_n * y_range, nan=0.0)
            pred_sigma[:, col_j] = np.nan_to_num(sig_n * y_range, nan=0.0)

        # ── Pareto-Dominanz-Check ─────────────────────────────────────────
        # Kandidat i dominiert j wenn: ∀k: f_i[k] ≥ f_j[k] UND ∃k: f_i[k] > f_j[k]
        n_cands = len(candidates)
        is_dominated = np.zeros(n_cands, dtype=bool)
        for i in range(n_cands):
            if is_dominated[i]:
                continue
            for j in range(n_cands):
                if i == j or is_dominated[j]:
                    continue
                # Prüfe ob j durch i dominiert wird
                if np.all(pred_means[i] >= pred_means[j]) and np.any(pred_means[i] > pred_means[j]):
                    is_dominated[j] = True

        pareto_indices = np.where(~is_dominated)[0]

        # ── Crowding-Distance-Selektion für diverse Repräsentanten ───────
        selected_indices = self._crowding_distance_select(pareto_indices, pred_means, n_c)

        proposals: list[ParameterProposal] = []
        it = self._iterations.get(material, 0)
        for sel_idx in selected_indices:
            params = _denormalize_params(candidates[sel_idx], self._space)
            params = _apply_context_priors(params, self._space, chain_hint=chain_hint, memory_prior=memory_prior)
            mean_quality = float(np.mean(pred_means[sel_idx]))
            mean_uncertainty = float(np.mean(pred_sigma[sel_idx]))
            proposals.append(
                ParameterProposal(
                    parameters=params,
                    expected_quality=mean_quality,
                    uncertainty=mean_uncertainty,
                    ucb_value=mean_quality + _UCB_KAPPA * mean_uncertainty,
                    from_memory=True,
                    iteration=it,
                    param_names=_param_names_sorted(self._space),
                )
            )

        if not proposals:
            proposals.append(
                self.propose(
                    material=material,
                    n_init=n_init,
                    era_warmstart=era_warmstart,
                    chain_hint=chain_hint,
                    memory_prior=memory_prior,
                )
            )

        self._iterations[material] = it + 1
        logger.debug(
            "propose_pareto: %d Pareto-Kandidaten (aus %d nicht-dominierten) für material='%s'",
            len(proposals),
            len(pareto_indices),
            material,
        )
        return proposals

    def _pareto_ucb_fallback(
        self,
        material: str,
        n_c: int,
        n_init: int,
        memory: list[MemoryEntry],
        era_warmstart: dict[str, float] | None,
        chain_hint: dict[str, Any] | None = None,
        memory_prior: dict[str, Any] | None = None,
    ) -> list[ParameterProposal]:
        """UCB kappa-diversity fallback when insufficient MOO goal_scores data."""
        proposals: list[ParameterProposal] = []
        all_X = [np.array(e.params_normalized) for e in memory if len(e.params_normalized) == self._dim]
        all_y = [e.score for e in memory if len(e.params_normalized) == self._dim]
        kappa_values = [0.5, 1.0, 2.0, 3.0, 4.5][:n_c]

        if len(all_X) < 2:
            base = self.propose(
                material=material,
                n_init=n_init,
                era_warmstart=era_warmstart,
                chain_hint=chain_hint,
                memory_prior=memory_prior,
            )
            for k in range(n_c):
                varied_params = dict(base.parameters)
                rng_shift = self._rng.uniform(-0.05, 0.05, size=self._dim)
                for j_idx, pname in enumerate(_param_names_sorted(self._space)):
                    lo, hi, mode = self._space[pname]
                    raw = varied_params.get(pname, (lo + hi) / 2)
                    shifted = float(np.clip(raw + rng_shift[j_idx] * (hi - lo), lo, hi))
                    if mode == "int":
                        shifted = float(round(shifted))
                    varied_params[pname] = shifted
                varied_params = _apply_context_priors(
                    varied_params, self._space, chain_hint=chain_hint, memory_prior=memory_prior
                )
                proposals.append(
                    ParameterProposal(
                        parameters=varied_params,
                        expected_quality=base.expected_quality,
                        uncertainty=base.uncertainty + k * 0.02,
                        ucb_value=base.ucb_value,
                        from_memory=False,
                        iteration=self._iterations.get(material, 0),
                        param_names=base.param_names,
                    )
                )
            return proposals

        X_arr = np.array(all_X)
        y_arr = np.array(all_y)
        y_arr, y_min, y_range, y_norm = _safe_normalize_targets(y_arr)
        gp_fb = GaussianProcess(noise_var=_NOISE_VAR)
        gp_fb.fit(X_arr, y_norm)
        cands = _sample_random_candidates(256, self._dim, self._rng)
        mu_n, sig_n = gp_fb.predict(cands)
        mu_real = np.nan_to_num(y_min + mu_n * y_range, nan=0.0, posinf=1e6, neginf=-1e6)
        sig_real = np.nan_to_num(sig_n * y_range, nan=0.0, posinf=1e6, neginf=0.0)

        it = self._iterations.get(material, 0)
        for kappa in kappa_values:
            ucb_vals = mu_real + kappa * sig_real
            best_idx = int(np.argmax(ucb_vals))
            params = _denormalize_params(cands[best_idx], self._space)
            params = _apply_context_priors(params, self._space, chain_hint=chain_hint, memory_prior=memory_prior)
            proposals.append(
                ParameterProposal(
                    parameters=params,
                    expected_quality=float(mu_real[best_idx]),
                    uncertainty=float(sig_real[best_idx]),
                    ucb_value=float(ucb_vals[best_idx]),
                    from_memory=True,
                    iteration=it,
                    param_names=_param_names_sorted(self._space),
                )
            )
        if not proposals:
            proposals.append(
                self.propose(
                    material=material,
                    n_init=n_init,
                    era_warmstart=era_warmstart,
                    chain_hint=chain_hint,
                    memory_prior=memory_prior,
                )
            )
        return proposals

    @staticmethod
    def _crowding_distance_select(
        pareto_indices: np.ndarray,
        pred_means: np.ndarray,
        n_select: int,
    ) -> list[int]:
        """Wählt aus: n_select diverse representatives from Pareto front via crowding distance.

        Args:
            pareto_indices: 1-D array of non-dominated candidate indices
            pred_means:     (n_cands, n_objectives) posterior mean matrix
            n_select:       number of representatives to return

        Returns:
            List of selected candidate indices (subset of pareto_indices)
        """
        if len(pareto_indices) <= n_select:
            return [int(i) for i in pareto_indices.tolist()]

        n_obj = pred_means.shape[1]
        front = pred_means[pareto_indices]  # (|front|, n_obj)
        distances = np.zeros(len(pareto_indices))

        for m in range(n_obj):
            sorted_order = np.argsort(front[:, m])
            f_min = front[sorted_order[0], m]
            f_max = front[sorted_order[-1], m]
            f_range = max(f_max - f_min, 1e-12)
            # Boundary points get infinite distance
            distances[sorted_order[0]] = np.inf
            distances[sorted_order[-1]] = np.inf
            for k in range(1, len(sorted_order) - 1):
                distances[sorted_order[k]] += (front[sorted_order[k + 1], m] - front[sorted_order[k - 1], m]) / f_range

        # Select n_select with highest crowding distance
        top_k = np.argsort(distances)[::-1][:n_select]
        return [int(pareto_indices[i]) for i in top_k]

    def update(
        self,
        parameters: dict[str, Any],
        score: float,
        material: str = "unknown",
        goal_scores: dict[str, float] | None = None,
    ) -> None:
        """
        Aktualisiert das GP-Gedächtnis mit einem neuen Datenpunkt.

        Args:
            parameters:  Parameter-Dict (wie von propose() zurückgegeben)
            score:       Gesamt-Qualitätsscore (z.B. PQS-MOS normiert [0,1])
            material:    Trägermaterial
            goal_scores: Optionales Dict der 15 Musical-Goals-Scores
                         (Keys = PARETO_OBJECTIVES). Wird für echten MOO
                         in propose_pareto() benötigt. NaN/Inf-Werte werden
                         gefiltert; fehlende Keys bleiben leer.
        """
        x_norm = _normalize_params(parameters, self._space)

        # Validierung Gesamt-Score
        score_f = float(score)
        if not math.isfinite(score_f):
            logger.warning("GP.Aktualisierung: Nicht-finiter Wert wird übersprungen")
            return

        # Validierung goal_scores
        clean_goals: dict[str, float] = {}
        if goal_scores:
            for obj in PARETO_OBJECTIVES:
                val = goal_scores.get(obj)
                if val is not None:
                    val_f = float(val)
                    if math.isfinite(val_f):
                        clean_goals[obj] = val_f

        self._session_X.append(x_norm)
        self._session_y.append(score_f)

        # Persistenz
        entry = MemoryEntry(
            params_normalized=x_norm.tolist(),
            score=score_f,
            material=material,
            goal_scores=clean_goals,
        )
        memory = _load_memory(material)
        memory.append(entry)
        _save_memory(material, memory)

        logger.debug(
            "GP-Aktualisierung: material=%s Wert=%.4f goals=%d n_memory=%d",
            material,
            score,
            len(clean_goals),
            len(memory),
        )

    def best_known_parameters(self, material: str) -> dict[str, Any] | None:
        """Gibt die Parameter mit dem bisher höchsten Score zurück."""
        memory = _load_memory(material)
        if not memory:
            return MATERIAL_DEFAULTS.get(material)
        best = max(memory, key=lambda e: e.score)
        return _denormalize_params(np.array(best.params_normalized), self._space)

    def forget(self, material: str) -> None:
        """Löscht das Gedächtnis für ein Material."""
        path = _memory_path(material)
        if path.exists():
            path.unlink()
        self._session_X.clear()
        self._session_y.clear()

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _random_or_default(self, material: str, n_obs: int) -> tuple[dict[str, Any], float, float]:
        """Zufälliger oder Default-Parameter-Satz für Exploration."""
        if n_obs == 0:
            # Erster Aufruf: Material-Defaults
            defaults = MATERIAL_DEFAULTS.get(material, {})
            x_norm = _normalize_params(defaults, self._space)
        else:
            # Zufall mit leichtem Default-Bias
            x_rand = _sample_random_candidates(1, self._dim, self._rng)[0]
            defaults = MATERIAL_DEFAULTS.get(material, {})
            x_def = _normalize_params(defaults, self._space)
            x_norm = 0.6 * x_def + 0.4 * x_rand

        params = _denormalize_params(x_norm, self._space)
        return params, 0.5, 0.5  # uninformierter Prior

    def _augment_with_cross_material(
        self,
        material: str,
        all_X: list[np.ndarray],
        all_y: list[float],
        max_cross_entries: int = 20,
    ) -> tuple[list[np.ndarray], list[float]]:
        """§2.47 Cross-Material GP-Prior-Transfer.

        Wenn das eigene Material < 10 Beobachtungen hat, werden ähnliche
        Materialien (Ähnlichkeit ≥ _CROSS_MATERIAL_MIN_SIM) als zusätzliche
        Prior-Punkte einbezogen — gewichtet nach Ähnlichkeit.

        Gewichtung: Score wird mit Ähnlichkeit multipliziert → ähnlichere
        Materialien dominieren den Prior stärker.

        Args:
            material:          Primäres Material.
            all_X:             Bereits geladene eigene Beobachtungen (normiert).
            all_y:             Score-Werte der eigenen Beobachtungen.
            max_cross_entries: Maximale Fremd-Einträge die hinzugefügt werden.

        Returns:
            Erweiterte (all_X, all_y) Listen (oder unverändert wenn n_obs >= 10).
        """
        _CROSS_MIN_OBS = 10  # Spec 02 §2.47: Transfer nur bei < 10 Obs
        if len(all_X) >= _CROSS_MIN_OBS:
            return all_X, all_y

        augmented_X = list(all_X)
        augmented_y = list(all_y)
        added = 0

        # Sortiere andere Materialien nach Ähnlichkeit (absteigend)
        similar_materials = sorted(
            [
                (m, _material_similarity(material, m))
                for m in _MATERIAL_ALIAS
                if m != material and _material_similarity(material, m) >= _CROSS_MATERIAL_MIN_SIM
            ],
            key=lambda t: t[1],
            reverse=True,
        )
        # Duplikate entfernen (Alias-Normalisierung kann gleiche Materialien erzeugen)
        seen_normalized: set[str] = {_MATERIAL_ALIAS.get(material, material)}
        unique_similar = []
        for m_name, sim in similar_materials:
            nm = _MATERIAL_ALIAS.get(m_name, m_name)
            if nm not in seen_normalized:
                seen_normalized.add(nm)
                unique_similar.append((nm, sim))

        for cross_material, sim in unique_similar:
            if added >= max_cross_entries:
                break
            cross_memory = _load_memory(cross_material)
            cross_entries = [e for e in cross_memory if len(e.params_normalized) == self._dim]
            if not cross_entries:
                continue

            # Älteste Einträge bevorzugen (konservativere Prior-Schätzung)
            cross_entries.sort(key=lambda e: e.timestamp)
            for entry in cross_entries[: max_cross_entries - added]:
                # Ähnlichkeit als Dämpfungsfaktor für Score
                weighted_score = entry.score * sim
                augmented_X.append(np.array(entry.params_normalized))
                augmented_y.append(weighted_score)
                added += 1

        if added > 0:
            logger.info(
                "§2.47 cross-material augmentation: material='%s' own=%d cross=%d total=%d",
                material,
                len(all_X),
                added,
                len(augmented_X),
            )

        return augmented_X, augmented_y


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------

_optimizer: GPParameterOptimizer | None = None
_optimizer_lock = threading.Lock()


def get_optimizer() -> GPParameterOptimizer:
    """Globaler Singleton-Optimierer."""
    global _optimizer  # pylint: disable=global-statement
    if _optimizer is None:
        with _optimizer_lock:
            if _optimizer is None:
                _optimizer = GPParameterOptimizer()
    return _optimizer


def propose_parameters(
    material: str = "unknown",
    n_init: int = 5,
) -> ParameterProposal:
    """Convenience: Nächster Parameter-Vorschlag."""
    return get_optimizer().propose(material=material, n_init=n_init)


def record_quality(
    parameters: dict[str, Any],
    score: float,
    material: str = "unknown",
) -> None:
    """Convenience: Qualitätsscore für Parameter-Satz registrieren."""
    get_optimizer().update(parameters, score, material)
