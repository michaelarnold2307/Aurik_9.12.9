"""UI-Schwellwerte — zentrale Konstanten für das Frontend.

§V25-Äquivalent für die GUI: JEDE dimensionale/zeitliche Konstante
MUSS aus diesem Modul bezogen werden. Keine Magic Numbers in
modern_window.py oder anderen UI-Dateien.

Analog zu CalibratedConstants für das Backend.
"""

from __future__ import annotations

from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# Layout & Dimensionen (alle in Pixeln)
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 800
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600

SIDEBAR_WIDTH = 280
SIDEBAR_COLLAPSED_WIDTH = 48

PROGRESS_BAR_HEIGHT = 6
STATUS_BAR_HEIGHT = 24

PHASE_LIST_ROW_HEIGHT = 28
PHASE_LIST_ICON_SIZE = 16

# ═══════════════════════════════════════════════════════════════════════════════
# Timing (alle in Millisekunden)
# ═══════════════════════════════════════════════════════════════════════════════

POLL_INTERVAL_MS = 50  # Pipeline-Status-Polling
HEARTBEAT_INTERVAL_MS = 1000  # Watchdog-Check-Intervall
ANIMATION_DURATION_MS = 300  # Übergangs-Animationen
TOOLTIP_DELAY_MS = 500
DEBOUNCE_MS = 200  # Input-Debounce für Suchfelder

# ═══════════════════════════════════════════════════════════════════════════════
# Spacing & Padding
# ═══════════════════════════════════════════════════════════════════════════════

PADDING_TIGHT = 4
PADDING_NORMAL = 8
PADDING_WIDE = 12
PADDING_SECTION = 16

SPACING_TIGHT = 2
SPACING_NORMAL = 6
SPACING_WIDE = 10

MARGIN_CONTENT = 8

# ═══════════════════════════════════════════════════════════════════════════════
# Farben — §v10.990 Zentrale UI-Palette (einzige Quelle für Hex-Werte)
# ═══════════════════════════════════════════════════════════════════════════════
# Die bridge-seitigen quality_color-Hex-Werte (backend/api/bridge.py) MÜSSEN mit
# QUALITY_* übereinstimmen — abgesichert durch tests/unit/test_frontend_backend_harmony.py

SURFACE_BG = "#2a2a35"  # Badge-/Pill-Hintergrund
TEXT_PRIMARY = "#d0d0d0"  # Primärtext auf dunklem Grund
TEXT_MUTED = "#888"  # Sekundärtext / Inaktiv
HEADER_ACCENT = "#B8CCEE"  # Dialog-Header (Plugin-Manager)

QUALITY_STUDIO = "#2196F3"  # depth 1–2: Studio-Qualität (Blau)
QUALITY_MODERATE = "#4CAF50"  # depth 3: moderate Qualität (Grün)
QUALITY_DEEP_CHAIN = "#E6A817"  # depth 4+: erwartete Einschränkungen (Bernstein)

STATUS_OK_TEXT = "#6ab86a"
STATUS_OK_BG = "#1a2a1a"
STATUS_WARN_TEXT = "#b8a840"
STATUS_WARN_BG = "#2a2a1a"
STATUS_ORANGE_TEXT = "#c87830"
STATUS_ORANGE_BG = "#2a2010"
STATUS_CRIT_TEXT = "#c84848"
STATUS_CRIT_BG = "#2a1010"

BADGE_MATERIAL_TEXT = "#b8a068"  # Material-Badge (Gold)
BADGE_ERA_TEXT = "#6890b8"  # Ära-Badge (Blau)
BADGE_GENRE_TEXT = "#68a068"  # Genre-Badge (Grün)

# ═══════════════════════════════════════════════════════════════════════════════
# Opacity
# ═══════════════════════════════════════════════════════════════════════════════

OPACITY_DISABLED = 0.4
OPACITY_HOVER = 0.85
OPACITY_ACTIVE = 1.0
OPACITY_OVERLAY = 0.92

# ═══════════════════════════════════════════════════════════════════════════════
# Depth-abhängige UI-Werte (aus CalibrationContext)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DepthAwareUI:
    """UI-Werte die von der Transfer-Chain-Tiefe abhängen."""

    chain_depth: int = 1
    material_confidence: float = 1.0  # §v10.303.9

    @property
    def confidence_multiplier(self) -> float:
        """§v10.303.9: Bei niedriger Material-Confidence weniger Phasen."""
        if self.material_confidence < 0.25:
            return 0.40  # 60% weniger Phasen
        elif self.material_confidence < 0.35:
            return 0.55  # 45% weniger
        elif self.material_confidence < 0.50:
            return 0.75
        return 1.0

    @property
    def quality_color(self) -> str:
        """Eingeschränkte Qualitätsfarbe für tiefe Ketten (§v10.990: Palette-Tokens)."""
        if self.chain_depth >= 4:
            return QUALITY_DEEP_CHAIN  # Bernstein — erwartete Einschränkungen
        elif self.chain_depth >= 3:
            return QUALITY_MODERATE  # Grün — moderate Qualität
        return QUALITY_STUDIO  # Blau — Studio-Qualität

    @property
    def expected_duration_factor(self) -> float:
        """Erwartete längere Dauer für tiefe Ketten."""
        if self.chain_depth >= 4:
            return 2.5  # 2.5× länger als depth=1
        elif self.chain_depth >= 3:
            return 1.8
        return 1.0

    @property
    def phase_count_estimate(self) -> int:
        """Geschätzte Phasenanzahl — Confidence-bewusst."""
        if self.chain_depth >= 4:
            _base = 43
        elif self.chain_depth >= 3:
            _base = 35
        else:
            _base = 25  # Studio-Master
        return max(12, int(_base * self.confidence_multiplier))

    @property
    def progress_warning_threshold_pct(self) -> int:
        """Warnschwelle für Fortschritt in Prozent."""
        if self.chain_depth >= 4:
            return 75  # Längere Pipeline — später warnen
        return 85


def DEPTH_AWARE_UI_FACTORY(chain_label: str) -> DepthAwareUI:
    """Erzeugt DepthAwareUI aus einem Chain-Label wie 'reel_tape → vinyl → cassette → mp3_low'."""
    stages = chain_label.count(" → ") + 1 if chain_label else 1
    return DepthAwareUI(chain_depth=stages)


def de_num(value: float, digits: int = 2) -> str:
    """§GUI-T5: Formatiert eine Zahl mit deutschem Dezimalkomma (42,50 statt 42.50).

    Gemeinsame Quelle der Wahrheit für alle nutzersichtbaren Dezimalzahlen
    außerhalb von modern_window.py (z. B. Vorher/Nachher-Panel, Phasenbericht).
    modern_window._de_num ist funktional identisch (AST-getestet).
    """
    return f"{float(value):.{max(0, int(digits))}f}".replace(".", ",")
