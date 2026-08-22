"""
SongPrognoseWidget — Pre-flight deep analysis tab for Aurik 10.

Displays material, era/genre, restorability score, detected defects,
phase prognosis and an overall 'Chancen-Score' before restoration starts.
Updated asynchronously as background analysis threads complete.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QColor, QFont, QPainter, QPen  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (  # pylint: disable=no-name-in-module
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from Aurik10.i18n import t

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (Aurik dark theme)
# ---------------------------------------------------------------------------
_C_BG = "rgba(14, 18, 36, 0.92)"
_C_CARD = "rgba(22, 28, 50, 0.82)"
_C_BORDER = "rgba(102, 126, 234, 0.22)"
_C_GREEN = "#82B89A"
_C_AMBER = "#C8A84B"
_C_RED = "#B87A7A"
_C_BLUE = "#667EEA"
_C_MUTED = "#6E839D"
_C_TEXT = "#D0DCFF"
_C_TEXT_DIM = "#8898BB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card_style(border_color: str = _C_BORDER) -> str:
    return f"background:{_C_CARD}; border:1px solid {border_color}; border-radius:10px; padding:10px 14px;"


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{_C_TEXT_DIM}; font-size:8pt; font-weight:600; letter-spacing:0.8px; background:transparent; padding:0;"
    )
    return lbl


class _ScoreDial(QWidget):
    """Small circular score indicator (0–100)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._score: float = 0.0
        self._color = _C_MUTED
        self.setFixedSize(80, 80)

    def set_score(self, score: float, color: str) -> None:
        """Setzt Score und Farbe des Kreisindikators und triggert Repaint."""
        self._score = float(np.clip(score, 0.0, 100.0))
        self._color = color
        self.update()

    def paintEvent(self, _event: Any) -> None:  # type: ignore[override]
        """Zeichnet den ringförmigen Score-Indikator inklusive Zahlenwert."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) // 2 - 6
        cx, cy = w // 2, h // 2

        # Track
        pen = QPen(QColor("#1E253D"), 7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 225 * 16, -270 * 16)

        # Arc
        if self._score > 0:
            pen2 = QPen(QColor(self._color), 7)
            pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen2)
            span = int(-270 * 16 * self._score / 100.0)
            p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 225 * 16, span)

        # Value label
        p.setPen(QColor(_C_TEXT))
        font = QFont(self.font().family(), 14, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(self._score)}")
        p.end()


class _DefectPill(QLabel):
    """Compact pill showing a single defect with severity colour."""

    _SEVERITY_COLORS = {
        "low": ("#4A90D9", "rgba(74,144,217,0.14)", "rgba(74,144,217,0.30)"),
        "medium": ("#C8A84B", "rgba(200,168,75,0.14)", "rgba(200,168,75,0.30)"),
        "high": ("#B87A7A", "rgba(184,122,122,0.14)", "rgba(184,122,122,0.30)"),
        "resolved": ("#82B89A", "rgba(130,184,154,0.12)", "rgba(130,184,154,0.28)"),
    }

    # Blu-ray disc read-side iridescent metallic gradient (blue-violet reflex)
    _BLURAY_ACTIVE_STYLE = (
        "color:#B8D4FF;"
        " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        "   stop:0 rgba(93,79,205,0.22), stop:0.45 rgba(0,170,230,0.18),"
        "   stop:1.0 rgba(140,100,220,0.22));"
        " border:1px solid qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        "   stop:0 rgba(123,104,238,0.65), stop:0.5 rgba(0,191,255,0.55),"
        "   stop:1.0 rgba(159,122,234,0.65));"
        " border-radius:8px; padding:2px 8px; font-size:8pt; font-weight:700;"
    )

    def __init__(self, label: str, severity: str = "medium") -> None:
        super().__init__(label)
        self._base_label = label
        self._severity = severity
        fg, bg, border = self._SEVERITY_COLORS.get(severity, self._SEVERITY_COLORS["medium"])
        self._default_style = (
            f"color:{fg}; background:{bg}; border:1px solid {border};"
            " border-radius:8px; padding:2px 8px; font-size:8pt; font-weight:600;"
        )
        self.setStyleSheet(self._default_style)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_active(self) -> None:
        """Mark pill as actively being repaired: Blu-ray iridescent metallic highlight."""
        base_name = self._base_label.split("  ")[0].strip()
        self.setText(f"\U0001f527\u202f{base_name}")
        self.setStyleSheet(self._BLURAY_ACTIVE_STYLE)

    def clear_active(self) -> None:
        """Entfernt active highlight, restore severity-based style."""
        self.setText(self._base_label)
        self.setStyleSheet(self._default_style)

    def set_resolved(self) -> None:
        """Mark pill as fixed: green colour, count/metric stripped, checkmark added."""
        fg, bg, border = self._SEVERITY_COLORS["resolved"]
        # Strip trailing count/metric (separated by double space) to show only the defect name.
        base_name = self._base_label.split("  ")[0].strip()
        self.setText(f"\u2713\u202f{base_name}")
        self.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {border};"
            " border-radius:8px; padding:2px 8px; font-size:8pt; font-weight:600;"
        )

    def update_metric(self, new_label: str) -> None:
        """Aktualisiert displayed metric/count text (during correcting phase)."""
        self._base_label = new_label
        self.setText(new_label)


# ---------------------------------------------------------------------------
# DEFECT_LABELS: German user-readable names for limiting defects
# ---------------------------------------------------------------------------
_DEFECT_LABELS: dict[str, str] = {
    "extremes_rauschen": "Extremes Rauschen",
    "starkes_rauschen": "Starkes Rauschen",
    "leichtes_rauschen": "Leichtes Rauschen",
    "starkes_clipping": "Starke Übersteuerung",
    "leichtes_clipping": "Übersteuerung",
    "sehr_schmale_bandbreite": "Sehr schmale Bandbreite",
    "schmale_bandbreite": "Schmale Bandbreite",
    "starkes_crackle": "Starkes Knistern",
    "leichtes_crackle": "Leichtes Knistern",
    # DefectScanner keys
    "CRACKLE": "Knistern",
    "CLICKS": "Knackser",
    "CLIPPING": "Übersteuerung",
    "HUM": "Netzbrummen",
    "NOISE": "Rauschen",
    "DROPOUT": "Aussetzer",
    "WOW_FLUTTER": "Gleichlaufschwankungen",
    "REVERB": "Nachhall",
    "TAPE_HISS": "Bandrauschen",
    "DC_OFFSET": "Gleichspannung",
    # Extended DefectType enum labels
    "WOW": "Gleichlaufschwankung (Wow)",
    "FLUTTER": "Tonhöhenzittern (Flutter)",
    "ALIASING": "Aliasing-Verzerrung",
    "AZIMUTH_ERROR": "Azimuth-Abweichung",
    "HEAD_MISALIGNMENT": "Bandkopf-Ausrichtungsschaden",
    "BIAS_ERROR": "Vormagnetisierungsfehler",
    "BANDWIDTH_LOSS": "Bandbreitenverlust",
    "PHASE_ISSUES": "Phasenproblem",
    "STEREO_IMBALANCE": "Stereo-Ungleichgewicht",
    "PITCH_DRIFT": "Tonhöhendrift",
    "REVERB_EXCESS": "Übermäßiger Nachhall",
    "PRINT_THROUGH": "Übersprechen (Print-Through)",
    "QUANTIZATION_NOISE": "Quantisierungsrauschen",
    "JITTER_ARTIFACTS": "Jitter-Artefakte",
    "DYNAMIC_COMPRESSION_EXCESS": "Starke Dynamikkompression",
    "PRE_ECHO": "Vor-Echo",
    "TRANSIENT_SMEARING": "Transienten-Verschmierung",
    "SOFT_SATURATION": "Weiche Sättigung",
    "HEAD_WEAR": "Tonkopf-Verschleiß",
    "RIAA_CURVE_ERROR": "RIAA-Kurven-Abweichung",
    "TRANSPORT_BUMP": "Transportgeräusch",
    "VOCAL_HARSHNESS": "Stimmlich-Rauheit",
    "HISS": "Rauschen",
    "SURFACE_NOISE": "Oberflächenrauschen",
    "DISTORTION": "Verzerrung",
    "SATURATION": "Sättigung",
    "SIBILANCE": "Zischlaute",
}


def _defect_label(key: str) -> str:
    if key in _DEFECT_LABELS:
        return _DEFECT_LABELS[key]
    # Display dict uses lowercase keys; label map uses UPPERCASE → try uppercase lookup
    upper = key.upper()
    if upper in _DEFECT_LABELS:
        return _DEFECT_LABELS[upper]
    # Fallback: prettify
    return key.replace("_", " ").capitalize()


# Physical-unit metric formatters for continuous (location-less) defects.
# Value is already in the unit noted in the comment.
_DEFECT_METRIC_SUFFIX: dict[str, Any] = {
    "noise_level": lambda v: f"\u2212{v:.0f}\u202fdB",  # −45 dB (SNR)
    "hum": lambda v: f"{v:.0f}\u202fHz",  # 50 Hz netzbrummen
    "wow": lambda v: f"{v:.2f}\u202f%",  # 1.20 % pitch-variation
    "flutter": lambda v: f"{v:.2f}\u202f%",  # 0.80 % flutter
    "rumble": lambda v: f"{v:.2f}\u202f%",  # severity %
    "bandwidth_loss": lambda v: f"{v:.2f}\u202f%",
    "reverb_excess": lambda v: f"{v:.2f}\u202f%",
    "dc_offset": lambda v: f"{v:.2f}\u202f%",
    "pitch_drift": lambda v: f"{v:.2f}\u202f%",
    "stereo_imbalance": lambda v: f"{v:.2f}\u202f%",
    "phase_issues": lambda v: f"{v:.2f}\u202f%",
    "print_through": lambda v: f"{v:.2f}\u202f%",
    "dynamic_compression_excess": lambda v: f"{v:.2f}\u202f%",
    "compression_artifacts": lambda v: f"{v:.2f}\u202f%",
    "digital_artifacts": lambda v: f"{v:.2f}\u202f%",
    "quantization_noise": lambda v: f"{v:.2f}\u202f%",
    "jitter_artifacts": lambda v: f"{v:.2f}\u202f%",
    "pre_echo": lambda v: f"{v:.2f}\u202f%",
    "transient_smearing": lambda v: f"{v:.2f}\u202f%",
    "soft_saturation": lambda v: f"{v:.2f}\u202f%",
    "head_wear": lambda v: f"{v:.2f}\u202f%",
    "azimuth_error": lambda v: f"{v:.2f}\u202f%",
    "head_misalignment": lambda v: f"{v:.2f}\u202f%",
    "riaa_curve_error": lambda v: f"{v:.2f}\u202f%",
    "bias_error": lambda v: f"{v:.2f}\u202f%",
    "transport_bump": lambda v: f"{v:.2f}\u202f%",
    "vocal_harshness": lambda v: f"{v:.2f}\u202f%",
    "aliasing": lambda v: f"{v:.2f}\u202f%",
}


def _pill_label(key: str, val: float, n_events: int) -> str:
    """Erstellt pill display text: count suffix for event-defects, physical metric for continuous ones."""
    base = _defect_label(key)
    if n_events > 0:
        return f"{base}  \u00d7\u202f{n_events:,}"
    fmt = _DEFECT_METRIC_SUFFIX.get(key)
    if fmt is not None and val > 0:
        return f"{base}  {fmt(val)}"
    return base


# ---------------------------------------------------------------------------
# Phase prognosis heuristics
# ---------------------------------------------------------------------------
_GRADE_PHASES: dict[str, tuple[int, int, str]] = {
    "excellent": (22, 28, "Leichte Behandlung"),
    "good": (28, 34, "Standard-Restaurierung"),
    "fair": (34, 40, "Intensive Restaurierung"),
    "poor": (38, 45, "Tiefgreifende Restaurierung"),
    "critical": (40, 50, "Vollständige Rekonstruktion"),
    "unknown": (28, 38, "Analyse läuft …"),
}

_GRADE_LABELS_DE: dict[str, tuple[str, str, str]] = {
    "excellent": (_C_GREEN, "Exzellent", "▶  Sehr gut restaurierbar"),
    "good": (_C_GREEN, "Gut", "▶  Gut restaurierbar"),
    "fair": (_C_AMBER, "Mäßig", "▶  Mäßig restaurierbar"),
    "poor": (_C_RED, "Schwierig", "▶  Schwierig restaurierbar"),
    "critical": (_C_RED, "Kritisch", "▶  Sehr stark beschädigt"),
    "unknown": (_C_MUTED, "—", "— wird analysiert …"),
}

# Material human-readable names
_MATERIAL_NAMES: dict[str, str] = {
    "wax_cylinder": "Wachswalze",
    "lacquer_disc": "Lackfolie",
    "shellac": "Schellack",
    "vinyl": "Vinyl",
    "wire_recording": "Drahtband",
    "reel_tape": "Spulenband",
    "tape": "Kassette (Band)",
    "cassette": "Kassette (Band)",
    "dat": "DAT",
    "cd_digital": "CD / Digital",
    "cd": "CD",
    "digital": "Digital",
    "minidisc": "MiniDisc",
    "mp3_low": "MP3 (niedrige Bitrate)",
    "mp3_high": "MP3",
    "damaged_mp3": "MP3 (beschädigt)",
    "aac": "AAC",
    "streaming": "Streaming-Format",
    "unknown": "Unbekannt",
}


# ---------------------------------------------------------------------------
# SongPrognoseWidget
# ---------------------------------------------------------------------------


class SongPrognoseWidget(QWidget):
    """
    Comprehensive pre-flight analysis tab shown after file open.

    Public update interface (all must be called from GUI thread):
      - reset()                       → new file loaded
      - update_material(key, conf)    → after MediumClassifier
      - update_era_genre(decade, genre) → after EraClassifier / GenreClassifier
      - update_restorability(result)  → after RestorabilityEstimator
      - update_defects(defects_dict)  → after DefectScanner (optional, detected)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grade: str = "unknown"
        self._score100: float = 0.0
        self._material: str = "unknown"
        self._chain_label: str = ""
        self._is_multi_generation: bool = False
        self._chain_depth: int = 1  # §v10.131: DepthAwareUI
        self._decade: int | None = None
        self._genre: str = ""
        self._result_obj: Any = None
        self._pill_by_key: dict[str, _DefectPill] = {}
        self._detected_scores: dict[str, float] = {}
        self._detected_locations: dict[str, list] = {}
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background:{_C_BG}; border:none; }}"
            " QScrollBar:vertical { background:rgba(20,24,48,0.7); width:8px; border-radius:4px; }"
            " QScrollBar::handle:vertical { background:rgba(102,126,234,0.4); border-radius:4px; }"
        )

        container = QWidget()
        container.setStyleSheet(f"background:{_C_BG};")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        self._header_lbl = QLabel(t("prognose.header"))
        self._header_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:12pt; font-weight:bold; background:transparent;")
        hdr_row.addWidget(self._header_lbl)
        hdr_row.addStretch()
        self._status_lbl = QLabel(t("prognose.status.idle"))
        self._status_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:9pt; background:transparent;")
        hdr_row.addWidget(self._status_lbl)
        main_layout.addLayout(hdr_row)

        # ── Row 1: Chancen-Score (dial) + Material/Ära/Genre ─────────
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        # Left: big score dial + grade
        dial_card = QFrame()
        dial_card.setStyleSheet(_card_style())
        dial_inner = QVBoxLayout(dial_card)
        dial_inner.setContentsMargins(12, 10, 12, 10)
        dial_inner.setSpacing(4)
        dial_inner.addWidget(_section_label(t("prognose.section.score")))
        dial_row = QHBoxLayout()
        self._dial = _ScoreDial()
        dial_row.addWidget(self._dial)
        dial_label_col = QVBoxLayout()
        self._grade_lbl = QLabel("—")
        self._grade_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:10pt; font-weight:bold; background:transparent;")
        self._grade_sub_lbl = QLabel(t("prognose.placeholder.analyzing"))
        self._grade_sub_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        self._grade_sub_lbl.setWordWrap(True)
        dial_label_col.addWidget(self._grade_lbl)
        dial_label_col.addWidget(self._grade_sub_lbl)
        dial_label_col.addStretch()
        dial_row.addLayout(dial_label_col)
        dial_inner.addLayout(dial_row)
        dial_inner.addStretch()
        row1.addWidget(dial_card, 0)

        # Right: meta info (material, era, genre, SNR)
        meta_card = QFrame()
        meta_card.setStyleSheet(_card_style())
        meta_inner = QVBoxLayout(meta_card)
        meta_inner.setContentsMargins(12, 10, 12, 10)
        meta_inner.setSpacing(6)
        meta_inner.addWidget(_section_label(t("prognose.section.metadata")))

        self._meta_rows: dict[str, QLabel] = {}
        for key, placeholder in [
            ("material", t("prognose.placeholder.detecting")),
            ("era", t("prognose.placeholder.detecting")),
            ("genre", t("prognose.placeholder.detecting")),
            ("snr", t("prognose.placeholder.measuring")),
            ("bandwidth", t("prognose.placeholder.measuring")),
        ]:
            row_w = QHBoxLayout()
            row_w.setSpacing(6)
            row_key = _section_label(key.upper())
            row_key.setFixedWidth(80)
            row_val = QLabel(placeholder)
            row_val.setStyleSheet(f"color:{_C_TEXT}; font-size:9pt; background:transparent;")
            self._meta_rows[key] = row_val
            row_w.addWidget(row_key)
            row_w.addWidget(row_val)
            row_w.addStretch()
            meta_inner.addLayout(row_w)

        meta_inner.addStretch()
        row1.addWidget(meta_card, 1)
        main_layout.addLayout(row1)

        # ── Tiefen-Badge (unsichtbar bis depth ≥ 3) ──
        self._depth_badge = QLabel("")
        self._depth_badge.setStyleSheet(
            "color: #8898BB; font-size: 8pt; background: rgba(102,126,234,0.10);border-radius: 6px; padding: 3px 10px;"
        )
        self._depth_badge.setVisible(False)
        meta_inner.addWidget(self._depth_badge)

        # ── Row 2: MOS prediction bar ─────────────────────────────────
        mos_card = QFrame()
        mos_card.setStyleSheet(_card_style())
        mos_inner = QVBoxLayout(mos_card)
        mos_inner.setContentsMargins(12, 8, 12, 8)
        mos_inner.setSpacing(4)
        mos_inner.addWidget(_section_label(t("prognose.section.expected_quality")))

        mos_bar_row = QHBoxLayout()
        mos_bar_row.setSpacing(10)
        self._mos_bar = QProgressBar()
        self._mos_bar.setRange(0, 500)
        self._mos_bar.setValue(0)
        self._mos_bar.setTextVisible(False)
        self._mos_bar.setFixedHeight(14)
        self._mos_bar.setStyleSheet(
            "QProgressBar { background:rgba(30,37,61,0.9); border-radius:7px; border:none; }"
            f"QProgressBar::chunk {{ background:{_C_BLUE}; border-radius:7px; }}"
        )
        self._mos_val_lbl = QLabel("—")
        self._mos_val_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:10pt; font-weight:bold; background:transparent;")
        self._mos_val_lbl.setFixedWidth(60)
        _lbl_lo = QLabel("1,0")
        _lbl_lo.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        mos_bar_row.addWidget(_lbl_lo, 0)
        mos_bar_row.addWidget(self._mos_bar, 1)
        lbl5 = QLabel("5,0")
        lbl5.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        mos_bar_row.addWidget(lbl5, 0)
        mos_bar_row.addWidget(self._mos_val_lbl, 0)
        mos_inner.addLayout(mos_bar_row)

        self._mos_range_lbl = QLabel("")
        self._mos_range_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        mos_inner.addWidget(self._mos_range_lbl)
        main_layout.addWidget(mos_card)

        # ── Row 3: Defekte ────────────────────────────────────────────
        defect_card = QFrame()
        defect_card.setStyleSheet(_card_style())
        defect_inner = QVBoxLayout(defect_card)
        defect_inner.setContentsMargins(12, 8, 12, 8)
        defect_inner.setSpacing(6)
        defect_inner.addWidget(_section_label(t("prognose.section.defects")))

        self._defect_pills_row = QHBoxLayout()
        self._defect_pills_row.setSpacing(6)
        self._defect_pills_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._defect_placeholder = QLabel(t("prognose.placeholder.scan_after_start"))
        self._defect_placeholder.setStyleSheet(f"color:{_C_MUTED}; font-size:9pt; background:transparent;")
        self._defect_pills_row.addWidget(self._defect_placeholder)
        defect_inner.addLayout(self._defect_pills_row)

        self._defect_recs_lbl = QLabel("")
        self._defect_recs_lbl.setStyleSheet(f"color:{_C_TEXT_DIM}; font-size:8pt; background:transparent;")
        self._defect_recs_lbl.setWordWrap(True)
        defect_inner.addWidget(self._defect_recs_lbl)
        main_layout.addWidget(defect_card)

        # ── Row 4: Phase prognosis ────────────────────────────────────
        phase_card = QFrame()
        phase_card.setStyleSheet(_card_style())
        phase_inner = QVBoxLayout(phase_card)
        phase_inner.setContentsMargins(12, 8, 12, 8)
        phase_inner.setSpacing(4)
        phase_inner.addWidget(_section_label(t("prognose.section.phase_forecast")))

        phase_row = QHBoxLayout()
        phase_row.setSpacing(16)
        self._phase_count_lbl = QLabel("—")
        self._phase_count_lbl.setStyleSheet(
            f"color:{_C_TEXT}; font-size:12pt; font-weight:bold; background:transparent;"
        )
        phase_row.addWidget(self._phase_count_lbl)
        self._phase_desc_lbl = QLabel("—")
        self._phase_desc_lbl.setStyleSheet(f"color:{_C_TEXT_DIM}; font-size:9pt; background:transparent;")
        phase_row.addWidget(self._phase_desc_lbl)
        phase_row.addStretch()
        self._mode_rec_lbl = QLabel("")
        self._mode_rec_lbl.setStyleSheet(f"color:{_C_BLUE}; font-size:9pt; font-weight:bold; background:transparent;")
        phase_row.addWidget(self._mode_rec_lbl)
        phase_inner.addLayout(phase_row)
        main_layout.addWidget(phase_card)

        # ── Row 5: Recommendations ────────────────────────────────────
        rec_card = QFrame()
        rec_card.setStyleSheet(_card_style())
        rec_inner = QVBoxLayout(rec_card)
        rec_inner.setContentsMargins(12, 8, 12, 8)
        rec_inner.setSpacing(4)
        rec_inner.addWidget(_section_label(t("prognose.section.recommendations")))
        self._rec_lbl = QLabel(t("prognose.placeholder.analysis_running"))
        self._rec_lbl.setWordWrap(True)
        self._rec_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:9pt; background:transparent; line-height:160%;")
        rec_inner.addWidget(self._rec_lbl)

        self._preventive_lbl = QLabel("")
        self._preventive_lbl.setWordWrap(True)
        self._preventive_lbl.setStyleSheet(
            f"color:{_C_TEXT_DIM}; font-size:8pt; background:transparent; line-height:150%;"
        )
        rec_inner.addWidget(self._preventive_lbl)
        main_layout.addWidget(rec_card)

        main_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public update interface (GUI thread only)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Called when a new file is loaded — resets all fields."""
        self._grade = "unknown"
        self._score100 = 0.0
        self._material = "unknown"
        self._decade = None
        self._genre = ""
        self._result_obj = None

        self._dial.set_score(0.0, _C_MUTED)
        self._grade_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:10pt; font-weight:bold; background:transparent;")
        self._grade_lbl.setText("—")
        self._grade_sub_lbl.setText(t("prognose.placeholder.analyzing"))
        self.set_runtime_status("analyzing")

        for key in ("material", "era", "genre", "snr", "bandwidth"):
            self._meta_rows[key].setText(t("prognose.placeholder.detecting"))

        self._mos_bar.setValue(0)
        self._mos_val_lbl.setText("—")
        self._mos_range_lbl.setText("")

        self._phase_count_lbl.setText("—")
        self._phase_desc_lbl.setText("—")
        self._mode_rec_lbl.setText("")

        self._rec_lbl.setText(t("prognose.placeholder.analysis_running"))
        self._preventive_lbl.setText("")
        self._defect_recs_lbl.setText("")
        self._clear_defect_pills()
        self._defect_placeholder.setText(t("prognose.placeholder.scan_after_start"))
        self._defect_placeholder.setVisible(True)
        self._pill_by_key = {}
        self._detected_scores = {}
        self._detected_locations = {}

    def update_material(self, material_key: str, confidence: float) -> None:
        """Aktualisiert material row. Call from GUI thread after MediumClassifier."""
        self._material = str(material_key or "unknown")
        self._material_confidence = float(confidence or 1.0)  # §v10.303.9
        name = _MATERIAL_NAMES.get(self._material, self._material)
        pct = int(round(confidence * 100))
        self._meta_rows["material"].setText(
            f"{name}  <span style='color:{_C_MUTED};font-size:8pt;'>({pct}\u202f% Sicherheit)</span>"
        )
        self._meta_rows["material"].setTextFormat(Qt.TextFormat.RichText)
        self._refresh_phase_prognosis()

    def update_chain(self, chain_label: str, is_multi_generation: bool) -> None:
        """§2.46b: Aktualisiert die Tonträgerkette-Anzeige im Prognose-Widget."""
        self._chain_label = str(chain_label or "")
        self._is_multi_generation = bool(is_multi_generation)
        # §v10.131 DepthAwareUI: chain_depth für Farbcodierung speichern
        _stages = self._chain_label.count(" → ") + 1 if self._chain_label else 1
        self._chain_depth = _stages
        self._update_depth_aware_ui()

    def update_era_genre(self, decade: int | None, genre: str | None) -> None:
        """Aktualisiert era/genre rows. Call from GUI thread after EraClassifier."""
        self._decade = decade
        self._genre = str(genre or "")
        era_txt = f"{decade}er" if decade else "—"
        self._meta_rows["era"].setText(era_txt)
        self._meta_rows["genre"].setText(self._genre if self._genre else "—")

    def update_restorability(self, result: Any) -> None:
        """
        Aktualisiert all score fields. Call from GUI thread after RestorabilityEstimator.
        result: RestorabilityResult
        """
        self._result_obj = result
        if result is None:
            return

        score100 = float(getattr(result, "restorability_score", 50.0))
        predicted_mos = float(getattr(result, "predicted_mos", 3.5))
        mos_range = getattr(result, "predicted_mos_range", (predicted_mos - 0.3, predicted_mos + 0.3))
        limiting = list(getattr(result, "limiting_defects", []))
        snr_db = float(getattr(result, "snr_db", 0.0))
        grade = str(getattr(result, "grade", "unknown"))
        recommendations = list(getattr(result, "recommendations", []))

        self._grade = grade
        self._score100 = score100

        color, grade_name, grade_line = _GRADE_LABELS_DE.get(grade, (_C_MUTED, grade, grade))

        # Dial
        self._dial.set_score(score100, color)
        self._grade_lbl.setText(grade_name)
        self._grade_lbl.setStyleSheet(f"color:{color}; font-size:10pt; font-weight:bold; background:transparent;")
        self._grade_sub_lbl.setText(grade_line)
        self.set_runtime_status("ready")

        # MOS bar
        self._mos_bar.setValue(int(round(predicted_mos * 100)))
        bar_color = _C_GREEN if predicted_mos >= 4.0 else (_C_AMBER if predicted_mos >= 3.0 else _C_RED)
        self._mos_bar.setStyleSheet(
            "QProgressBar { background:rgba(30,37,61,0.9); border-radius:7px; border:none; }"
            f"QProgressBar::chunk {{ background:{bar_color}; border-radius:7px; }}"
        )
        self._mos_val_lbl.setText(f"{predicted_mos:.1f} / 5")
        self._mos_val_lbl.setStyleSheet(f"color:{bar_color}; font-size:10pt; font-weight:bold; background:transparent;")
        lo = float(mos_range[0]) if mos_range else predicted_mos - 0.3
        hi = float(mos_range[1]) if len(mos_range) >= 2 else predicted_mos + 0.3  # type: ignore[arg-type]
        self._mos_range_lbl.setText(f"90\u202f%-Wahrscheinlichkeit: {lo:.1f} – {hi:.1f}")

        # SNR
        self._meta_rows["snr"].setText(
            f"{snr_db:.1f}\u202fdB" + ("  ✓" if snr_db >= 20 else ("  △" if snr_db >= 5 else "  ✗"))
        )

        # Defect pills from limiting_defects
        if limiting:
            self._clear_defect_pills()
            self._defect_placeholder.setVisible(False)
            for d in limiting[:6]:
                sev = "high" if any(s in str(d) for s in ("stark", "extrem", "critical")) else "medium"
                pill = _DefectPill(_defect_label(str(d)), sev)
                self._defect_pills_row.addWidget(pill)

        # Recommendations
        if recommendations:
            self._rec_lbl.setText("\n".join(f"• {r}" for r in recommendations))
        else:
            self._rec_lbl.setText("—")

        # Phase prognosis
        self._refresh_phase_prognosis()

        # Print terminal report
        _print_prognose_terminal(
            material=self._material,
            chain_label=self._chain_label,
            decade=self._decade,
            genre=self._genre,
            score100=score100,
            grade=grade,
            predicted_mos=predicted_mos,
            mos_range=(lo, hi),
            snr_db=snr_db,
            limiting_defects=limiting,
            recommendations=recommendations,
            material_confidence=float(getattr(self, "_material_confidence", 1.0) or 1.0),
        )

    def _update_depth_aware_ui(self) -> None:
        """§v10.131 DepthAwareUI: Tiefenabhängige UI-Anpassungen.

        Passt Farben, Warnungen und Schätzwerte an die Chain-Tiefe an.
        Wird von update_chain() und update_restorability() aufgerufen.
        """
        from Aurik10.ui.ui_constants import DepthAwareUI

        _depth = getattr(self, "_chain_depth", 1)
        _conf = float(getattr(self, "_material_confidence", 1.0) or 1.0)
        self._depth_ui = DepthAwareUI(chain_depth=_depth, material_confidence=_conf)

        # ── Farbcodierung des Chancen-Scores ──
        if hasattr(self, "_score_dial"):
            _color = self._depth_ui.quality_color
            self._score_dial.setStyleSheet(f"color: {_color}; background: transparent;")

        # ── Phasen-Schätzung anpassen ──
        _daui_est = self._depth_ui.phase_count_estimate
        if hasattr(self, "_phase_count_lbl") and _depth >= 3:
            _current = self._phase_count_lbl.text()
            if f"({_depth}-stufig" not in _current:
                _note = f"  ({_depth}-stufige Kette)"
                self._phase_count_lbl.setText(_current + _note)

        # ── Deep-Chain-Warnung ──
        if hasattr(self, "_rec_lbl") and _depth >= 4:
            _warn = "⚠️ Tiefe Transfer-Kette — erwartete Einschränkungen bei Brillanz und Transparenz."
            _current_rec = self._rec_lbl.text() if hasattr(self._rec_lbl, "text") else ""
            if _warn not in str(_current_rec):
                self._rec_lbl.setText(str(_current_rec) + "\n" + _warn)

        # ── Tiefen-Badge ──
        if hasattr(self, "_depth_badge") and _depth >= 3:
            _stage_names = {2: "2-stufig", 3: "3-stufig", 4: "4-stufig", 5: "5-stufig"}
            _label = _stage_names.get(_depth, f"{_depth}-stufig")
            _color = self._depth_ui.quality_color
            self._depth_badge.setText(f"◉ {_label}e Transfer-Kette")
            self._depth_badge.setStyleSheet(
                f"color: {_color}; font-size: 8pt; "
                f"background: rgba(102,126,234,0.10);"
                f"border-radius: 6px; padding: 3px 10px;"
            )
            self._depth_badge.setVisible(True)

    def update_depth_from_bridge(self, transfer_generation_count: int) -> None:
        """§Bridge: Empfängt Chain-Depth vom Backend via Bridge-Callback.

        Wird von ModernMainWindow aufgerufen, wenn die Pre-Analysis
        die transfer_generation_count liefert. Kein Direktimport!
        """
        if transfer_generation_count > 0:
            self._chain_depth = max(self._chain_depth, transfer_generation_count)
            self._update_depth_aware_ui()

    def update_from_bridge_calibration(self, calib_data: dict) -> None:
        """§Bridge: ZENTRALER Einstiegspunkt für ALLE Kalibrierungsdaten.

        Empfängt ein Dict von BridgeCalibrationData.to_frontend_dict()
        und aktualisiert ALLE depth-abhängigen UI-Elemente auf einmal.

        Dies ist der EINZIGE Pfad für Backend→Frontend-Kalibrierungsdaten.
        Kein Direktimport von backend.* Modulen!
        """
        if not isinstance(calib_data, dict):
            return

        # Kernwerte übernehmen
        self._chain_depth = max(self._chain_depth, int(calib_data.get("transfer_chain_depth", 1)))

        # DepthAwareUI neu laden
        self._update_depth_aware_ui()

        # Quality-Color aus Bridge übernehmen (überschreibt lokale Berechnung)
        _bridge_color = calib_data.get("quality_color", "")
        if _bridge_color and hasattr(self, "_score_dial"):
            self._score_dial.setStyleSheet(f"color: {_bridge_color}; background: transparent;")

        # Deep-Chain-Warnung aus Bridge
        _warning = calib_data.get("deep_chain_warning", "")
        if _warning and hasattr(self, "_rec_lbl"):
            _cur = str(self._rec_lbl.text() if hasattr(self._rec_lbl, "text") else "")
            if _warning not in _cur:
                self._rec_lbl.setText(_cur + "\n" + _warning)

    def update_defects(self, defects: dict) -> None:
        """
        Aktualisiert defect pills from DefectScanner result (BatchProcessingThread).

        status='detected'   → (re)build pills with counts / physical metrics.
        status='correcting' → update existing pills; mark resolved when phase fixed the defect.
        """
        status = defects.get("status")
        if status not in ("detected", "correcting"):
            return

        _locs: dict = defects.get("_locations", {})

        if status == "detected":
            self.set_runtime_status("detected")
            # Collect significant defects
            significant: list[tuple[str, float]] = []
            for k, v in defects.items():
                if k in ("status", "_no_anim", "_locations", "_channel_locations", "_event_counts", "_severity_raw"):
                    continue
                if isinstance(v, (int, float)) and float(v) > 0.15:
                    significant.append((str(k), float(v)))

            if not significant:
                return

            significant.sort(key=lambda x: x[1], reverse=True)
            self._clear_defect_pills()
            self._defect_placeholder.setVisible(False)

            self._detected_scores = {}
            self._detected_locations = {}
            self._pill_by_key = {}

            _event_counts: dict = defects.get("_event_counts", {})
            _severity_raw: dict = defects.get("_severity_raw", {})

            for key, sev_val in significant[:8]:
                self._detected_scores[key] = sev_val
                self._detected_locations[key] = list(_locs.get(key, []))
                # Prefer authoritative metadata count; fall back to location list length
                n_events = _event_counts.get(key, len(self._detected_locations[key]))
                # Use normalised 0-1 severity for pill colour; fall back per-key scaling
                raw_sev = _severity_raw.get(key)
                if raw_sev is None:
                    # sev_val is already scaled; try best-effort normalisation
                    raw_sev = min(1.0, sev_val / 100.0)
                sev = "high" if raw_sev > 0.6 else ("medium" if raw_sev > 0.25 else "low")
                label = _pill_label(key, sev_val, n_events)
                pill = _DefectPill(label, sev)
                self._defect_pills_row.addWidget(pill)
                self._pill_by_key[key] = pill

        else:  # status == "correcting"
            self.set_runtime_status("correcting")
            if not getattr(self, "_pill_by_key", None):
                return
            # §11.4 Active defect keys from the current pipeline phase
            _active_keys: set[str] = set(defects.get("_active_defects") or [])
            for key, pill in self._pill_by_key.items():
                current_val = defects.get(key)
                if current_val is None:
                    continue
                current_val = float(current_val)
                initial_val = self._detected_scores.get(key, 0.0)
                if initial_val <= 0:
                    continue
                ratio = current_val / initial_val
                if ratio < 0.15:
                    # Phase hat diesen Defekt behoben → Zähler/Metrik entfernen, grün markieren
                    pill.set_resolved()
                elif key in _active_keys:
                    # Blu-ray metallic highlight: this defect is targeted by the running phase
                    pill.set_active()
                else:
                    # Not active right now: restore default severity style, update metric
                    pill.clear_active()
                    n_initial = len(self._detected_locations.get(key, []))
                    n_current = int(n_initial * ratio) if n_initial > 0 else 0
                    new_label = _pill_label(key, current_val, n_current)
                    pill.update_metric(new_label)

    def set_runtime_status(self, state: str) -> None:
        """Setzt den Laufzeit-Status der Analysekarte konsistent inkl. Farbcodierung."""
        _state = str(state or "").strip().lower()
        _map: dict[str, tuple[str, str]] = {
            "idle": (t("prognose.status.idle"), _C_MUTED),
            "analyzing": (t("prognose.status.analyzing"), _C_MUTED),
            "ready": (t("prognose.status.ready"), _C_GREEN),
            "detected": (t("prognose.status.detected"), _C_AMBER),
            "correcting": (t("prognose.status.correcting"), _C_BLUE),
            "completed": (t("prognose.status.completed"), _C_GREEN),
            "warning": (t("prognose.status.warning"), _C_AMBER),
            "error": (t("prognose.status.error"), _C_RED),
            "cancelled": (t("prognose.status.cancelled"), _C_RED),
        }
        _text, _color = _map.get(_state, _map["idle"])
        self._status_lbl.setText(_text)
        self._status_lbl.setStyleSheet(f"color:{_color}; font-size:9pt; background:transparent;")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_defect_pills(self) -> None:
        for i in reversed(range(self._defect_pills_row.count())):
            item = self._defect_pills_row.itemAt(i)
            if item and item.widget() is not self._defect_placeholder:
                w = item.widget()
                if w:
                    self._defect_pills_row.removeWidget(w)
                    w.deleteLater()
        self._pill_by_key = {}
        self._detected_scores = {}
        self._detected_locations = {}

    def _refresh_phase_prognosis(self) -> None:
        lo, hi, desc = _GRADE_PHASES.get(self._grade, _GRADE_PHASES["unknown"])

        # Analog materials get more phases
        if self._material in ("shellac", "wax_cylinder", "lacquer_disc", "wire_recording"):
            lo += 4
            hi += 6
            desc += " (historisches Analog-Material)"
        elif self._material in ("vinyl", "reel_tape", "tape", "cassette"):
            lo += 2
            hi += 3

        # §2.46b: Multi-Generation-Ketten brauchen mehr Phasen
        # Jede zusätzliche Kettenstufe fügt eigene Defekte hinzu, die separate
        # Phasen erfordern. Faustregel: +2 Phasen pro Zwischenstufe.
        if self._is_multi_generation and self._chain_label:
            _stages = self._chain_label.count(" → ") + 1
            if _stages >= 3:
                lo += (_stages - 1) * 2
                hi += (_stages - 1) * 2
                desc += f" ({_stages}-stufige Kette)"

        self._phase_count_lbl.setText(f"~ {lo}–{hi}")
        self._phase_desc_lbl.setText(desc)

        # Mode recommendation
        if self._grade in ("excellent", "good"):
            self._mode_rec_lbl.setText(t("prognose.mode.studio_recommended"))
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_GREEN}; font-size:9pt; font-weight:bold; background:transparent;"
            )
        elif self._grade in ("fair",):
            self._mode_rec_lbl.setText(t("prognose.mode.restoration_recommended"))
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_AMBER}; font-size:9pt; font-weight:bold; background:transparent;"
            )
        elif self._grade in ("poor", "critical"):
            self._mode_rec_lbl.setText(t("prognose.mode.restoration_safe"))
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_RED}; font-size:9pt; font-weight:bold; background:transparent;"
            )

    def set_recommended_mode(self, mode: str) -> None:
        """Override the grade-based mode recommendation with the authoritative result
        from _recommend_mode_from_ui_context() (which considers material, defect severity,
        era, and genre in addition to the plain restorability grade).  Called by
        ModernMainWindow._apply_mode_recommendation_visuals() once pre-analysis is final."""
        if mode == "STUDIO_2026":
            self._mode_rec_lbl.setText(t("prognose.mode.studio_recommended_alt"))
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_GREEN}; font-size:9pt; font-weight:bold; background:transparent;"
            )
        else:  # RESTORATION
            if self._grade in ("poor", "critical"):
                self._mode_rec_lbl.setText(t("prognose.mode.restoration_safe"))
                self._mode_rec_lbl.setStyleSheet(
                    f"color:{_C_RED}; font-size:9pt; font-weight:bold; background:transparent;"
                )
            else:
                self._mode_rec_lbl.setText(t("prognose.mode.restoration_recommended"))
                self._mode_rec_lbl.setStyleSheet(
                    f"color:{_C_AMBER}; font-size:9pt; font-weight:bold; background:transparent;"
                )

    def set_preventive_actions(self, actions: list[str]) -> None:
        """Zeigt an: post-run preventive DSP actions in the analysis tab."""
        if not actions:
            self._preventive_lbl.setText("")
            return
        _txt = "\n".join(f"🛡 {a}" for a in actions[:3])
        self._preventive_lbl.setText(_txt)


# ---------------------------------------------------------------------------
# Terminal report (ANSI colour output)
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_GREEN = "\033[32m"
_ANSI_AMBER = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_BLUE = "\033[34;1m"
_ANSI_CYAN = "\033[36m"
_ANSI_DIM = "\033[2m"


def _color_grade(grade: str, text: str) -> str:
    c = {
        "excellent": _ANSI_GREEN,
        "good": _ANSI_GREEN,
        "fair": _ANSI_AMBER,
        "poor": _ANSI_RED,
        "critical": _ANSI_RED,
    }.get(grade, "")
    return f"{c}{_ANSI_BOLD}{text}{_ANSI_RESET}" if c else text


def _print_prognose_terminal(
    *,
    material: str,
    chain_label: str = "",
    decade: int | None,
    genre: str,
    score100: float,
    grade: str,
    predicted_mos: float,
    mos_range: tuple[float, float],
    snr_db: float,
    limiting_defects: list[str],
    recommendations: list[str],
    material_confidence: float = 1.0,
) -> None:
    """Gibt aus: a colored prognosis report to the logger (INFO level)."""
    grade_de = {
        "excellent": "Exzellent",
        "good": "Gut",
        "fair": "Mäßig",
        "poor": "Schwierig",
        "critical": "Kritisch",
    }.get(grade, grade)
    mat_name = _MATERIAL_NAMES.get(material, material)
    lo, hi, _ = _GRADE_PHASES.get(grade, _GRADE_PHASES["unknown"])
    # §v10.303.9: Confidence-bewusste Phasen-Schätzung
    if material_confidence < 0.25:
        _mult = 0.40
    elif material_confidence < 0.35:
        _mult = 0.55
    elif material_confidence < 0.50:
        _mult = 0.75
    else:
        _mult = 1.0
    lo = max(10, int(lo * _mult))
    hi = max(12, int(hi * _mult))

    lines = [
        "",
        f"{_ANSI_BOLD}{_ANSI_BLUE}╔══════════════════════════════════════════════════════╗{_ANSI_RESET}",
        f"{_ANSI_BOLD}{_ANSI_BLUE}║        Aurik 10 — Song-Prognose (Pre-Flight)          ║{_ANSI_RESET}",
        f"{_ANSI_BOLD}{_ANSI_BLUE}╚══════════════════════════════════════════════════════╝{_ANSI_RESET}",
        f"  {_ANSI_DIM}Trägermedium{_ANSI_RESET}    : {mat_name}",
    ]
    # §2.46b: Tonträgerkette anzeigen wenn mehrstufig
    if chain_label and " → " in chain_label:
        lines.append(f"  {_ANSI_DIM}Tonträgerkette{_ANSI_RESET} : {_ANSI_CYAN}{chain_label}{_ANSI_RESET}")
    lines.extend(
        [
            f"  {_ANSI_DIM}Aufnahme-Ära{_ANSI_RESET}    : {f'{decade}er' if decade else '—'}",
            f"  {_ANSI_DIM}Genre{_ANSI_RESET}           : {genre if genre else '—'}",
            f"  {_ANSI_DIM}SNR{_ANSI_RESET}             : {snr_db:.1f} dB (source_snr_db)",
            "",
            f"  {_ANSI_BOLD}Chancen-Score{_ANSI_RESET}   : {_color_grade(grade, f'{score100:.0f} / 100  ({grade_de})')}",
            f"  {_ANSI_BOLD}MOS-Prognose{_ANSI_RESET}    : {predicted_mos:.2f}  [{mos_range[0]:.1f}–{mos_range[1]:.1f}]",
            f"  {_ANSI_BOLD}Phasen-Schätzung{_ANSI_RESET}: {lo}–{hi} Phasen",
            "",
        ]
    )

    if limiting_defects:
        lines.append(f"  {_ANSI_AMBER}Hauptschäden{_ANSI_RESET}:")
        for d in limiting_defects:
            lines.append(f"    • {_defect_label(d)}")
        lines.append("")

    if recommendations:
        lines.append(f"  {_ANSI_CYAN}Empfehlungen{_ANSI_RESET}:")
        for r in recommendations:
            lines.append(f"    • {r}")
        lines.append("")

    lines.append(f"{_ANSI_DIM}  ─────────────────────────────────────────────────────{_ANSI_RESET}")

    report = "\n".join(lines)
    logger.info("PROGNOSE_REPORT:\n%s", report)
