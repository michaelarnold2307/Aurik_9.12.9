"""
Musical Goals Bar Chart Widget für Aurik 10
Zeigt alle 15 musikalischen Qualitätsziele als lesbares Balkendiagramm.

Rein PyQt5-basiert (kein Matplotlib) — CPU-leicht, sofort responsiv.
Vollständig laienfreundlich: Deutsche Labels, Farb-Kodierung, Prozentwerte, Tooltips.
"""
# pylint: disable=c-extension-no-member

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from Aurik10.i18n import t

if TYPE_CHECKING:
    # isort: skip
    # Type-only imports for static analysis
    from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, QTimer  # type: ignore
    from PyQt5.QtGui import (
        QBrush,
        QColor,
        QFont,
        QMouseEvent,
        QPainter,
        QPaintEvent,
        QPen,
        QPolygonF,
    )  # type: ignore
    from PyQt5.QtWidgets import QScrollArea, QSizePolicy, QToolTip, QWidget  # type: ignore
else:  # pyright: ignore[reportUnreachable]
    try:
        from PyQt5 import QtCore as _QtCore
        from PyQt5 import QtGui as _QtGui
        from PyQt5 import QtWidgets as _QtWidgets

        QPointF = _QtCore.QPointF
        QRectF = _QtCore.QRectF
        QSize = _QtCore.QSize
        Qt = _QtCore.Qt
        QTimer = _QtCore.QTimer

        QBrush = _QtGui.QBrush
        QColor = _QtGui.QColor
        QFont = _QtGui.QFont
        QMouseEvent = _QtGui.QMouseEvent
        QPainter = _QtGui.QPainter
        QPaintEvent = _QtGui.QPaintEvent
        QPen = _QtGui.QPen
        QPolygonF = _QtGui.QPolygonF

        QSizePolicy = _QtWidgets.QSizePolicy
        QToolTip = _QtWidgets.QToolTip
        QWidget = _QtWidgets.QWidget
        QScrollArea = _QtWidgets.QScrollArea
    except Exception:
        # Fall back to Any so static type-checkers stop emitting assignment/type errors
        QPointF = Any
        QRectF = Any
        QSize = Any
        Qt = Any
        QTimer = Any

        QBrush = Any
        QColor = Any
        QFont = Any
        QMouseEvent = Any
        QPainter = Any
        QPaintEvent = Any
        QPen = Any
        QPolygonF = Any

        QSizePolicy = Any
        QToolTip = Any
        QWidget = Any
        QScrollArea = Any

# ─────────────────────────────────────────────────────────────────────────────
# Datmodell: Die 15 Musical Goals
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GoalEntry:
    """Ein einzelnes Musical Goal mit allen relevanten Metadaten."""

    key: str  # interner Schlüssel
    label: str  # Vollbezeichnung (Deutsch)
    threshold: float  # Pflicht-Schwellwert laut Spec
    score: float = 0.0  # aktueller Wert ∈ [0, 1]
    adaptive_threshold: float = -1.0  # adaptierter Schwellwert (–1 = nicht gesetzt)
    applicable: bool = True  # GoalApplicabilityFilter
    inapplicable_reason: str = ""  # Deutsch: Warum nicht messbar?
    synthesized: bool = False  # EraAuthenticPerceptualCompletion aktiv?
    adaptation_reason: str = ""  # Deutsch: Warum Schwellwert angepasst?


# Standard-Goals gemäß §1.2 der Spec — vollständige deutsche Bezeichnungen
DEFAULT_GOALS: list[GoalEntry] = [
    GoalEntry("brillanz", "Brillanz", 0.85),
    GoalEntry("waerme", "Wärme", 0.80),
    GoalEntry("natuerlichkeit", "Natürlichkeit", 0.90),
    GoalEntry("authentizitaet", "Authentizität", 0.88),
    GoalEntry("emotionalitaet", "Emotionalität", 0.87),
    GoalEntry("transparenz", "Transparenz", 0.89),
    GoalEntry("bass_kraft", "Bass-Kraft", 0.85),
    GoalEntry("groove", "Groove", 0.88),
    GoalEntry("spatial_depth", "Raumtiefe", 0.75),
    GoalEntry("timbre_authentizitaet", "Klangfarbe", 0.87),
    GoalEntry("tonal_center", "Tonart-Treue", 0.95),
    GoalEntry("micro_dynamics", "Fein-Dynamik", 0.92),
    GoalEntry("separation_fidelity", "Trennschärfe", 0.82),
    GoalEntry("artikulation", "Artikulation", 0.85),
    GoalEntry("transient_energie", "Transienten-Energie", 0.88),
]

# Farb-Konstanten
COLOR_PASS = QColor(76, 175, 80, 220)  # Grün  — Ziel erfüllt
COLOR_WARN = QColor(255, 193, 7, 220)  # Gelb  — knapp am Limit (< +0.04)
COLOR_FAIL = QColor(244, 67, 54, 220)  # Rot   — Ziel unterschritten
COLOR_NA = QColor(100, 110, 130, 140)  # Grau  — nicht anwendbar
COLOR_SYNTH = QColor(220, 130, 240, 200)  # Lila  — era-authentisch ergänzt
COLOR_BAR_BG = QColor(30, 38, 58, 180)  # Balken-Hintergrund
COLOR_THRESH = QColor(255, 255, 255, 120)  # Schwellwert-Linie
COLOR_TEXT = QColor(190, 200, 218)  # Standard-Textfarbe
COLOR_DIM = QColor(120, 130, 150, 160)  # Gedimmter Text

# Vollständige Farbpalette für Bar BG je nach Status
_BAR_BG_PASS = QColor(76, 175, 80, 40)
_BAR_BG_WARN = QColor(255, 193, 7, 35)
_BAR_BG_FAIL = QColor(244, 67, 54, 35)
_BAR_BG_NA = QColor(60, 70, 90, 60)


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Widget — Balkendiagramm statt Radar
# ─────────────────────────────────────────────────────────────────────────────


class MusicalGoalsRadarWidget(QWidget):
    """
    Horizontales Balkendiagramm für die 15 Musical Goals.

    Zeigt pro Ziel:
    - Vollständiger deutscher Name
    - Farbiger Balken (Score-Länge) mit Schwellwert-Markierung
    - Prozentwert als Zahl (lesbar)
    - Status-Icon (✅ / ⚠ / ✗ / –)
    - Farb-Kodierung: Grün/Gelb/Rot/Grau

    Vor Restaurierung: freundliche Platzhaltermeldung.
    Hover-Tooltips: erklären das Ziel auf Deutsch.
    """

    # Geometry constants
    _LABEL_W = 98  # px width of the goal name column
    _SCORE_W = 34  # px width of the "87 %" score column
    _ICON_W = 18  # px width of status icon
    _ROW_H = 16  # px height per goal row
    _ROW_GAP = 2  # px gap between rows
    _PAD_X = 8  # horizontal outer padding
    _PAD_TOP = 28  # top padding (header)
    _PAD_BOT = 22  # bottom padding (legend)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

        self._goals: list[GoalEntry] = copy.deepcopy(DEFAULT_GOALS)
        self._hovered_idx: int = -1
        self._has_data: bool = False  # False = vor erster Restaurierung
        self._anim_scores: dict[str, float] = {}

        # Animation
        # inner imports were consolidated to module top to satisfy linters

    def paintEvent(self, event: QPaintEvent) -> None:
        """Zeichnet das Balkendiagramm oder den Platzhaltertext."""
        _ = event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if not self._has_data:
            self._draw_placeholder(p)
        else:
            self._draw_header(p)
            self._draw_bars(p)
            self._draw_footer_legend(p)

        p.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Zeigt Platzhaltertext wenn noch keine Restaurierung stattgefunden hat."""
        w, h = float(self.width()), float(self.height())
        # Sanfte Linie als Rahmen-Hint
        painter.setPen(QPen(QColor(80, 100, 140, 60), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(4, 4, w - 8, h - 8), 8, 8)

        # Icon
        font_icon = QFont(self.font().family(), 20)
        painter.setFont(font_icon)
        painter.setPen(QPen(QColor(80, 100, 140, 100)))
        painter.drawText(QRectF(0, h / 2 - 54, w, 40), Qt.AlignmentFlag.AlignCenter, "🎵")

        # Haupttext
        font_main = QFont(self.font().family(), 8, QFont.Weight.Bold)
        painter.setFont(font_main)
        painter.setPen(QPen(QColor(130, 150, 190, 180)))
        painter.drawText(
            QRectF(10, h / 2 - 14, w - 20, 22),
            Qt.AlignmentFlag.AlignCenter,
            t("ui.not_measured"),
        )

        # Subtext
        font_sub = QFont(self.font().family(), 7)
        painter.setFont(font_sub)
        painter.setPen(QPen(QColor(100, 120, 160, 140)))
        painter.drawText(
            QRectF(10, h / 2 + 10, w - 20, 36),
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
            t("ui.not_measured_sub"),
        )

        # Goal-Namen als gedimmte Vorschau
        font_prev = QFont(self.font().family(), 6)
        painter.setFont(font_prev)
        painter.setPen(QPen(QColor(80, 95, 130, 80)))
        names = [g.label for g in self._goals]
        half = len(names) // 2
        col_w = (w - 20) / 2
        for col, chunk in enumerate([names[:half], names[half:]]):
            for row, name in enumerate(chunk):
                x = 10 + col * col_w
                y = h / 2 + 48 + row * 10
                painter.drawText(
                    QRectF(x, y, col_w, 10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"· {name}"
                )

    def _draw_header(self, painter: QPainter) -> None:
        """Kopfzeile: Statusübersicht (✅ N  ⚠ N  ✗ N)."""
        n_pass = n_warn = n_fail = n_na = 0
        for g in self._goals:
            if not g.applicable:
                n_na += 1
                continue
            t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
            if g.score >= t + 0.04:
                n_pass += 1
            elif g.score >= t:
                n_warn += 1
            else:
                n_fail += 1

        w = float(self.width())
        font = QFont(self.font().family(), 7, QFont.Weight.Bold)
        painter.setFont(font)

        parts: list[tuple[str, str, QColor]] = [
            (str(n_pass), "pass", COLOR_PASS),
            (str(n_warn), "warn", COLOR_WARN),
            (str(n_fail), "fail", COLOR_FAIL),
        ]
        if n_na > 0:
            parts.append((str(n_na), "na", COLOR_NA))

        x = float(self._PAD_X)
        y_top = 6.0
        h_row = 18.0
        _icon_r = 5.0
        for label, kind, col in parts:
            self._draw_icon(painter, x + _icon_r, y_top + h_row * 0.5, _icon_r, kind, col)
            painter.setPen(QPen(col))
            painter.drawText(
                QRectF(x + _icon_r * 2 + 3, y_top, 32, h_row),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            x += 48
            x += 52

        # Thin separator line
        painter.setPen(QPen(QColor(80, 100, 150, 60), 1))
        painter.drawLine(QPointF(self._PAD_X, self._PAD_TOP - 4), QPointF(w - self._PAD_X, self._PAD_TOP - 4))

    @staticmethod
    def _draw_icon(painter: QPainter, cx: float, cy: float, r: float, kind: str, col: QColor) -> None:
        """Zeichnet ein kleines Icon (gefüllter Kreis) für die Legende."""
        painter.setBrush(QBrush(col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def _row_y(self, idx: int) -> float:
        """Y-Position einer Goal-Zeile."""
        return float(self._PAD_TOP + idx * (self._ROW_H + self._ROW_GAP))

    def _bar_rect(self, idx: int) -> QRectF:
        """Rechteck für den Goal-Balken."""
        y = self._row_y(idx)
        bar_x = self._PAD_X + self._ICON_W + self._LABEL_W + 4
        bar_w = self.width() - bar_x - self._PAD_X - self._SCORE_W - 4
        return QRectF(bar_x, y, bar_w, self._ROW_H)

    def _draw_bars(self, painter: QPainter) -> None:
        """Zeichnet alle 15 Goal-Balken."""
        font_label = QFont(self.font().family(), 6, QFont.Weight.Bold)
        font_score = QFont(self.font().family(), 6)

        for idx, g in enumerate(self._goals):
            y = self._row_y(idx)
            bar_rect = self._bar_rect(idx)
            t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold

            # Determine status color — §GUI-T1: pure Entscheidungslogik nutzen
            _state = goal_bar_state(g.score, t, applicable=g.applicable, synthesized=g.synthesized)
            _state_color = {
                "na": COLOR_NA,
                "synth": COLOR_SYNTH,
                "pass": COLOR_PASS,
                "warn": COLOR_WARN,
                "fail": COLOR_FAIL,
            }
            _state_bg = {
                "na": _BAR_BG_NA,
                "synth": QColor(200, 100, 230, 35),
                "pass": _BAR_BG_PASS,
                "warn": _BAR_BG_WARN,
                "fail": _BAR_BG_FAIL,
            }
            color = _state_color[_state]
            bar_bg = _state_bg[_state]

            # Hover highlight
            is_hovered = idx == self._hovered_idx
            if is_hovered:
                painter.setBrush(QBrush(QColor(255, 255, 255, 12)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(
                    QRectF(self._PAD_X, y - 1, self.width() - 2 * self._PAD_X, self._ROW_H + 2), 3, 3
                )

            # ── Status icon (vector) ──
            _ik = _state
            _icx = self._PAD_X + self._ICON_W * 0.5
            _icy = y + self._ROW_H * 0.5
            _ir = min(self._ICON_W, self._ROW_H) * 0.38
            self._draw_icon(painter, _icx, _icy, _ir, _ik, color)

            # ── Goal name ──
            painter.setFont(font_label)
            label_color = color if g.applicable else COLOR_DIM
            painter.setPen(QPen(label_color))
            name_text = g.label if g.applicable else f"({g.label})"
            painter.drawText(
                QRectF(self._PAD_X + self._ICON_W, y, self._LABEL_W, self._ROW_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                name_text,
            )

            # ── Bar background ──
            painter.setBrush(QBrush(bar_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_rect, 3, 3)

            # ── Bar fill ──
            if g.applicable and g.score > 0.001:
                fill_w = bar_rect.width() * min(1.0, g.score)
                fill_rect = QRectF(bar_rect.x(), bar_rect.y(), fill_w, bar_rect.height())
                fill_color = QColor(color)
                fill_color.setAlpha(180)
                painter.setBrush(QBrush(fill_color))
                painter.drawRoundedRect(fill_rect, 3, 3)

            # ── Threshold marker (vertical line) ──
            if g.applicable:
                thresh_x = bar_rect.x() + bar_rect.width() * min(1.0, t)
                painter.setPen(QPen(COLOR_THRESH, 1.5))
                painter.drawLine(
                    QPointF(thresh_x, bar_rect.y() - 1),
                    QPointF(thresh_x, bar_rect.y() + bar_rect.height() + 1),
                )

            # ── Score value ──
            painter.setFont(font_score)
            painter.setPen(QPen(color if g.applicable else COLOR_DIM))
            score_str = f"{int(g.score * 100)} %" if g.applicable else "—"
            score_x = bar_rect.x() + bar_rect.width() + 3
            painter.drawText(
                QRectF(score_x, y, self._SCORE_W, self._ROW_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                score_str,
            )

    def _draw_footer_legend(self, painter: QPainter) -> None:
        """Legende am unteren Rand."""
        y = float(self.height()) - self._PAD_BOT + 4
        font = QFont(self.font().family(), 6)
        painter.setFont(font)

        items = [
            (COLOR_PASS, "Erfüllt"),
            (COLOR_WARN, "Knapp"),
            (COLOR_FAIL, "Nicht erreicht"),
            (COLOR_THRESH, "Zielwert"),
        ]
        x = float(self._PAD_X)
        for color, text in items:
            # Color dot
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(x, y + 2, 7, 7))
            painter.setPen(QPen(QColor(160, 170, 190, 180)))
            painter.drawText(
                QRectF(x + 10, y, 58, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text
            )
            x += 66

    # ──────────────────────────── Tooltip / Hover ─────────────────────────

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Aktualisiert den Hover-Index und zeigt den Tooltip für das ausgewählte Goal."""
        if not self._has_data:
            super().mouseMoveEvent(event)
            return

        py = float(event.pos().y())
        found = -1
        for idx in range(len(self._goals)):
            row_y = self._row_y(idx)
            if row_y <= py <= row_y + self._ROW_H:
                found = idx
                break

        if found != self._hovered_idx:
            self._hovered_idx = found
            self.update()

        if found >= 0:
            QToolTip.showText(event.globalPos(), self._tooltip_text(self._goals[found]), self)
        else:
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        """Setzt den Hover-Index zurück wenn die Maus das Widget verlässt."""
        self._hovered_idx = -1
        self.update()
        super().leaveEvent(event)

    @staticmethod
    def _tooltip_text(g: GoalEntry) -> str:
        """Laienverständlicher Tooltip-Text für ein Goal."""
        # Human-readable descriptions per goal
        _DESCRIPTIONS: dict[str, str] = {
            "brillanz": "Helligkeit und Klarheit in den Höhen (Obertöne, Luftigkeit).",
            "waerme": "Wärme und Fülle im Bassbereich — angenehmes Klangfundament.",
            "natuerlichkeit": "Klingt die Aufnahme natürlich und unverarbeitet?",
            "authentizitaet": "Entspricht der Klang dem Original-Charakter der Aufnahme?",
            "emotionalitaet": "Transportiert die Restaurierung die emotionale Intensität?",
            "transparenz": "Sind alle Instrumente klar voneinander trennbar?",
            "bass_kraft": "Kraft und Tiefe im Bass inkl. fehlende Grundtöne.",
            "groove": "Rhythmisches Timing — keine Verschmierung von Transienten.",
            "spatial_depth": "Stereobreite und Raumtiefe (nur bei Stereo-Aufnahmen).",
            "timbre_authentizitaet": "Klangfarbe der Instrumente — klingt die Oboe noch wie eine Oboe?",
            "tonal_center": "Tonart und Stimmung — kein Pitch-Shift durch die Restaurierung.",
            "micro_dynamics": "Lautstärke-Feinstruktur — Atemzüge, Pianissimo-Stellen.",
            "separation_fidelity": "Stimme und Instrumente bleiben getrennt (kein Matsch).",
            "artikulation": "Ansätze, Transienten und Konsonanten bleiben scharf.",
            "transient_energie": "Attack und Einschwingenergie bleiben erhalten.",
        }
        lines: list[str] = [f"<b>{g.label}</b>"]
        if _desc := _DESCRIPTIONS.get(g.key, ""):
            lines.append(f"<br><i>{_desc}</i>")

        if not g.applicable:
            lines.append("<br><br>⚪ <i>Nicht messbar für diese Aufnahme.</i>")
            if g.inapplicable_reason:
                lines.append(f"<br>{g.inapplicable_reason}")
            return "".join(lines)

        t_eff = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
        score_pct = int(g.score * 100)
        thresh_pct = int(t_eff * 100)

        if g.score >= t_eff + 0.04:
            status = "✅ Exzellent — deutlich über dem Zielwert"
            color = "#4CAF50"
        elif g.score >= t_eff:
            status = "🟡 Bestanden — knapp am Zielwert"
            color = "#FFC107"
        else:
            status = "❌ Unter Zielwert"
            color = "#F44336"

        lines.append(f'<br><br><span style="color:{color}"><b>{status}</b></span>')
        lines.append(f"<br>Erreicht: <b>{score_pct} %</b> &nbsp;|&nbsp; Zielwert: <b>{thresh_pct} %</b>")

        if g.adaptive_threshold >= 0 and abs(g.adaptive_threshold - g.threshold) > 0.005:
            orig_pct = int(g.threshold * 100)
            diff = int((g.adaptive_threshold - g.threshold) * 100)
            sign = "+" if diff >= 0 else ""
            lines.append(f"<br><i>Zielwert angepasst: {sign}{diff} % gegenüber Standard ({orig_pct} %)</i>")
        if g.adaptation_reason:
            lines.append(f"<br>{g.adaptation_reason}")
        if g.synthesized:
            lines.append("<br>✦ <i>Frequenzbereich wurde era-authentisch ergänzt.</i>")

        return "".join(lines)

    def sizeHint(self) -> QSize:
        """Gibt die bevorzugte Widget-Größe basierend auf der Zeilenzahl zurück."""
        total_h = self._PAD_TOP + len(DEFAULT_GOALS) * (self._ROW_H + self._ROW_GAP) + self._PAD_BOT
        return QSize(300, total_h)

    def minimumSizeHint(self) -> QSize:
        """Gibt die minimale Widget-Größe zurück."""
        total_h = self._PAD_TOP + len(DEFAULT_GOALS) * (self._ROW_H + self._ROW_GAP) + self._PAD_BOT
        return QSize(220, total_h)

    def update_scores(
        self,
        scores: dict[str, float] | None = None,
        adaptive_thresholds: dict[str, float] | None = None,
        applicable_goals: set[str] | None = None,
        inapplicable_reasons: dict[str, str] | None = None,
        synthesized_goals: set[str] | None = None,
        adaptation_reasons: dict[str, str] | None = None,
    ) -> None:
        """Aktualisiert die internen Goal-Entries und triggert ein Repaint.

        Diese Methode wird von `apply_restoration_result` verwendet. Felder
        werden nur gesetzt, wenn sie im übergebenen Dict vorhanden sind (graceful).
        """
        scores = scores or {}
        adaptive_thresholds = adaptive_thresholds or {}
        inapplicable_reasons = inapplicable_reasons or {}
        adaptation_reasons = adaptation_reasons or {}
        synthesized_goals = synthesized_goals or set()

        key_to_goal = {g.key: g for g in self._goals}
        for key, goal in key_to_goal.items():
            if key in scores:
                # clamp to [0,1]
                goal.score = float(max(0.0, min(1.0, float(scores[key]))))
            if key in adaptive_thresholds:
                goal.adaptive_threshold = float(adaptive_thresholds[key])
            if applicable_goals is not None:
                goal.applicable = key in applicable_goals
            if key in inapplicable_reasons:
                goal.inapplicable_reason = inapplicable_reasons.get(key, "") or ""
            goal.synthesized = key in synthesized_goals
            if key in adaptation_reasons:
                goal.adaptation_reason = adaptation_reasons.get(key, "") or ""

        # Update anim state immediately (no animation smoothing here)
        for k, v in scores.items():
            if k in self._anim_scores:
                self._anim_scores[k] = float(max(0.0, min(1.0, float(v))))

        self._has_data = True
        self.update()


def goal_bar_state(score: float, threshold: float, *, applicable: bool = True, synthesized: bool = False) -> str:
    """§GUI-T1: Reine Entscheidungslogik der Balken-Farbe (headless-testbar).

    Reihenfolge entspricht _draw_bars: n/a > synth > pass > warn > fail.
    Rückgabe: Zustands-String, der 1:1 auch als Icon-Art dient.
    """
    if not applicable:
        return "na"
    if synthesized:
        return "synth"
    if score >= threshold + 0.04:
        return "pass"
    if score >= threshold:
        return "warn"
    return "fail"


def build_radar_update_payload(result: object) -> dict[str, Any]:
    """§GUI-T1: Reine Extraktion der Radar-Daten aus einem RestorationResult
    (headless-testbar, kein Qt). Graceful degradation bei fehlenden Feldern.
    """
    # 1. Musical Goals Scores
    scores: dict[str, float] = {}
    if hasattr(result, "musical_goals") and result.musical_goals:
        mg = result.musical_goals
        scores = mg if isinstance(mg, dict) else {}

    # 2. Adaptive Thresholds
    adaptive: dict[str, float] = {}
    if hasattr(result, "adaptive_thresholds") and result.adaptive_thresholds:
        at = result.adaptive_thresholds
        if hasattr(at, "thresholds"):
            adaptive = at.thresholds
        elif isinstance(at, dict):
            adaptive = at

    # 3. Goal Applicability
    applicable: set[str] | None = None
    inapplicable_reasons: dict[str, str] = {}
    if hasattr(result, "goal_applicability") and result.goal_applicability:
        ga = result.goal_applicability
        if hasattr(ga, "applicable"):
            applicable = set(ga.applicable)
        if hasattr(ga, "reasons"):
            inapplicable_reasons = ga.reasons or {}

    # 4. Synthesierte Ziele (EraAuthentic)
    synthesized: set[str] = set()
    if hasattr(result, "genealogy") and result.genealogy:
        gen = result.genealogy
        if hasattr(gen, "operations"):
            for op in gen.operations:
                if hasattr(op, "operation_type") and "synthesize" in str(op.operation_type):
                    # Brillanz ist die typische synthesierte Achse
                    synthesized.add("brillanz")

    # 5. Adaptation Reasons
    adapt_reasons: dict[str, str] = {}
    if hasattr(result, "adaptive_thresholds") and result.adaptive_thresholds:
        at = result.adaptive_thresholds
        if hasattr(at, "adaptations"):
            adapt_reasons = at.adaptations or {}

    # Merge common MUSHRA-style fields if present (graceful support)
    possible_mushra_fields = [
        "mushra",
        "mushra_scores",
        "per_goal_mushra",
        "musical_goals_mushra",
    ]
    for f in possible_mushra_fields:
        if hasattr(result, f) and getattr(result, f):
            ms = getattr(result, f)
            if isinstance(ms, dict):
                # Convert percent-like values (>1) to [0,1]
                for k, v in ms.items():
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if val > 1.0:
                        val = max(0.0, min(100.0, val)) / 100.0
                    scores[k] = val

    return {
        "scores": scores,
        "adaptive_thresholds": adaptive if adaptive else None,
        "applicable_goals": applicable,
        "inapplicable_reasons": inapplicable_reasons,
        "synthesized_goals": synthesized if synthesized else None,
        "adaptation_reasons": adapt_reasons if adapt_reasons else None,
    }


def apply_restoration_result(
    widget: MusicalGoalsRadarWidget,
    result: object,
) -> None:
    """
    Liest ein RestorationResult-Objekt aus und aktualisiert das Radar-Widget.
    Versteht alle in der Spec definierten Felder (graceful degradation wenn fehlen).
    """
    widget.update_scores(**build_radar_update_payload(result))


def create_musical_goals_radar_scrollarea(parent: QWidget | None = None) -> QScrollArea:
    """Hilfsfunktion: Wrappt ein `MusicalGoalsRadarWidget` in eine `QScrollArea`.

    Verwenden wenn das Widget in begrenzten Fensterbereichen platziert wird und
    vertikal gescrollt werden soll (z.B. bei kleinen Dialogen). Minimal-invasiv
    — erzeugt kein neues Layout, gibt nur das ScrollArea-Objekt zurück.
    """
    area = QScrollArea(parent)
    widget = MusicalGoalsRadarWidget(parent=area)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    area.setWidget(widget)
    area.setWidgetResizable(True)
    return area
