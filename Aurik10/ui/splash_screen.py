"""AURIK Splash Screen — Premium Startup-Erfahrung.

Randloser animierter Splash-Screen mit:
  - App-Icon (Vinyl-Schallplatte Vorher/Nachher-Metapher) mit Ambient-Glow
  - »AURIK«-Titel — Aurora-Gradient-Typografie (Gold → Weiß → Elektrisch-Blau)
  - Tagline + Gradient-Trennlinie
  - Animierte Equalizer-Balken: linke Hälfte Gold, rechte Hälfte Blau
    (spiegelt Icon), angetrieben durch Mehrfrequenz-Sinus-Oszillatoren
  - Pulsierender Dot-Loader und Ladestatus-Text
  - Versions- und »PROFESSIONAL«-Badge

Opazitäts-Übergänge werden von main() via setWindowOpacity()
mit kurzen processEvents()-Schleifen gesteuert — kein QPropertyAnimation nötig.
"""

import logging
import math
import re
from pathlib import Path

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer  # pylint: disable=no-name-in-module
from PyQt5.QtGui import (  # pylint: disable=no-name-in-module
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt5.QtWidgets import QApplication, QWidget  # pylint: disable=no-name-in-module

from Aurik10.i18n import t

logger = logging.getLogger(__name__)

_RES = Path(__file__).parent.parent / "resources"

try:
    from Aurik10 import __version__ as _VERSION  # type: ignore[attr-defined]
except Exception:
    _VERSION = "unknown"
    try:
        _root = Path(__file__).resolve().parents[2]
        _pyproject = _root / "pyproject.toml"
        _content = _pyproject.read_text(encoding="utf-8")
        _match = re.search(r'^version\s*=\s*"([^"]+)"', _content, re.MULTILINE)
        if _match:
            _VERSION = _match.group(1)
    except Exception:
        logger.warning("splash_screen.py::unknown Ersatzpfad", exc_info=True)


class AurikSplashScreen(QWidget):
    """Premium animierter Splash-Screen für AURIK Professional.

    Layout (760 × 420 px):
      y=  22: App-Icon 88×88 (zentriert)
      y= 172: »AURIK«-Titel-Baseline (Schrift 56 pt fett)
      y= 191: Tagline
      y= 204: Gradient-Trennlinie
      y= 330: Equalizer-Balken-Baseline  (max_h=90, Balken ab y=240)
      y= 375: Pulsierender Dot-Loader
      y= 402: Statustext (zentriert) + Versions-Badge (rechts)
    """

    _W = 760
    _H = 420
    _CORNER = 22
    _N_BARS = 22

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.FramelessWindowHint  # type: ignore[attr-defined]
            | Qt.Tool,  # type: ignore[attr-defined]  # no taskbar entry
            # WindowStaysOnTopHint intentionally omitted: would cover dialogs
            # that appear during startup (model check, warnings). raise_() is
            # called after show() to bring the splash to the front instead.
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # type: ignore[attr-defined]
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # type: ignore[attr-defined]
        self.setFixedSize(self._W, self._H)
        self._center()

        # App icon
        self._icon: QPixmap | None = None
        for _ip in (_RES / "vinyl_gold.png", _RES / "icon_premium.svg", _RES / "icon.png"):
            if _ip.exists():
                px = QPixmap(str(_ip))
                if not px.isNull():
                    self._icon = px.scaled(
                        88,
                        88,
                        Qt.KeepAspectRatio,  # type: ignore[attr-defined]
                        Qt.SmoothTransformation,  # type: ignore[attr-defined]
                    )
                    break

        self._status: str = "Initialisierung …"
        self._phase: float = 0.0  # equalizer / dot animation phase

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        """Aktualisiert die Ladestatus-Zeile und löst sofortigen Repaint aus."""
        self._status = text
        self.repaint()

    def stop_animation(self) -> None:
        """Stoppt die Equalizer-Animation (vor Fade-out aufrufen)."""
        self._timer.stop()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Inkrementiert die Animationsphase und fordert ein Repaint an."""
        self._phase += 0.09
        self.update()

    def _center(self) -> None:
        scr = QApplication.primaryScreen()
        if scr:
            g = scr.geometry()
            self.move(
                g.x() + (g.width() - self._W) // 2,
                g.y() + (g.height() - self._H) // 2,
            )

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _ev) -> None:
        """Zeichnet den Splash-Screen mit Hintergrund, Icon, Titel und Equalizer."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)  # type: ignore[attr-defined]
        p.setRenderHint(QPainter.TextAntialiasing)  # type: ignore[attr-defined]
        p.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore[attr-defined]

        w, h = self._W, self._H
        self._draw_background(p, w, h)
        self._draw_icon(p, w, h)
        self._draw_title(p, w, h)
        self._draw_tagline(p, w, h)
        self._draw_equalizer(p, w, h)
        self._draw_footer(p, w, h)
        p.end()

    # ── Background ────────────────────────────────────────────────────────────

    def _draw_background(self, p: QPainter, w: int, h: int) -> None:
        # Clip to rounded rectangle
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), self._CORNER, self._CORNER)
        p.setClipPath(clip)

        # Deep dark base
        p.fillRect(0, 0, w, h, QColor(7, 9, 22))

        # Left warm glow — gold ("before / damaged")
        g1 = QRadialGradient(w * 0.15, h * 0.42, w * 0.40)
        g1.setColorAt(0.0, QColor(188, 128, 28, 52))
        g1.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g1))

        # Right cool glow — electric blue ("after / restored")
        g2 = QRadialGradient(w * 0.85, h * 0.42, w * 0.40)
        g2.setColorAt(0.0, QColor(18, 95, 218, 55))
        g2.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g2))

        # Top purple accent
        g3 = QRadialGradient(w * 0.50, 0.0, w * 0.45)
        g3.setColorAt(0.0, QColor(102, 78, 205, 42))
        g3.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g3))

        # Bottom teal glow (equalizer area)
        g4 = QRadialGradient(w * 0.50, float(h), w * 0.48)
        g4.setColorAt(0.0, QColor(0, 188, 212, 28))
        g4.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g4))

        # Gradient border: gold TL → purple mid → blue BR
        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor(212, 162, 50, 155))
        bg.setColorAt(0.4, QColor(102, 126, 234, 110))
        bg.setColorAt(1.0, QColor(18, 145, 255, 135))
        p.setPen(QPen(QBrush(bg), 1.5))
        p.setBrush(Qt.NoBrush)  # type: ignore[attr-defined]
        p.drawRoundedRect(QRectF(0.75, 0.75, w - 1.5, h - 1.5), self._CORNER, self._CORNER)
        p.setClipping(False)

    # ── Icon ──────────────────────────────────────────────────────────────────

    def _draw_icon(self, p: QPainter, w: int, _h: int) -> None:
        if self._icon is None:
            return
        iw = self._icon.width()
        ih = self._icon.height()
        ix = (w - iw) // 2
        iy = 22

        # Ambient glow circle
        gr = QRadialGradient(w * 0.5, iy + ih * 0.5, iw * 0.75)
        gr.setColorAt(0.0, QColor(102, 130, 234, 62))
        gr.setColorAt(0.5, QColor(18, 100, 220, 24))
        gr.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(ix - 26, iy - 14, iw + 52, ih + 28, QBrush(gr))

        p.drawPixmap(ix, iy, self._icon)

    # ── Title ─────────────────────────────────────────────────────────────────

    def _draw_title(self, p: QPainter, w: int, _h: int) -> None:
        text = "AURIK"

        # Pick best available bold font
        font = QFont("Arial", 56, QFont.Bold)  # type: ignore[attr-defined]
        for family in ("Inter", "Noto Sans", "Ubuntu", "Segoe UI", "Helvetica Neue", "Arial"):
            candidate = QFont(family, 56, QFont.Bold)  # type: ignore[attr-defined]
            candidate.setLetterSpacing(QFont.AbsoluteSpacing, 10)  # type: ignore[attr-defined]
            if QFontMetrics(candidate).horizontalAdvance("M") > 25:
                font = candidate
                break
        font.setLetterSpacing(QFont.AbsoluteSpacing, 10)  # type: ignore[attr-defined]

        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        tx = (w - tw) // 2
        ty = 172  # baseline

        # Drop shadow
        sp = QPainterPath()
        sp.addText(tx + 3, ty + 3, font, text)
        p.fillPath(sp, QBrush(QColor(0, 0, 0, 110)))

        # Aurora gradient: gold → warm white → cool white → electric blue → teal
        tp = QPainterPath()
        tp.addText(tx, ty, font, text)
        br = tp.boundingRect()

        aurora = QLinearGradient(br.left(), 0.0, br.right(), 0.0)
        aurora.setColorAt(0.00, QColor(232, 178, 58))  # gold
        aurora.setColorAt(0.24, QColor(248, 222, 152))  # warm white
        aurora.setColorAt(0.50, QColor(202, 216, 255))  # cool white
        aurora.setColorAt(0.78, QColor(70, 155, 255))  # electric blue
        aurora.setColorAt(1.00, QColor(22, 208, 222))  # teal

        p.fillPath(tp, QBrush(aurora))

        # Top shimmer overlay
        sh = QLinearGradient(0.0, br.top(), 0.0, br.top() + br.height() * 0.38)
        sh.setColorAt(0.0, QColor(255, 255, 255, 58))
        sh.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.save()
        p.setClipPath(tp)
        p.fillRect(br, QBrush(sh))
        p.restore()

    # ── Tagline ───────────────────────────────────────────────────────────────

    def _draw_tagline(self, p: QPainter, w: int, _h: int) -> None:
        text = t("splash.tagline")

        font = QFont("Arial", 9)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 3.2)  # type: ignore[attr-defined]
        p.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        tx = (w - tw) // 2
        ty = 191

        p.setPen(QPen(QColor(162, 182, 228, 192)))
        p.drawText(tx, ty, text)

        # Developer credit — below tagline, above separator
        dev_text = t("splash.credit")
        font_dev = QFont("Arial", 8)
        font_dev.setItalic(True)
        p.setFont(font_dev)
        fm_dev = QFontMetrics(font_dev)
        dw = fm_dev.horizontalAdvance(dev_text)
        p.setPen(QPen(QColor(180, 170, 140, 150)))
        p.drawText((w - dw) // 2, ty + 16, dev_text)

        # Gradient separator line
        line_y = ty + 22
        lg = QLinearGradient(w * 0.22, 0.0, w * 0.78, 0.0)
        lg.setColorAt(0.00, QColor(218, 162, 50, 0))
        lg.setColorAt(0.18, QColor(218, 162, 50, 118))
        lg.setColorAt(0.50, QColor(102, 126, 234, 128))
        lg.setColorAt(0.82, QColor(18, 145, 255, 108))
        lg.setColorAt(1.00, QColor(18, 145, 255, 0))
        p.setPen(QPen(QBrush(lg), 1.0))
        p.drawLine(int(w * 0.22), line_y, int(w * 0.78), line_y)

    # ── Equalizer bars ────────────────────────────────────────────────────────

    def _draw_equalizer(self, p: QPainter, w: int, _h: int) -> None:
        """Animierte Equalizer-Balken, angetrieben durch Mehrfrequenz-Sinus-Oszillatoren.

        Linke Hälfte: Gold-Töne (»Vorher/Vintage-Quelle«).
        Rechte Hälfte: Blau-Töne (»Nachher/Restauriert«).
        Balken verwenden zusammengesetzte Sinuswellen für organische Bewegung.
        """
        n = self._N_BARS
        bar_w = 10
        gap = 7
        total_w = n * bar_w + (n - 1) * gap
        x0 = (w - total_w) // 2
        base_y = 330
        max_h = 90

        # Faint baseline
        bl_g = QLinearGradient(float(x0), 0.0, float(x0 + total_w), 0.0)
        bl_g.setColorAt(0.0, QColor(202, 152, 38, 42))
        bl_g.setColorAt(0.5, QColor(102, 126, 234, 52))
        bl_g.setColorAt(1.0, QColor(18, 140, 255, 40))
        p.setPen(QPen(QBrush(bl_g), 1))
        p.drawLine(x0, base_y + 1, x0 + total_w, base_y + 1)
        p.setPen(Qt.NoPen)  # type: ignore[attr-defined]

        for i in range(n):
            # Four-harmonic sine oscillator for organic bar movement
            ph = self._phase + i * 0.43
            hf = (
                0.38
                + 0.26 * math.sin(ph)
                + 0.19 * math.sin(ph * 1.71 + 1.1)
                + 0.11 * math.sin(ph * 3.13 + 0.6)
                + 0.06 * math.sin(ph * 5.29 + 2.1)
            )
            hf = max(0.07, min(1.0, hf))
            bh = int(max_h * hf)
            x = x0 + i * (bar_w + gap)
            y = base_y - bh

            # Color split: gold (left = before) / blue (right = after)
            frac = i / max(n - 1, 1)
            if frac < 0.5:
                c_bot = QColor(222, 145, 26, 218)
                c_top = QColor(255, 205, 72, 238)
            else:
                c_bot = QColor(14, 115, 222, 215)
                c_top = QColor(52, 200, 255, 238)

            gr = QLinearGradient(float(x), float(base_y), float(x), float(base_y - max_h))
            gr.setColorAt(0.0, c_bot)
            gr.setColorAt(1.0, c_top)

            bp = QPainterPath()
            bp.addRoundedRect(QRectF(x, y, bar_w, bh), 2.5, 2.5)
            p.fillPath(bp, QBrush(gr))

            # Subtle reflection below baseline
            ref_h = min(bh // 5, 11)
            if ref_h > 1:
                rg = QLinearGradient(
                    float(x),
                    float(base_y + 1),
                    float(x),
                    float(base_y + 1 + ref_h),
                )
                rg.setColorAt(0.0, QColor(c_bot.red(), c_bot.green(), c_bot.blue(), 48))
                rg.setColorAt(1.0, QColor(0, 0, 0, 0))
                rp = QPainterPath()
                rp.addRect(QRectF(x, base_y + 2, bar_w, ref_h))
                p.fillPath(rp, QBrush(rg))

    # ── Footer ────────────────────────────────────────────────────────────────

    def _draw_footer(self, p: QPainter, w: int, h: int) -> None:
        # Pulsing dot loader — 5 beats, gold left, blue right
        pulse_y = float(h - 42)
        n_dots = 5
        dot_r = 3.2
        spacing = 12.0
        total_d = (n_dots - 1) * spacing
        dx0 = (w - total_d) / 2.0

        p.setPen(Qt.NoPen)  # type: ignore[attr-defined]
        for i in range(n_dots):
            beat = math.sin(self._phase * 1.8 + i * 0.9) * 0.5 + 0.5
            alpha = int(75 + 158 * beat)
            frac = i / max(n_dots - 1, 1)
            c = QColor(222, 162, 44, alpha) if frac < 0.5 else QColor(38, 158, 242, alpha)
            p.setBrush(QBrush(c))
            p.drawEllipse(QPointF(dx0 + i * spacing, pulse_y), dot_r, dot_r)

        # Status text — centered
        font_s = QFont("Arial", 9)
        p.setFont(font_s)
        fm_s = QFontMetrics(font_s)
        tw = fm_s.horizontalAdvance(self._status)
        p.setPen(QPen(QColor(115, 152, 215, 172)))
        p.drawText((w - tw) // 2, h - 18, self._status)

        # Version badge — bottom right
        font_v = QFont("Arial", 8)
        p.setFont(font_v)
        fm_v = QFontMetrics(font_v)
        vt = f"v{_VERSION}"
        p.setPen(QPen(QColor(82, 108, 158, 138)))
        p.drawText(w - fm_v.horizontalAdvance(vt) - 14, h - 18, vt)
