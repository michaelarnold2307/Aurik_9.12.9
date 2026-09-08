"""
Aurik10/ui/help_system.py — Kontextsensitive Hilfe und vereinfachte Fehlermeldungen.

Bietet:
- HelpTooltip: ?-Button mit Kurzhilfe für jedes UI-Element
- ErrorSimplifier: Wandelt technische Fehler in Laien-freundliche Nachrichten um
- HelpSearchDialog: Durchsuchbare Hilfe (F1-Taste)
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets

from Aurik10.i18n import t

logger = logging.getLogger(__name__)


class HelpTooltip(QtWidgets.QPushButton):
    """Ein runder ?-Button der beim Klick eine Tooltip-ähnliche Hilfe zeigt."""

    def __init__(self, help_key: str, parent=None):
        super().__init__("?", parent)
        self._help_key = help_key
        self.setFixedSize(20, 20)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("""
            QPushButton {
                background: rgba(102,126,234,0.3);
                color: #8894A8; border-radius: 10px;
                font-size: 11pt; font-weight: bold; border: none;
            }
            QPushButton:hover {
                background: rgba(102,126,234,0.6);
                color: white;
            }
        """)
        self.setToolTip(t(help_key))
        self.clicked.connect(self._show_help)

    def _show_help(self):
        QtWidgets.QToolTip.showText(
            self.mapToGlobal(QtCore.QPoint(0, -10)),
            t(self._help_key),
            self,
            QtCore.QRect(),
            8000,
        )


class ErrorSimplifier:
    """Wandelt technische Fehler in Laien-verständliche Nachrichten.

    Usage:
        friendly = ErrorSimplifier.simplify(exception_or_string)
    """

    # Mapping von Fehlerpattern → Laien-Nachricht (i18n-Key)
    _PATTERNS: list[tuple[str, str]] = [
        ("MemoryError", "help.error.memory"),
        ("cannot allocate memory", "help.error.memory"),
        ("out of memory", "help.error.memory"),
        ("ImportError", "help.error.import"),
        ("ModuleNotFoundError", "help.error.import"),
        ("No module named", "help.error.import"),
        ("FileNotFoundError", "help.error.file_not_found"),
        ("No such file", "help.error.file_not_found"),
        ("Permission denied", "help.error.permission"),
        ("Access is denied", "help.error.permission"),
        ("sounddevice", "help.error.audio_device"),
        ("PortAudio", "help.error.audio_device"),
        ("invalid sample rate", "help.error.sample_rate"),
        ("unsupported format", "help.error.format"),
        ("corrupt", "help.error.corrupt"),
        ("truncated", "help.error.corrupt"),
        ("NaN", "help.error.nan"),
        ("inf", "help.error.nan"),
        ("cannot reshape", "help.error.internal"),
        ("shape mismatch", "help.error.internal"),
        ("ValueError", "help.error.value"),
        ("RuntimeError", "help.error.runtime"),
        ("CUDA", "help.error.gpu"),
        ("ROCm", "help.error.gpu"),
        ("GPU", "help.error.gpu"),
        ("timeout", "help.error.timeout"),
        ("timed out", "help.error.timeout"),
        ("Connection", "help.error.network"),
        ("SSL", "help.error.network"),
        ("Disk full", "help.error.disk_full"),
        ("No space left", "help.error.disk_full"),
    ]

    @classmethod
    def simplify(cls, error: Any) -> str:
        """Vereinfacht einen Fehler. Gibt Laien-Nachricht zurück.

        §GUI-T3 (2026-09-08): Für Exception-Instanzen wird der KLASSENNAME
        mit einbezogen — sonst fielen Exceptions mit leerer/plain Message
        (z. B. MemoryError("x") → str "x") durch alle Pattern und wurden
        roh angezeigt statt der freundlichen Meldung.
        """
        if isinstance(error, BaseException):
            msg = f"{type(error).__name__}: {error}"
        else:
            msg = str(error) if error else ""
        msg_lower = msg.lower()

        for pattern, key in cls._PATTERNS:
            if pattern.lower() in msg_lower:
                return t(key)

        # Fallback
        if "Error" in msg or "Exception" in msg:
            return t("help.error.generic", detail=msg.split(chr(10))[0][:100])
        return msg[:200]

    @classmethod
    def get_all_messages(cls) -> dict[str, str]:
        """Gibt alle Fehler-Nachrichten als dict zurück."""
        return {key: t(key) for _, key in cls._PATTERNS}


class HelpSearchDialog(QtWidgets.QDialog):
    """Durchsuchbare Hilfe — F1-Taste."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("help.search.title"))
        self.setMinimumSize(500, 400)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        search = QtWidgets.QLineEdit()
        search.setPlaceholderText(t("help.search.placeholder"))
        search.textChanged.connect(self._filter)
        layout.addWidget(search)

        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)

        self._populate_topics()
        close_btn = QtWidgets.QPushButton(t("action.cancel"))
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _populate_topics(self):
        topics = [
            ("help.topic.restoration_vs_studio", "help.answer.restoration_vs_studio"),
            ("help.topic.supported_formats", "help.answer.supported_formats"),
            ("help.topic.output_location", "help.answer.output_location"),
            ("help.topic.material_detection", "help.answer.material_detection"),
            ("help.topic.offline", "help.answer.offline"),
            ("help.topic.batch", "help.answer.batch"),
            ("help.topic.quality", "help.answer.quality"),
        ]
        for topic_key, answer_key in topics:
            item = QtWidgets.QListWidgetItem(t(topic_key))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, t(answer_key))
            self.list_widget.addItem(item)

        self.list_widget.itemClicked.connect(self._show_answer)

    def _show_answer(self, item):
        answer = item.data(QtCore.Qt.ItemDataRole.UserRole)
        QtWidgets.QMessageBox.information(self, item.text(), answer)

    def _filter(self, text: str):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())
