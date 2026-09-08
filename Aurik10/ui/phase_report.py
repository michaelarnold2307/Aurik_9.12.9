"""§v10.14 P2: Phase-Report-Widget — zeigt nach der Restaurierung,
welche Phasen liefen, wie lange, und was sie bewirkt haben.

§GUI-T8 (2026-09-08): Von PyQt6 auf PyQt5 migriert (einzige GUI-Bindung,
kein Qt5/Qt6-Doppelladen), alle sichtbaren Texte über t() (i18n) und
Dezimalzahlen deutsch über ui_constants.de_num (§GUI-T5).
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from Aurik10.i18n import t
from Aurik10.ui.ui_constants import de_num


def _format_seconds(s: float) -> str:
    if s < 60:
        return f"{de_num(s, 1)} s"
    m, s = divmod(int(s), 60)
    return f"{m}:{s:02d}"


class PhaseReportWidget(QWidget):
    """Zeigt eine Tabelle aller ausgeführten/übersprungenen Phasen mit Metriken."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #AFC3DA; font-size: 9pt; padding: 4px 0;")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            [
                t("phase_report.col.phase"),
                t("phase_report.col.duration"),
                t("phase_report.col.quality_delta"),
                t("phase_report.col.hpi"),
                t("phase_report.col.status"),
            ]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setStyleSheet(
            "QTableWidget { background: transparent; border: none; gridline-color: #2A3040; }"
            "QTableWidget::item { color: #B8C8E0; padding: 2px 4px; }"
            "QHeaderView::section { background: #1A2030; color: #7B93B8; border: none; padding: 4px; }"
        )
        layout.addWidget(self._table)
        self.setMaximumHeight(400)

    def load(
        self,
        phases_executed: list[str],
        phases_skipped: list[str],
        phase_deltas: dict | None = None,
        total_time_s: float = 0.0,
    ) -> None:
        """Befüllt den Report mit Phasen-Daten.

        Args:
            phases_executed: Liste der ausgeführten Phasen-IDs.
            phases_skipped: Liste der übersprungenen Phasen-IDs.
            phase_deltas: Dict mit {phase_id: {quality_delta, hpi_live, duration_s, ...}}.
            total_time_s: Gesamtzeit der Restaurierung.
        """
        deltas = phase_deltas or {}
        all_phases = list(phases_executed) + list(phases_skipped)
        n_exec = len(phases_executed)
        n_skip = len(phases_skipped)

        self._summary_label.setText(
            t(
                "phase_report.summary",
                n_exec=n_exec,
                n_skip=n_skip,
                time=_format_seconds(total_time_s),
            )
        )

        self._table.setRowCount(len(all_phases))
        for row, phase_id in enumerate(all_phases):
            is_skipped = phase_id in phases_skipped
            pd = deltas.get(phase_id, {})

            # Phase name
            name_item = QTableWidgetItem(phase_id.replace("phase_", "").replace("_", " ").title())
            self._table.setItem(row, 0, name_item)

            # Duration
            dur = pd.get("duration_s", 0.0)
            dur_item = QTableWidgetItem(_format_seconds(dur) if dur > 0 else "—")
            self._table.setItem(row, 1, dur_item)

            # Quality delta
            qd = pd.get("quality_delta", 0.0)
            if is_skipped:
                qd_text = t("phase_report.skipped")
                qd_color = Qt.gray
            elif qd > 0.001:
                qd_text = f"+{de_num(qd, 3)}"
                qd_color = Qt.darkGreen
            elif qd < -0.001:
                qd_text = de_num(qd, 3)
                qd_color = Qt.darkRed
            else:
                qd_text = "±0"
                qd_color = Qt.gray
            qd_item = QTableWidgetItem(qd_text)
            qd_item.setForeground(qd_color)
            self._table.setItem(row, 2, qd_item)

            # HPI
            hpi = pd.get("hpi_live", 0.0)
            hpi_item = QTableWidgetItem(de_num(hpi, 2) if hpi > 0 else "—")
            self._table.setItem(row, 3, hpi_item)

            # Status
            status = t("phase_report.status.skipped") if is_skipped else t("phase_report.status.executed")
            status_item = QTableWidgetItem(status)
            self._table.setItem(row, 4, status_item)

        self._table.resizeRowsToContents()
