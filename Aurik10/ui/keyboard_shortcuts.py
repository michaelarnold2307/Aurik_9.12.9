"""Keyboard-Shortcuts — Leertaste=Play/Pause, Pfeiltasten=Navigation. Spec v10.206 §8.

Integriert in ModernMainWindow.keyPressEvent().
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class KeyboardShortcuts:
    """Zentrale Tastatursteuerung für den Audio-Player.

    Shortcuts:
        Leertaste        = Play/Pause
        Pfeil Links      = -5s
        Pfeil Rechts     = +5s
        Shift+Pfeil Links  = -30s
        Shift+Pfeil Rechts = +30s
        Escape           = Stop
        E                = Experten-Modus toggeln
    """

    SEEK_SMALL_S: float = 5.0
    SEEK_LARGE_S: float = 30.0

    def __init__(self, player=None, window=None) -> None:
        self._player = player
        self._window = window
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def handle_key_press(self, key: int, modifiers: int) -> bool:
        """Verarbeitet Tastendruck. Gibt True zurück wenn behandelt."""
        if not self._enabled:
            return False

        from PyQt5.QtCore import Qt

        if key == Qt.Key_Space:
            return self._toggle_play_pause()

        if key == Qt.Key_Left:
            if modifiers & Qt.ShiftModifier:
                return self._seek_relative(-self.SEEK_LARGE_S)
            return self._seek_relative(-self.SEEK_SMALL_S)

        if key == Qt.Key_Right:
            if modifiers & Qt.ShiftModifier:
                return self._seek_relative(self.SEEK_LARGE_S)
            return self._seek_relative(self.SEEK_SMALL_S)

        if key == Qt.Key_Escape:
            return self._stop()

        if key == Qt.Key_E:
            return self._toggle_expert_mode()

        return False

    def _toggle_play_pause(self) -> bool:
        if self._player is None:
            return False
        try:
            if self._player.is_playing():
                self._player.stop()
            else:
                self._player.play()
            return True
        except Exception:
            logger.debug("keyboard_shortcuts: play/pause action failed", exc_info=True)
            return False

    def _seek_relative(self, delta_s: float) -> bool:
        if self._player is None:
            return False
        try:
            current = self._player.elapsed_seconds() or 0.0
            duration = self._player.duration_seconds() or 0.0
            new_pos = max(0.0, min(duration, current + delta_s))
            self._player.seek(new_pos / max(duration, 0.001))
            return True
        except Exception:
            logger.debug("keyboard_shortcuts: seek action failed", exc_info=True)
            return False

    def _stop(self) -> bool:
        if self._player is None:
            return False
        try:
            self._player.stop()
            return True
        except Exception:
            logger.debug("keyboard_shortcuts: stop action failed", exc_info=True)
            return False

    def _toggle_expert_mode(self) -> bool:
        try:
            from Aurik10.core.expert_mode import get_expert_mode

            em = get_expert_mode()
            em.enabled = not em.enabled
            if self._window:
                self._window._update_expert_mode_visibility()
            return True
        except Exception:
            logger.debug("keyboard_shortcuts: toggle expert mode failed", exc_info=True)
            return False
