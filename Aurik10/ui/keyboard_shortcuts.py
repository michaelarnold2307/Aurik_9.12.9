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

        _action = self.key_action(key, modifiers)
        if _action is None:
            return False
        _kind, _delta = _action
        if _kind == "play_pause":
            return self._toggle_play_pause()
        if _kind == "seek":
            return self._seek_relative(_delta)
        if _kind == "stop":
            return self._stop()
        if _kind == "expert":
            return self._toggle_expert_mode()
        return False

    @staticmethod
    def key_action(key: int, modifiers: int) -> tuple[str, float] | None:
        """§GUI-T4: Reine Tasten→Aktion-Entscheidung (headless-testbar).

        Rückgabe: (action, seek_delta) mit action ∈ {play_pause, seek,
        stop, expert}; None wenn keine Taste passt.
        """
        from PyQt5.QtCore import Qt

        if key == Qt.Key_Space:
            return ("play_pause", 0.0)
        if key == Qt.Key_Left:
            _delta = -KeyboardShortcuts.SEEK_LARGE_S if modifiers & Qt.ShiftModifier else -KeyboardShortcuts.SEEK_SMALL_S
            return ("seek", _delta)
        if key == Qt.Key_Right:
            _delta = KeyboardShortcuts.SEEK_LARGE_S if modifiers & Qt.ShiftModifier else KeyboardShortcuts.SEEK_SMALL_S
            return ("seek", _delta)
        if key == Qt.Key_Escape:
            return ("stop", 0.0)
        if key == Qt.Key_E:
            return ("expert", 0.0)
        return None

    @staticmethod
    def seek_frac(current_s: float, delta_s: float, duration_s: float) -> float:
        """§GUI-T4: Reine Seek-Klemmung (headless-testbar)."""
        new_pos = max(0.0, min(duration_s, current_s + delta_s))
        return new_pos / max(duration_s, 0.001)

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
