"""
backend/core/module_communication_bus.py — Inter-module pub/sub message bus
===========================================================================

Simple publish/subscribe message bus for intra-process module communication.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any


class ModuleCommunicationBus:
    """Leichtgewichtiges publish/subscribe message bus.

    Subscribers receive (topic, message) callbacks on publish.
    All calls are synchronous (single-threaded).
    """

    def __init__(self) -> None:
        self._subscribers_by_topic: dict[str, list[Callable[..., None]]] = {}
        self._history: list[dict] = []
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable[..., None]) -> None:
        """Registriert *callback* for *topic*."""
        with self._lock:
            self._subscribers_by_topic.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: Callable[..., None]) -> None:
        """Entfernt *callback* from the subscription list for *topic*."""
        with self._lock:
            callbacks = self._subscribers_by_topic.get(topic)
            if not callbacks:
                return

            try:
                callbacks.remove(callback)
            except ValueError:
                return

            if not callbacks:
                del self._subscribers_by_topic[topic]

    def publish(self, topic: str, message: Any) -> None:
        """Publish *message* to *topic*; matching subscribers are called."""
        with self._lock:
            self._history.append({"topic": topic, "message": message})
            subscribers = list(self._subscribers_by_topic.get(topic, []))
        for cb in subscribers:
            try:
                cb(message)
            except Exception:
                with contextlib.suppress(Exception):
                    cb(topic, message)

    def get_history(self) -> list[dict]:
        """Gibt list of all published messages as {'topic': …, 'message': …} dicts zurück."""
        with self._lock:
            return list(self._history)

    def get_message_history(self) -> list[dict]:
        """Backward-compatible alias for message history."""
        return self.get_history()

    def clear(self) -> None:
        """Entfernt all subscribers and history."""
        with self._lock:
            self._subscribers_by_topic.clear()
            self._history.clear()


import threading as _threading

_module_communication_bus_instance = None
_module_communication_bus_lock = _threading.Lock()


def get_module_communication_bus() -> ModuleCommunicationBus:
    """Gibt the process-wide singleton ``ModuleCommunicationBus`` instance zurück."""
    global _module_communication_bus_instance
    if _module_communication_bus_instance is None:
        with _module_communication_bus_lock:
            if _module_communication_bus_instance is None:
                _module_communication_bus_instance = ModuleCommunicationBus()
    return _module_communication_bus_instance


logger = logging.getLogger(__name__)
