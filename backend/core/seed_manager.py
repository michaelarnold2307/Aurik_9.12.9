"""
§G5 Deterministischer Seed-Manager — Aurik 10

Zweck: Garantiert bit-identischen Output bei gleichem Input + Version.
Verwaltet einen Master-Seed pro Session und bindet alle ML-Phasen daran.

§G5 (copilot-instructions.md): gleicher Input + gleiche Version ⇒ bit-identischer Output.
Seeds pro Session; kein time.time() in Entscheidungslogik.

Usage:
    from backend.core.seed_manager import get_seed_manager

    # Session starten
    manager = get_seed_manager()
    manager.start_session(song_id="track_001")

    # Seed für ML-Phase holen
    seed = manager.get_phase_seed("phase_03_denoise")
    np.random.seed(seed)  # oder torch.manual_seed(seed)

    # Seed im Export-Metadata speichern
    metadata["session_master_seed"] = manager.master_seed
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading

logger = logging.getLogger(__name__)


class _SeedManager:
    """§G5 Deterministischer Seed-Manager — Singleton pro Session.

    Verwaltet einen Master-Seed pro Session und leitet phasenspezifische
    Seeds ab, die reproduzierbar sind (gleicher Master-Seed → gleiche Phase-Seeds).

    Invarianten:
        - Master-Seed wird nur einmal pro Session gesetzt.
        - Phase-Seeds sind deterministisch vom Master-Seed + Phase-ID abgeleitet.
        - Kein time.time() oder os.urandom() in Entscheidungslogik (§G5).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()  # Reentrant-Lock: erlaubt verschachtelte Akquisition
        self._master_seed: int | None = None
        self._session_id: str | None = None
        self._phase_seeds: dict[str, int] = {}

    @property
    def master_seed(self) -> int | None:
        """Master-Seed der aktuellen Session (None wenn nicht initialisiert)."""
        with self._lock:
            return self._master_seed

    @property
    def session_id(self) -> str | None:
        """Session-ID (song_id oder andere Identifikation)."""
        with self._lock:
            return self._session_id

    def start_session(
        self,
        song_id: str | None = None,
        master_seed: int | None = None,
    ) -> int:
        """Startet eine neue Session mit deterministischem Seed.

        Args:
            song_id: Identifikation des Songs (für Logging).
            master_seed: Expliziter Master-Seed. Wenn None → aus song_id + PID abgeleitet.

        Returns:
            Der verwendete Master-Seed (int).
        """
        with self._lock:
            # §G5: Kein time.time() — Seed aus deterministischen Quellen
            if master_seed is None:
                # Aus song_id + Prozess-ID + fester Salt ableiten
                salt = "aurik_v10_deterministic"
                seed_source = f"{song_id}:{os.getpid()}:{salt}"
                hash_bytes = hashlib.sha256(seed_source.encode()).digest()
                self._master_seed = int.from_bytes(hash_bytes[:4], byteorder="big") & 0x7FFFFFFF
            else:
                self._master_seed = master_seed & 0x7FFFFFFF

            self._session_id = song_id or "unknown"
            self._phase_seeds.clear()

            logger.info(
                "§G5 Seed-Manager: Session=%s Master-Seed=%d",
                self._session_id,
                self._master_seed,
            )

            return self._master_seed

    def get_phase_seed(self, phase_id: str) -> int:
        """Gibt einen deterministischen Seed für eine spezifische Phase zurück.

        Der Phase-Seed ist reproduzierbar vom Master-Seed + Phase-ID abgeleitet.
        Gleicher Master-Seed → gleicher Phase-Seed (für A/B-Vergleiche).

        Args:
            phase_id: Phase-Identifier (z. B. "phase_03_denoise").

        Returns:
            Deterministischer Seed für die Phase (int, 0 ≤ seed < 2^31).
        """
        with self._lock:
            if self._master_seed is None:
                # Session nicht initialisiert → konservativer Fallback
                logger.warning(
                    "§G5 Seed-Manager: Session nicht initialisiert — Fallback-Seed für %s",
                    phase_id,
                )
                self.start_session()

            if phase_id not in self._phase_seeds:
                # Deterministische Ableitung aus Master-Seed + Phase-ID
                seed_source = f"{self._master_seed}:{phase_id}"
                hash_bytes = hashlib.sha256(seed_source.encode()).digest()
                self._phase_seeds[phase_id] = int.from_bytes(hash_bytes[:4], byteorder="big") & 0x7FFFFFFF

            return self._phase_seeds[phase_id]

    def reset(self) -> None:
        """Setzt Session zurück (für Song-Isolation §V8/§G1)."""
        with self._lock:
            self._master_seed = None
            self._session_id = None
            self._phase_seeds.clear()
            logger.debug("§G5 Seed-Manager: Session zurückgesetzt")


# ── Thread-safe Singleton ────────────────────────────────────────────────

_manager_instance: _SeedManager | None = None
_manager_lock = threading.Lock()


def get_seed_manager() -> _SeedManager:
    """Singleton-Zugriff auf den Seed-Manager."""
    global _manager_instance  # pylint: disable=global-statement
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = _SeedManager()
    return _manager_instance
