"""backend/core/plugin_lifecycle_manager.py — Automatisches Plugin-RAM-Management.

Zentraler Manager, der:
  1. Alle registrierten unload_*()-Funktionen kennt (Singleton-Muster).
  2. Bei RAM-Druck (> 85 % oder psutil-Messung) automatisch selten genutzte
     Plugins entlädt (LRU-Strategie — zuletzt weniger genutzt zuerst).
  3. Wird vom AurikDenker nach jeder Datei aufgerufen (Batch-Cleanup).
  4. Kann jederzeit manuell via force_evict_all() geleert werden.

Kein ML-Modell wird dabei ohne Zustimmung der Pipeline beendet —
nur Modelle AUSSERHALB der laufenden Phase werden evicted.

Thread-sicher: alle öffentlichen Methoden sind Lock-geschützt.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from importlib import import_module

logger = logging.getLogger(__name__)

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
_RAM_EVICT_THRESHOLD_PCT: float = 75.0  # RAM% ab der Eviction beginnt (gesenkt: früher reagieren)
_RAM_TARGET_PCT: float = 65.0  # RAM% auf die wir evicten wollen
_MIN_FREE_MB_HARD: float = 3000.0  # immer mind. 3 GB frei halten
_PIPELINE_EMERGENCY_PCT: float = 70.0  # RAM% ab der auch WÄHREND Pipeline evicted wird (gesenkt von 78%)
_SWAP_EVICT_THRESHOLD_PCT: float = 80.0  # Swap% ab der Eviction erzwungen wird (unabhängig von RAM-%).
_SWAP_EVICT_FORCE_PCT: float = 95.0  # Harte Swap-Notlage: immer evicten (Crash-Prävention)
_SWAP_RELAX_RAM_PCT: float = 70.0  # Unterhalb davon gilt hoher Swap nicht automatisch als akut
_SWAP_RELAX_FREE_MB: float = 10_000.0  # Mit viel freiem RAM sind alte Swap-Reste oft unkritisch (10 GB Schwelle — 11–12 GB frei ist kein Notfall)
# Rationale: Crash 2026-04-14 — swap=99 %, avail=18.79 GB → OOM-Killer wegen
# Apollo-TorchScript-Allokation. RAM-only-Guards erkannten die Gefahr nicht.
_MONITOR_JOIN_TIMEOUT_S: float = 1.0  # Shutdown darf Tests/App-Ende nicht unbounded blockieren
# Begründung Absenkung: 82% war zu konservativ — FlashSR (7 GB) konnte nicht geladen werden weil
# Eviction erst bei 82% erlaubt war, RAM aber schon bei 78% knapp wurde. Modelle die gerade
# NICHT in Inferenz sind (active=False) können sicher entladen werden auch bei 78%.

# ---------------------------------------------------------------------------
# §2.37 Phase-zu-Modell-Mapping: Welche ML-Modelle braucht welche Phase?
# Nur Phasen mit ML-Modellen sind hier gelistet. DSP-only-Phasen fehlen bewusst.
# Modellnamen müssen EXAKT mit dem Namen in ml_memory_budget.try_allocate() übereinstimmen.
# ---------------------------------------------------------------------------
_PHASE_REQUIRED_MODELS: dict[str, frozenset[str]] = {
    "phase_01_click_removal": frozenset({"DeepFilterNetV3"}),
    "phase_02_hum_removal": frozenset({"DeepFilterNetV3"}),
    "phase_03_denoise": frozenset(
        {"SGMSE+", "ResembleEnhance", "DeepFilterNetV3", "MIIPHER_DiT"}
    ),  # §v10.14: MIIPHER-DiT für Gesang
    "phase_06_frequency_restoration": frozenset({"FlashSR", "AudioSR"}),
    "phase_09_crackle_removal": frozenset({"BanquetVinyl"}),
    "phase_12_wow_flutter_fix": frozenset({"FCPE", "RMVPE", "CREPE"}),
    "phase_18_noise_gate": frozenset({"SileroVAD"}),
    "phase_20_reverb_reduction": frozenset(
        {"SGMSE+", "ResembleEnhance"}
    ),  # §4.6c: HybridDereverb uses SGMSE+ primary + ResembleEnhance fallback
    "phase_23_spectral_repair": frozenset({"Apollo", "FlashSR", "AudioSR"}),
    "phase_24_dropout_repair": frozenset(
        {"FlashSR", "GACELA", "AudioLDM2"}
    ),  # §4.6c: cascade DSP→GACELA→FlashSR→AudioLDM2
    "phase_29_tape_hiss_reduction": frozenset({"DeepFilterNetV3"}),
    "phase_31_speed_pitch_correction": frozenset(
        {"BasicPitch", "FCPE", "RMVPE", "CREPE"}
    ),  # §4.6c: HybridSpeedPitch loads FCPE→RMVPE→CREPE cascade
    "phase_42_vocal_enhancement": frozenset(
        {"MelBandRoformer", "MDX23C_vocals", "MDX23C_inst", "DemucsV4"}
    ),  # §4.6c: DemucsV4 is stem-sep fallback
    "phase_43_ml_deesser": frozenset({"MP-SENet"}),
    "phase_49_advanced_dereverb": frozenset({"SGMSE+"}),
    "phase_55_diffusion_inpainting": frozenset(
        {"CQTdiffPlus", "FlowMatching", "DiffWave", "ConsistencyInpaint", "DACInpaint"}
    ),
    "phase_56_spectral_band_gap_repair": frozenset({"FCPE", "RMVPE", "CREPE"}),  # §4.6c: f0-cascade FCPE→RMVPE→CREPE
}


def _release_ml_memory_budget(plugin_name: str) -> None:
    """Release ML memory-budget accounting for an evicted plugin if available."""
    try:
        ml_memory_budget = import_module("backend.core.ml_memory_budget")
        release_fn = getattr(ml_memory_budget, "release", None)
        if callable(release_fn):
            release_fn(plugin_name)
    except ImportError:
        logger.debug("ML-Speicherbudget nicht verfügbar — Freigabe übersprungen")


# ---------------------------------------------------------------------------
# Registry-Eintrag
# ---------------------------------------------------------------------------


class _PluginEntry:
    __slots__ = ("active", "last_used_ts", "name", "size_gb", "unload_fn")

    def __init__(
        self,
        name: str,
        size_gb: float,
        unload_fn: Callable[[], None],
    ) -> None:
        self.name = name
        self.size_gb = size_gb
        self.unload_fn = unload_fn
        self.last_used_ts: float = time.monotonic()
        self.active: bool = False  # True = darf NICHT evicted werden (Phase läuft)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_INSTANCE_HOLDER: dict[str, PluginLifecycleManager | None] = {"manager": None}
_singleton_lock = threading.Lock()


def get_plugin_lifecycle_manager() -> PluginLifecycleManager:
    """Gibt den PluginLifecycleManager-Singleton zurück (Double-Checked Locking)."""
    if _INSTANCE_HOLDER["manager"] is None:
        with _singleton_lock:
            if _INSTANCE_HOLDER["manager"] is None:
                _INSTANCE_HOLDER["manager"] = PluginLifecycleManager()
    manager = _INSTANCE_HOLDER["manager"]
    assert manager is not None
    return manager


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class PluginLifecycleManager:
    """LRU-basierter Plugin-Memory-Manager mit automatischer Eviction.

    Registrierte Plugins werden nach LRU (Least-Recently-Used) entladen,
    sobald der Systemspeicher unter den Schwellwert fällt.

    Lock-order: Priority 2 (PLM) — see §3.9.8.
    Never acquire PLM._lock while already holding MLMemoryBudget._lock (Priority 1).
    Always acquire PLM._lock BEFORE AdaptiveResourceManager.lock (Priority 3).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _PluginEntry] = {}
        self._auto_evict_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pipeline_active: int = 0  # Refcount: >0 suppresses auto-eviction during pipeline
        # §Perf Look-Ahead-Eviction: Modelle, die eine baldige Phase erneut braucht.
        # Vom Orchestrator vor jeder Phase via evict_for_phase_window() gesetzt; von ALLEN
        # evict_for_phase()-Call-Sites (auch phasen-intern) respektiert → kein Reload-Thrashing.
        # Leer = Originalverhalten (nur eigenes Phasenmodell geschützt). Rein RAM-Scheduling,
        # bit-identisches Audio.
        self._lookahead_models: frozenset[str] = frozenset()
        self._last_swap_warn_ts: float = (
            0.0  # Cooldown für Swap-Druck-Warnungen (min. 60 s zwischen identischen Meldungen)
        )
        self._start_auto_evict_monitor()
        logger.info("PluginLifecycleManager: initialisiert (RAM eviction Schwelle %.0f %%)", _RAM_EVICT_THRESHOLD_PCT)

    # ------------------------------------------------------------------
    # Registrierung
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        size_gb: float,
        unload_fn: Callable[[], None],
    ) -> None:
        """Registriert ein Plugin mit seiner Unload-Funktion.

        Idempotent — mehrfacher Aufruf mit demselben `name` aktualisiert
        nur `last_used_ts` und `size_gb`.

        Args:
            name:      Eindeutiger Plugin-Name (z. B. 'MelBandRoformer').
            size_gb:   Geschätzter RAM-Verbrauch des Modells in GB.
            unload_fn: Callable das das Modell aus dem RAM entlädt
                       (ruft typisch `global _instance; _instance = None; gc.collect()` auf).
        """
        with self._lock:
            if name in self._entries:
                self._entries[name].last_used_ts = time.monotonic()
                self._entries[name].size_gb = size_gb
                return
            self._entries[name] = _PluginEntry(name, size_gb, unload_fn)
            logger.debug("PLM: '%s' registriert (%.2f GB).", name, size_gb)

    def touch(self, name: str) -> None:
        """Aktualisiert 'last_used_ts' für `name` (vor jeder Plugin-Nutzung aufrufen)."""
        with self._lock:
            if name in self._entries:
                self._entries[name].last_used_ts = time.monotonic()

    def set_active(self, name: str, active: bool) -> None:
        """Markiert ein Plugin als aktiv (Eviction gesperrt) oder inaktiv."""
        with self._lock:
            if name in self._entries:
                self._entries[name].active = active

    def enter_pipeline(self) -> int:
        """Increment pipeline-active refcount in a thread-safe way."""
        with self._lock:
            self._pipeline_active += 1
            return self._pipeline_active

    def leave_pipeline(self) -> int:
        """Decrement pipeline-active refcount in a thread-safe way."""
        with self._lock:
            self._pipeline_active = max(0, self._pipeline_active - 1)
            # §Perf: Look-Ahead-Fenster zurücksetzen, wenn keine Pipeline mehr läuft —
            # verhindert, dass ein veraltetes Fenster spätere Standalone-Eviction über-schützt.
            if self._pipeline_active == 0:
                self._lookahead_models = frozenset()
            return self._pipeline_active

    def pipeline_active_count(self) -> int:
        """Gibt current pipeline-active refcount thread-safely zurück."""
        with self._lock:
            return self._pipeline_active

    def unregister(self, name: str) -> None:
        """Entfernt ein Plugin aus der Registry (nach manuell erfolgtem Unload)."""
        with self._lock:
            self._entries.pop(name, None)

    # ------------------------------------------------------------------
    # Manuelle Eviction
    # ------------------------------------------------------------------

    def evict_if_needed(self, required_mb: float = 0.0) -> int:
        """Entlädt inaktive Plugins falls RAM-Druck besteht.

        Args:
            required_mb: Mindest-RAM der sofort benötigt wird [MB].
                         Falls 0, wird nur auf Basis des RAM-% entschieden.

        Returns:
            Anzahl der entladenen Plugins.
        """
        ram_pct = self._ram_percent()
        free_mb = self._free_mb()
        swap_pct = self._swap_percent()
        swap_emergency = self._swap_pressure_requires_evict(ram_pct=ram_pct, free_mb=free_mb, swap_pct=swap_pct)
        pipeline_emergency = (
            self._pipeline_active > 0 and ram_pct >= _PIPELINE_EMERGENCY_PCT and free_mb < _SWAP_RELAX_FREE_MB
        )
        # §Safety: Während Pipeline-Ausführung normalerweise keine automatische
        # Eviction — ONNX-Session-Destruktoren können mit laufender Inferenz
        # kollidieren. ABER: bei kritischem RAM-Druck (>= 82%) MUSS trotzdem
        # evicted werden, sonst killt systemd-oomd den gesamten Prozess.
        if self._pipeline_active > 0 and required_mb <= 0:
            if not pipeline_emergency and free_mb >= _MIN_FREE_MB_HARD and not swap_emergency:
                return 0
            _now = time.monotonic()
            # §v10.306: Adaptiver Cooldown. Wenn letzter Evict nichts brachte,
            # 5 Minuten warten statt jede Minute warnen.
            _cooldown = 300.0 if getattr(self, "_last_evict_was_empty", False) else 60.0
            if _now - self._last_swap_warn_ts >= _cooldown:
                _log_fn = logger.warning if (free_mb < _MIN_FREE_MB_HARD or swap_emergency) else logger.info
                _log_fn(
                    "PLM: Pipeline aktiv, Speicherpflege (RAM=%.0f %%, frei=%.0f MB, swap=%.0f %%) "
                    "— evicte inaktive Plugins",
                    ram_pct,
                    free_mb,
                    swap_pct,
                )
                self._last_swap_warn_ts = _now
        # required_mb kommt bereits MIT Margin aus ml_memory_budget._preflight_system_memory.
        # Keine doppelte Margin (war 1.25×) — direkt prüfen ob genug frei ist.
        needs_evict = (
            ram_pct > _RAM_EVICT_THRESHOLD_PCT
            or free_mb < _MIN_FREE_MB_HARD
            or (required_mb > 0 and free_mb < required_mb)
            or pipeline_emergency
            or swap_emergency
        )
        if swap_emergency and not (ram_pct > _RAM_EVICT_THRESHOLD_PCT or free_mb < _MIN_FREE_MB_HARD):
            _now_s = time.monotonic()
            _cooldown = 300.0 if getattr(self, "_last_evict_was_empty", False) else 60.0
            if _now_s - self._last_swap_warn_ts >= _cooldown:
                logger.warning(
                    "PLM: Swap-Druck kritisch (%.0f %%) — erzwinge Eviction inaktiver Plugins (RAM=%.0f %%, frei=%.0f MB)",
                    swap_pct,
                    ram_pct,
                    free_mb,
                )
                self._last_swap_warn_ts = _now_s
        if not needs_evict:
            return 0
        _evicted = self._do_evict(target_pct=_RAM_TARGET_PCT, required_mb=required_mb)
        # §v10.306: Tracke ob letzte Eviction leer war → Cooldown-Verlängerung
        self._last_evict_was_empty = _evicted == 0
        return _evicted

    def force_evict_all(self) -> int:
        """Entlädt ALLE registrierten, inaktiven Plugins sofort.

        Wird vom AurikDenker nach abgeschlossener Batch-Datei aufgerufen.
        """
        with self._lock:
            # §Perf: Look-Ahead-Fenster der abgeschlossenen Datei verwerfen.
            self._lookahead_models = frozenset()
        return self._do_evict(target_pct=0.0, force_all=True)

    def evict_for_phase(self, phase_id: str) -> int:
        """Entlädt alle ML-Modelle die für die kommende Phase NICHT benötigt werden.

        §2.37 Automatische RAM-Verwaltung: Vor jeder Phase werden nur die
        tatsächlich benötigten Modelle im RAM behalten. Alle anderen —
        auch während aktiver Pipeline — werden entladen.

        Sicher: Nur inaktive (nicht gerade in Inferenz befindliche) Modelle
        werden entladen. Aktive Modelle (entry.active=True) bleiben geschützt.

        Args:
            phase_id: Die nächste auszuführende Phase (z. B. "phase_06_frequency_restoration").

        Returns:
            Anzahl der entladenen Plugins.
        """
        needed = _PHASE_REQUIRED_MODELS.get(phase_id, frozenset())

        with self._lock:
            # §Perf Look-Ahead: zusätzlich zur eigenen Phase auch Modelle schützen, die eine
            # baldige Phase im aktuellen Pipeline-Fenster erneut braucht (vom Orchestrator via
            # evict_for_phase_window gesetzt). Leeres Fenster → exakt Originalverhalten.
            needed = needed | self._lookahead_models
            candidates = [e for e in self._entries.values() if not e.active and e.name not in needed]
            # LRU: älteste zuerst
            candidates.sort(key=lambda e: e.last_used_ts)

        if not candidates:
            return 0

        evicted = 0
        for entry in candidates:
            try:
                logger.info(
                    "PLM: Entlade '%s' (%.2f GB) vor %s — nicht benötigt",
                    entry.name,
                    entry.size_gb,
                    phase_id,
                )
                entry.unload_fn()
                _release_ml_memory_budget(entry.name)
                with self._lock:
                    self._entries.pop(entry.name, None)
                evicted += 1
            except Exception as exc:
                logger.warning("PLM: Fehler beim Entladen von '%s': %s", entry.name, exc)

        if evicted > 0:
            # GC und malloc_trim EINMAL nach der Schleife statt pro Plugin.
            # Jedes gc.collect() hält die Python-GIL für 50–200 ms →
            # pro-Plugin-Aufruf blockierte den Qt-Hauptthread kumulativ.
            gc.collect()
            time.sleep(0)  # GIL explizit freigeben → Qt-Event-Loop kann X11-Pings beantworten
            # NOTE: malloc_trim(0) entfernt — kann SIGABRT verursachen wenn
            # sbrk() aus diesem Thread gleichzeitig mit numpy-Allokationen
            # im Restaurierungs-Thread läuft (gleiche Root-Cause wie _do_evict).
            logger.info(
                "PLM: %d Plugin(s) entladen vor %s — RAM nach GC: %.0f %% (%.0f MB frei)",
                evicted,
                phase_id,
                self._ram_percent(),
                self._free_mb(),
            )
        return evicted

    def evict_for_phase_window(self, upcoming_phase_ids: list[str] | tuple[str, ...]) -> int:
        """§Perf Look-Ahead-Eviction: schützt Modelle, die eine *baldige* Phase erneut braucht.

        Identisch zu ``evict_for_phase()``, ABER die Menge der geschützten Modelle ist
        die Vereinigung der von ALLEN anstehenden Phasen (``upcoming_phase_ids``, inkl.
        der aktuellen an Position 0) benötigten Modelle. Dadurch wird Reload-Thrashing
        eliminiert: Ein 6-GB-Modell wie ``AudioSR`` (phase_06 → phase_23 → phase_24) wird
        NICHT zwischen seinen Nutzungen entladen und teuer von Disk neu geladen, nur weil
        die unmittelbar nächste Phase es nicht braucht.

        WISSENSCHAFTLICHE INVARIANTE (strikt): Diese Methode ändert ausschließlich, WANN
        ML-Modelle im RAM liegen. Sie berührt NIEMALS das restaurierte Audio, die
        Carrier-Chain-Reihenfolge (§2.46), die Phasenabfolge oder irgendeine Messung/Gate
        (PMGG/CIG/AFG/HPI/VQI). Das Ergebnis ist bit-identisch zur aktuellen Pipeline —
        nur ohne redundante Modell-Reloads. Sie erzeugt KEIN OOM-Risiko: Bei echtem
        Speicherdruck greift weiterhin unabhängig ``evict_if_needed()`` (LRU + Swap-Guard)
        und entlädt auch fenster-geschützte (inaktive) Modelle. Look-Ahead unterdrückt nur
        die *proaktive* Entladung — niemals die druckgetriebene.

        Args:
            upcoming_phase_ids: Verbleibende Phasen ab der aktuellen (Position 0 = aktuelle
                                Phase). Üblicherweise ``selected_phases[len(executed):]``.

        Returns:
            Anzahl der entladenen Plugins.
        """
        if not upcoming_phase_ids:
            return 0

        needed: frozenset[str] = frozenset().union(
            *(_PHASE_REQUIRED_MODELS.get(pid, frozenset()) for pid in upcoming_phase_ids)
        )
        _current_phase = upcoming_phase_ids[0]

        with self._lock:
            # Fenster persistieren, damit phasen-interne evict_for_phase()-Calls (15 Phasen)
            # dieselbe Look-Ahead-Protektion erben und das Modell nicht doch noch entladen.
            self._lookahead_models = needed
            candidates = [e for e in self._entries.values() if not e.active and e.name not in needed]
            # LRU: älteste zuerst
            candidates.sort(key=lambda e: e.last_used_ts)

        if not candidates:
            return 0

        evicted = 0
        for entry in candidates:
            try:
                logger.info(
                    "PLM: Entlade '%s' (%.2f GB) vor %s — von keiner anstehenden Verarbeitungsschritt benötigt (Look-Ahead)",
                    entry.name,
                    entry.size_gb,
                    _current_phase,
                )
                entry.unload_fn()
                _release_ml_memory_budget(entry.name)
                with self._lock:
                    self._entries.pop(entry.name, None)
                evicted += 1
            except Exception as exc:
                logger.warning("PLM: Fehler beim Entladen von '%s': %s", entry.name, exc)

        if evicted > 0:
            gc.collect()
            time.sleep(0)  # GIL explizit freigeben → Qt-Event-Loop bleibt responsiv
            logger.info(
                "PLM: %d Plugin(s) Look-Ahead-entladen vor %s — RAM nach GC: %.0f %% (%.0f MB frei)",
                evicted,
                _current_phase,
                self._ram_percent(),
                self._free_mb(),
            )
        return evicted

    def _do_evict(
        self,
        target_pct: float = _RAM_TARGET_PCT,
        required_mb: float = 0.0,
        force_all: bool = False,
    ) -> int:
        """Führt die eigentliche Eviction durch (LRU-Reihenfolge)."""
        required_free_mb = max(_MIN_FREE_MB_HARD, float(required_mb))
        with self._lock:
            # LRU-Sortierung: ältester Zugriff zuerst; aktive ausgenommen
            candidates: list[_PluginEntry] = sorted(
                [e for e in self._entries.values() if not e.active],
                key=lambda e: e.last_used_ts,
            )

        evicted = 0
        for entry in candidates:
            if not force_all:
                ram_pct = self._ram_percent()
                free_mb = self._free_mb()
                swap_pct = self._swap_percent()
                swap_emergency = self._swap_pressure_requires_evict(ram_pct=ram_pct, free_mb=free_mb, swap_pct=swap_pct)
                if ram_pct <= target_pct and free_mb >= required_free_mb and not swap_emergency:
                    break
            try:
                logger.info(
                    "PLM: Entlade '%s' (%.2f GB, last_used=%.0fs ago) …",
                    entry.name,
                    entry.size_gb,
                    time.monotonic() - entry.last_used_ts,
                )
                entry.unload_fn()
                _release_ml_memory_budget(entry.name)
                with self._lock:
                    self._entries.pop(entry.name, None)
                evicted += 1
                logger.info("PLM: '%s' erfolgreich entladen.", entry.name)
            except Exception as exc:
                logger.warning("PLM: Fehler beim Entladen von '%s': %s", entry.name, exc)

        if evicted > 0:
            # GC und malloc_trim EINMAL nach der Schleife statt pro Plugin.
            # Jedes gc.collect() hält die Python-GIL für 50–200 ms →
            # pro-Plugin-Aufruf blockierte den Qt-Hauptthread kumulativ.
            gc.collect()
            time.sleep(0)  # GIL explizit freigeben → Qt-Event-Loop kann X11-Pings beantworten
            # NOTE: malloc_trim(0) wurde entfernt. Es kann SIGABRT verursachen wenn
            # sbrk() im PLM-Thread gleichzeitig mit numpy-Allokationen (z.B.
            # sliding_window_view.copy()) im Restaurierungs-Thread läuft.
            # gc.collect() ist für die RAM-Freigabe ausreichend.
            logger.info("PLM: %d Plugin(s) entladen — RAM nach GC: %.0f %%", evicted, self._ram_percent())
        return evicted

    # ------------------------------------------------------------------
    # Automatisches Monitoring
    # ------------------------------------------------------------------

    def _start_auto_evict_monitor(self) -> None:
        """Startet einen Daemon-Thread, der alle 10 s den RAM prüft."""
        self._auto_evict_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PLM_auto_evict",
        )
        self._auto_evict_thread.start()

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(timeout=10.0):
            try:
                self.evict_if_needed()
            except Exception as exc:
                logger.debug("PLM Monitor-Fehler: %s", exc)

    def shutdown(self) -> None:
        """Stoppt den Monitor-Thread best-effort ohne unbounded block."""
        self._stop_event.set()
        thread = self._auto_evict_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_MONITOR_JOIN_TIMEOUT_S)
        self._auto_evict_thread = None

    # ------------------------------------------------------------------
    # RAM-Messung
    # ------------------------------------------------------------------

    @staticmethod
    def _ram_percent() -> float:
        if _psutil is not None:
            return float(_psutil.virtual_memory().percent)
        return 0.0  # ohne psutil kein automatisches Evict

    @staticmethod
    def _free_mb() -> float:
        if _psutil is not None:
            return float(_psutil.virtual_memory().available / (1024 * 1024))
        return float("inf")

    @staticmethod
    def _swap_percent() -> float:
        """Gibt swap usage in percent (0–100). 0 if psutil unavailable or no swap zurück."""
        if _psutil is not None:
            try:
                return float(_psutil.swap_memory().percent)
            except Exception as e:
                logger.warning("plugin_lifecycle_manager.py::_swap_percent Ersatzpfad: %s", e)
                return 0.0
        return 0.0

    @staticmethod
    def _swap_pressure_requires_evict(*, ram_pct: float, free_mb: float, swap_pct: float) -> bool:
        """Bewertet, ob Swap-Druck akut genug fuer erzwungene Eviction ist.

        Hohe Swap-Belegung allein kann ein historischer Restzustand sein.
        Eviction wird daher nur bei echter Notlage erzwungen:
        - swap >= _SWAP_EVICT_FORCE_PCT (harte Notlage), oder
        - swap > _SWAP_EVICT_THRESHOLD_PCT UND gleichzeitig hoher RAM-Druck
          oder knappes freies RAM.
        """
        if swap_pct >= _SWAP_EVICT_FORCE_PCT:
            return True
        if swap_pct <= _SWAP_EVICT_THRESHOLD_PCT:
            return False
        return bool(ram_pct >= _SWAP_RELAX_RAM_PCT or free_mb < _SWAP_RELAX_FREE_MB)

    # ------------------------------------------------------------------
    # Status (Logging/Debug)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Gibt a lightweight snapshot of current RAM pressure and registered plugins zurück."""
        with self._lock:
            return {
                "ram_pct": round(self._ram_percent(), 1),
                "free_mb": round(self._free_mb(), 0),
                "registered_plugins": [
                    {
                        "name": e.name,
                        "size_gb": e.size_gb,
                        "active": e.active,
                        "idle_s": round(time.monotonic() - e.last_used_ts, 0),
                    }
                    for e in self._entries.values()
                ],
            }


# ---------------------------------------------------------------------------
# Convenience-Funktionen (Modul-Level)
# ---------------------------------------------------------------------------


def register_plugin(name: str, size_gb: float, unload_fn: Callable[[], None]) -> None:
    """Registriert ein Plugin beim globalen Lifecycle-Manager."""
    get_plugin_lifecycle_manager().register(name, size_gb, unload_fn)


def touch_plugin(name: str) -> None:
    """Aktualisiert 'last_used_ts' für ein Plugin (vor Nutzung)."""
    get_plugin_lifecycle_manager().touch(name)


def evict_stale_plugins(required_mb: float = 0.0) -> int:
    """Entlädt inaktive Plugins falls RAM-Druck besteht. Gibt Anzahl zurück."""
    return get_plugin_lifecycle_manager().evict_if_needed(required_mb)


def evict_for_phase(phase_id: str) -> int:
    """Entlädt alle ML-Modelle die für ``phase_id`` NICHT benötigt werden.

    §2.37: Vor jeder Phase nur benötigte Modelle im RAM behalten.
    """
    return get_plugin_lifecycle_manager().evict_for_phase(phase_id)


def evict_for_phase_window(upcoming_phase_ids: list[str] | tuple[str, ...]) -> int:
    """§Perf Look-Ahead-Eviction: behält Modelle, die eine baldige Phase erneut braucht.

    Schützt die Vereinigung aller von ``upcoming_phase_ids`` benötigten Modelle vor
    proaktiver Entladung → eliminiert Reload-Thrashing (z. B. FlashSR phase_06 → phase_23).
    Bit-identisch zum Audio-Ergebnis; berührt nur RAM-Scheduling. Druckgetriebene
    Eviction (``evict_if_needed``) bleibt unabhängig aktiv → kein OOM-Risiko.
    """
    return get_plugin_lifecycle_manager().evict_for_phase_window(upcoming_phase_ids)


def evict_for_chunk_series(chunk_idx: int, total_chunks: int, phase_id: str) -> int:
    """§Perf Chunk-Series-Look-Ahead: behält Pitch-Modelle über 8 Chunks warm.

    Für §2.37 Pitch-Tracking über Chunk-Grenzen (phase_12, phase_31, phase_56 etc.):
    Chunk 1–8 laden ein Modell einmal, nutzen es 8x, dann Cleanup.
    Ohne Look-Ahead: jeder Chunk lädt Modell neu → 7x redundante Disk-Reads (35-80s × 7).
    Mit dieser Funktion: FCPE/RMVPE/CREPE werden über die gesamte 8-Chunk-Serie
    hinweg im RAM behalten.

    Args:
        chunk_idx: Chunk-Nummer (0-basiert, 0..7 für 8-Chunk-Serie)
        total_chunks: Gesamtzahl Chunks (typisch: 8)
        phase_id: Phase-Name (z. B. "phase_12_wow_flutter_fix")

    Returns:
        Anzahl der entladenen Plugins.

    Invariante:
        - Nur Modelle, die phase_id NICHT braucht, werden entladen
        - Chunk-1 bis Chunk-7: Modelle bleiben geschützt
        - Chunk-8: Nach Abschluss darf Cleanup-Phase Modelle entladen (nächste Phase
          beginnt neue Serie)
    """
    pitch_models = frozenset({"FCPE", "RMVPE", "CREPE"})
    phase_needed = _PHASE_REQUIRED_MODELS.get(phase_id, frozenset())

    # Modelle, die diese Phase BRAUCHT → bleiben über alle Chunks hinweg
    protected = pitch_models & phase_needed

    if not protected:
        # Phase hat keine Pitch-Modelle → Standard-Cleanup
        return get_plugin_lifecycle_manager().evict_if_needed()

    # Nur wenn wir NICHT in der letzten Chunk der Serie sind, schütze die Modelle
    mgr = get_plugin_lifecycle_manager()
    if chunk_idx < total_chunks - 1:
        # Chunk 1–7: Protektfenster = aktuelle Phase
        # Das Fenster wird in den nächsten 7 Chunks konsistent gehalten
        with mgr._lock:
            mgr._lookahead_models = protected
        # Standardmäßig: kein aggressives Evicten; nur bei RAM-Druck
        return mgr.evict_if_needed()
    else:
        # Chunk 8 (letzte): Fenster aufräumen → nächste Phase kann Cleanup starten
        with mgr._lock:
            mgr._lookahead_models = frozenset()
        # Nach letzter Chunk: Modelle sind nun "fair game" für Eviction
        return 0


def set_pipeline_active(active: bool) -> None:
    """Sperrt/entsperrt automatische Plugin-Eviction während Pipeline-Ausführung.

    Uses a refcount so nested enter/leave pairs work correctly:
    AurikDenker._run_rest() → UV3._execute_pipeline() both call this.
    Eviction blocked while count > 0.
    """
    mgr = get_plugin_lifecycle_manager()
    if active:
        mgr.enter_pipeline()
    else:
        mgr.leave_pipeline()


def cleanup_after_file() -> int:
    """Vollständiger Cleanup nach Abschluss einer Batch-Datei.

    Entlädt ALLE inaktiven Plugins. Sollte vom AurikDenker nach jeder
    verarbeiteten Datei im Batch-Modus aufgerufen werden.
    """
    mgr = get_plugin_lifecycle_manager()
    n = mgr.force_evict_all()
    gc.collect()
    logger.info("PLM: Batch-File-Cleanup: %d Plugin(s) entladen.", n)
    return n
