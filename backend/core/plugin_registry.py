"""backend/core/plugin_registry.py — §v10.700 H5.

from typing import Any
Plugin-Registry: Scannt plugins/, validiert Manifeste, stellt Discovery-API bereit.

Jedes Plugin muss ein manifest.json enthalten:
    {
      "name": "example_plugin",
      "version": "1.0.0",
      "description": "...",
      "author": "...",
      "entry_point": "plugin.py",
      "dependencies": [],
      "aurik_version_min": "10.15.0"
    }

Nutzung:
    registry = PluginRegistry()
    registry.discover()
    plugins = registry.list_plugins()
    # → [{"name": "...", "version": "...", "valid": true}, ...]
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = REPO_ROOT / "plugins"


@dataclass
class PluginInfo:
    """Informationen über ein entdecktes Plugin."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    path: str = ""
    entry_point: str = ""
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


class PluginRegistry:
    """Scannt und validiert Plugins im plugins/-Verzeichnis."""

    def __init__(self, plugins_dir: Path | None = None):
        self._plugins_dir = plugins_dir or PLUGINS_DIR
        self._plugins: dict[str, PluginInfo] = {}

    def discover(self) -> list[PluginInfo]:
        """Scannt plugins/ nach gültigen Plugin-Verzeichnissen."""
        self._plugins.clear()
        if not self._plugins_dir.exists():
            logger.warning("Plugins-Verzeichnis nicht gefunden: %s", self._plugins_dir)
            return []

        for item in sorted(self._plugins_dir.iterdir()):
            if not item.is_dir():
                continue
            if item.name.startswith("_") or item.name.startswith("."):
                continue
            if item.name == "sdk":
                continue

            manifest_path = item / "manifest.json"
            if not manifest_path.exists():
                continue

            plugin = self._load_plugin(item, manifest_path)
            self._plugins[plugin.name] = plugin

        logger.info("Plugin-Discovery: %d Plugins gefunden", len(self._plugins))
        return list(self._plugins.values())

    def _load_plugin(self, directory: Path, manifest_path: Path) -> PluginInfo:
        """Lädt und validiert ein einzelnes Plugin."""
        errors: list[str] = []
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("§V6 Manifest-Laden fehlgeschlagen für %s: %s — Plugin als ungültig markiert", directory.name, e)
            return PluginInfo(
                name=directory.name,
                path=str(directory),
                valid=False,
                errors=[f"Manifest-Fehler: {e}"],
            )

        name = manifest.get("name", directory.name)
        plugin = PluginInfo(
            name=name,
            version=manifest.get("version", "0.0.0"),
            description=manifest.get("description", ""),
            author=manifest.get("author", ""),
            path=str(directory),
            entry_point=manifest.get("entry_point", "plugin.py"),
            dependencies=manifest.get("dependencies", []),
        )

        # Pflichtfelder
        if not manifest.get("name"):
            errors.append("manifest.json: 'name' fehlt")
        if not manifest.get("version"):
            errors.append("manifest.json: 'version' fehlt")

        # Entry-Point existiert?
        entry_file = directory / plugin.entry_point
        if not entry_file.exists():
            errors.append(f"Entry-Point '{plugin.entry_point}' nicht gefunden")

        # Python-Syntax validieren
        if entry_file.exists():
            try:
                import ast

                ast.parse(entry_file.read_text())
            except SyntaxError as e:
                errors.append(f"Syntax-Fehler in {plugin.entry_point}: {e}")

        # AurikPlugin-Base-Class?
        if entry_file.exists() and not errors:
            content = entry_file.read_text()
            if "AurikPlugin" not in content and "class" in content:
                errors.append(f"Keine AurikPlugin-Base-Class in {plugin.entry_point} gefunden")

        plugin.errors = errors
        plugin.valid = len(errors) == 0
        return plugin

    def list_plugins(self) -> list[dict[str, Any]]:  # type: ignore[name-defined]
        """Gibt alle Plugins als Dict-Liste zurück."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "valid": p.valid,
                "errors": p.errors,
                "dependencies": p.dependencies,
            }
            for p in self._plugins.values()
        ]

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Findet ein Plugin nach Namen."""
        return self._plugins.get(name)

    def validate_all(self) -> bool:
        """Validiert alle Plugins. Gibt True zurück wenn alle gültig."""
        self.discover()
        return all(p.valid for p in self._plugins.values())

    def reload(self) -> list[PluginInfo]:
        """Erzwingt Rediscovery."""
        return self.discover()


# ── Singleton ────────────────────────────────────────────────────

_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Gibt die globale PluginRegistry-Instanz zurück."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        _registry.discover()
    return _registry
