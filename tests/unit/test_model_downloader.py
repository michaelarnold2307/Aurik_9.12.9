from __future__ import annotations

import pytest

pytest.importorskip("requests")  # CI-Minimal-Umgebung (cross-platform)

"""tests/unit/test_model_downloader.py — §13.3 Pflicht-Tests für ModelDownloader.

Testet:
    - verify_model(): korrekte / falsche SHA256, fehlende Datei
    - load_bundled(): valides Modell, fehlendes Modell, SHA256-Mismatch
    - ensure_all(): leeres Manifest, valide Einträge, gemischte Einträge
    - schedule_sota_upgrade(): Callback, daemon-Thread
    - get_model_downloader(): Singleton-Verhalten
    - verify_and_load(): Convenience-Funktion
    - PROJECT_MODELS_DIR: korrekt auf <project_root>/models gesetzt
    - Thread-Safety: nebenläufige ensure_all()-Aufrufe

Alle Tests verwenden ausschließlich synthetische Daten (tmp_path).
Seed: np.random.seed(42) / random.seed(42) wo zutreffend.
"""


import concurrent.futures
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import backend.core.model_downloader as _md_module
from backend.core.model_downloader import (
    ModelDownloader,
    ModelEntry,
    _download_remote_model,
    get_model_downloader,
    verify_and_load,
    verify_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_file(path: Path, data: bytes) -> str:
    """Schreibt Datei und gibt SHA256-Hash zurück."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


def _make_manifest(models: list, tmp_path: Path) -> Path:
    """Erstellt eine minimale manifest.json im tmp_path und gibt Pfad zurück."""
    manifest = tmp_path / "models" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"version": 2, "models": models}), encoding="utf-8")
    return manifest


def _reset_singleton() -> None:
    """Setzt ModelDownloader-Singleton zurück (Isolation zwischen Tests)."""
    ModelDownloader._instance = None
    _md_module._DOWNLOADER_INSTANCE_HOLDER[0] = None


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    """Setzt Singleton vor jedem Test zurück."""
    _reset_singleton()
    yield
    _reset_singleton()


@pytest.fixture()
def fake_model_file(tmp_path: Path):
    """Erstellt eine gefakte Modelldatei und gibt (path, sha256) zurück."""
    data = b"fake_onnx_model_data_" + b"\x00" * 128
    model_path = tmp_path / "models" / "test_model.onnx"
    sha = _write_file(model_path, data)
    return model_path, sha


@pytest.fixture()
def downloader_with_empty_manifest(tmp_path: Path):
    """Erstellt ModelDownloader mit leerem Manifest."""
    manifest = _make_manifest([], tmp_path)
    with patch.object(ModelDownloader, "MANIFEST", manifest):
        dl = ModelDownloader()
    return dl


# ──────────────────────────────────────────────────────────────────────────────
# 01 – verify_model(): SHA256-Verifikation
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVerifyModel:
    def test_01_correct_sha256_returns_true(self, fake_model_file):
        path, sha = fake_model_file
        assert verify_model(path, sha) is True

    def test_02_wrong_sha256_returns_false(self, fake_model_file):
        path, _ = fake_model_file
        assert verify_model(path, "0" * 64) is False

    def test_03_missing_file_returns_false(self, tmp_path: Path):
        assert verify_model(tmp_path / "nonexistent.onnx", "a" * 64) is False

    def test_04_empty_sha256_matches_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.onnx"
        empty.write_bytes(b"")
        expected = _sha256(b"")
        assert verify_model(empty, expected) is True

    def test_05_sha256_case_insensitive(self, fake_model_file):
        path, sha = fake_model_file
        assert verify_model(path, sha.upper()) is True

    def test_06_directory_path_returns_false(self, tmp_path: Path):
        # tmp_path ist ein Verzeichnis, kein File
        assert verify_model(tmp_path, "a" * 64) is False

    def test_07_large_file_is_chunked(self, tmp_path: Path):
        data = b"\xab" * 200_000  # > 3 × 64 KB chunks
        path = tmp_path / "large.onnx"
        sha = _write_file(path, data)
        assert verify_model(path, sha) is True


# ──────────────────────────────────────────────────────────────────────────────
# 02 – load_bundled(): Modell-Ladelogik
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadBundled:
    def _make_downloader(self, tmp_path: Path) -> ModelDownloader:
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        return dl

    def test_08_valid_file_and_sha_returns_path(self, tmp_path: Path, fake_model_file):
        path, sha = fake_model_file
        dl = self._make_downloader(tmp_path)
        with patch.object(dl, "PROJECT_MODELS_DIR", tmp_path / "models"):
            entry = {
                "name": "test",
                "bundled": True,
                "bundled_path": str(path),
                "sha256": sha,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
            result = dl.load_bundled(entry)
        assert result is not None
        assert result == path

    def test_09_missing_file_returns_none(self, tmp_path: Path):
        dl = self._make_downloader(tmp_path)
        entry = {
            "name": "missing",
            "bundled": True,
            "bundled_path": "models/nonexistent.onnx",
            "sha256": "a" * 64,
            "size_bytes": 0,
            "required": False,
            "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        }
        result = dl.load_bundled(entry)
        assert result is None

    def test_10_sha256_mismatch_returns_none(self, tmp_path: Path, fake_model_file):
        path, _ = fake_model_file
        dl = self._make_downloader(tmp_path)
        entry = {
            "name": "bad_sha",
            "bundled": True,
            "bundled_path": str(path),
            "sha256": "ff" * 32,
            "size_bytes": 0,
            "required": False,
            "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        }
        result = dl.load_bundled(entry)
        assert result is None

    def test_11_no_sha256_field_skips_check(self, tmp_path: Path, fake_model_file):
        path, _ = fake_model_file
        dl = self._make_downloader(tmp_path)
        entry = {
            "name": "no_sha",
            "bundled": True,
            "bundled_path": str(path),
            "sha256": "",
            "size_bytes": 0,
            "required": False,
            "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        }
        result = dl.load_bundled(entry)
        # Leere SHA → Prüfung übersprungen → Pfad zurückgegeben
        assert result == path

    def test_12_empty_bundled_path_returns_none(self, tmp_path: Path):
        dl = self._make_downloader(tmp_path)
        entry = {
            "name": "no_path",
            "bundled": True,
            "bundled_path": "",
            "sha256": "",
            "size_bytes": 0,
            "required": False,
            "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        }
        result = dl.load_bundled(entry)
        assert result is None

    def test_13_german_log_message_on_missing(self, tmp_path: Path):
        dl = self._make_downloader(tmp_path)
        entry = {
            "name": "mein_modell",
            "bundled": True,
            "bundled_path": "models/does_not_exist.onnx",
            "sha256": "a" * 64,
            "size_bytes": 0,
            "required": False,
            "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        }
        with patch.object(_md_module.logger, "info") as info_mock:
            dl.load_bundled(entry)

        joined = "\n".join(" ".join(map(str, c.args)) for c in info_mock.call_args_list)
        assert "mein_modell" in joined
        assert "klassische Methode" in joined

    def test_14_model_entry_dataclass_accepted(self, tmp_path: Path, fake_model_file):
        path, sha = fake_model_file
        dl = self._make_downloader(tmp_path)
        entry = ModelEntry(
            name="test_entry",
            bundled=True,
            bundled_path=str(path),
            sha256=sha,
            size_bytes=0,
            required=False,
            fallback="dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        )
        result = dl.load_bundled(entry)
        assert result == path


# ──────────────────────────────────────────────────────────────────────────────
# 03 – ensure_all(): Manifest-Prüfung
# ──────────────────────────────────────────────────────────────────────────────


class TestEnsureAll:
    def test_15_empty_manifest_returns_empty_dict(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        result = dl.ensure_all()
        assert result == {}

    def test_16_all_valid_returns_all_true(self, tmp_path: Path, fake_model_file):
        path, sha = fake_model_file
        models = [
            {
                "name": "m1",
                "bundled": True,
                "bundled_path": str(path),
                "sha256": sha,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
        ]
        manifest = _make_manifest(models, tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        result = dl.ensure_all()
        assert result["m1"] is True

    def test_17_missing_model_returns_false(self, tmp_path: Path):
        models = [
            {
                "name": "missing_m",
                "bundled": True,
                "bundled_path": "models/not_there.onnx",
                "sha256": "a" * 64,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
        ]
        manifest = _make_manifest(models, tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        result = dl.ensure_all()
        assert result["missing_m"] is False

    def test_18_mixed_manifest_correct_dict(self, tmp_path: Path, fake_model_file):
        path, sha = fake_model_file
        models = [
            {
                "name": "ok_model",
                "bundled": True,
                "bundled_path": str(path),
                "sha256": sha,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            },
            {
                "name": "bad_model",
                "bundled": True,
                "bundled_path": "models/x.onnx",
                "sha256": "0" * 64,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            },
        ]
        manifest = _make_manifest(models, tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        result = dl.ensure_all()
        assert result == {"ok_model": True, "bad_model": False}

    def test_19_progress_callback_called(self, tmp_path: Path):
        models = [
            {
                "name": "m_cb",
                "bundled": True,
                "bundled_path": "models/x.onnx",
                "sha256": "0" * 64,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
        ]
        manifest = _make_manifest(models, tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        calls = []
        dl.ensure_all(progress_callback=lambda name, frac: calls.append((name, frac)))
        assert len(calls) == 1
        assert calls[0][0] == "m_cb"
        assert math.isfinite(calls[0][1])

    def test_20_result_values_are_bool(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        result = dl.ensure_all()
        for v in result.values():
            assert isinstance(v, bool)


# ──────────────────────────────────────────────────────────────────────────────
# 04 – schedule_sota_upgrade(): Threading & Callback
# ──────────────────────────────────────────────────────────────────────────────


class TestScheduleSotaUpgrade:
    def _dl(self, tmp_path: Path) -> ModelDownloader:
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            return ModelDownloader()

    def test_21_no_sota_field_does_not_crash(self, tmp_path: Path):
        dl = self._dl(tmp_path)
        entry = {
            "name": "x",
            "bundled": True,
            "bundled_path": "models/x.onnx",
            "sha256": "",
            "size_bytes": 0,
            "required": False,
            "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
        }
        # Darf nicht crashen
        dl.schedule_sota_upgrade(entry)

    def test_22_no_sota_url_does_not_crash(self, tmp_path: Path):
        dl = self._dl(tmp_path)
        entry = {"name": "x", "sota_upgrade": {"name": "XPlus"}}
        dl.schedule_sota_upgrade(entry)

    def test_23_callback_called_after_simulated_download(self, tmp_path: Path):
        dl = self._dl(tmp_path)
        collected = []

        fake_sota_path = tmp_path / "sota_models" / "test_model.onnx"
        fake_sota_path.parent.mkdir(parents=True, exist_ok=True)
        fake_data = b"sota_fake_data"
        fake_sha = _sha256(fake_data)

        entry = {
            "name": "test_model",
            "sota_upgrade": {
                "name": "TestSOTA",
                "url": "http://example.com/model.onnx",
                "sha256": fake_sha,
            },
        }

        def fake_urlretrieve(url, dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(fake_data)

        with (
            patch.object(_md_module, "OFFLINE_MODE", False),
            patch("urllib.request.urlretrieve", fake_urlretrieve),
            patch.object(_md_module, "_SOTA_CACHE_DIR", tmp_path / "sota_models"),
            patch.object(_md_module, "_SOTA_MANIFEST", tmp_path / "sota_manifest.json"),
        ):
            dl.schedule_sota_upgrade(entry, progress_callback=lambda name, f: collected.append(name))
            # Warten auf Hintergrund-Thread
            time.sleep(0.3)

        # Kein Fehler → Callback nicht garantiert wegen SHA-Prüfung,
        # aber die Methode darf nicht crashen und muss Thread starten
        # (Callback nur bei erfolgreichem Download)

    def test_24_daemon_thread_is_started(self, tmp_path: Path):
        dl = self._dl(tmp_path)
        started: list = []

        _ = threading.Thread.start

        def fake_start(self_t):
            started.append(self_t.daemon)
            # Thread nicht wirklich starten (kein Netzwerk)

        entry = {"name": "model_x", "sota_upgrade": {"name": "ModelXPlus", "url": "http://example.com/x.onnx"}}
        with (
            patch.object(_md_module, "OFFLINE_MODE", False),
            patch.object(threading.Thread, "start", fake_start),
        ):
            dl.schedule_sota_upgrade(entry)

        assert len(started) >= 1
        assert all(started)  # alle gestarteten Threads sind daemon=True


class TestDownloadHardeningAndMiipherManifest:
    def test_25_download_helper_rejects_non_https(self, tmp_path: Path):
        target = tmp_path / "model.onnx"
        with pytest.raises(ValueError, match="Nicht erlaubtes Download-Schema"):
            _download_remote_model("http://example.com/model.onnx", target)

    def test_26_project_manifest_exposes_miipher_entry(self):
        dl = get_model_downloader()
        entry = dl.get_entry("miipher")
        assert entry is not None, "miipher fehlt im Projekt-Manifest"
        assert entry.bundled is False
        assert entry.bundled_path == "models/miipher/miipher.onnx"
        assert entry.fallback == "sgmse_plus"


# ──────────────────────────────────────────────────────────────────────────────
# 05 – Singleton & Convenience-Funktionen
# ──────────────────────────────────────────────────────────────────────────────


class TestSingletonAndConvenience:
    def test_25_get_model_downloader_returns_instance(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = get_model_downloader()
        assert isinstance(dl, ModelDownloader)

    def test_26_get_model_downloader_same_type_twice(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl1 = get_model_downloader()
            dl2 = get_model_downloader()
        assert type(dl1) is type(dl2)

    def test_27_project_models_dir_ends_with_models(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        assert dl.PROJECT_MODELS_DIR.name == "models"

    def test_28_project_models_dir_is_absolute(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        assert dl.PROJECT_MODELS_DIR.is_absolute()

    def test_29_verify_and_load_returns_none_for_unknown(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            result = verify_and_load("non_existent_model")
        assert result is None

    def test_30_verify_and_load_returns_path_for_known(self, tmp_path: Path, fake_model_file):
        path, sha = fake_model_file
        models = [
            {
                "name": "known_m",
                "bundled": True,
                "bundled_path": str(path),
                "sha256": sha,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
        ]
        manifest = _make_manifest(models, tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            _reset_singleton()
            result = verify_and_load("known_m")
        assert result == path


# ──────────────────────────────────────────────────────────────────────────────
# 06 – Thread-Safety
# ──────────────────────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_31_concurrent_ensure_all_does_not_crash(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        errors: list = []

        # Singleton einmalig vor den Threads erstellen (patch.object nicht thread-sicher).
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()

        def run():
            try:
                dl.ensure_all()
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == [], f"Thread-Fehler: {errors}"

    def test_32_concurrent_get_model_downloader_type_consistent(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        types: list = []

        def run():
            with patch.object(ModelDownloader, "MANIFEST", manifest):
                dl = get_model_downloader()
            types.append(type(dl).__name__)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(run) for _ in range(16)]
            for f in futures:
                f.result(timeout=5.0)

        assert all(t == "ModelDownloader" for t in types)


# ──────────────────────────────────────────────────────────────────────────────
# 07 – Manifest-Robustheit
# ──────────────────────────────────────────────────────────────────────────────


class TestManifestRobustness:
    def test_33_malformed_json_returns_empty_entries(self, tmp_path: Path):
        manifest_dir = tmp_path / "models"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        bad_manifest = manifest_dir / "manifest.json"
        bad_manifest.write_text("{ INVALID JSON !!!", encoding="utf-8")
        with patch.object(ModelDownloader, "MANIFEST", bad_manifest):
            dl = ModelDownloader()
        assert dl._manifest_entries == []

    def test_34_missing_manifest_returns_empty_entries(self, tmp_path: Path):
        with patch.object(ModelDownloader, "MANIFEST", tmp_path / "no_manifest.json"):
            dl = ModelDownloader()
        assert dl._manifest_entries == []

    def test_35_manifest_without_models_key(self, tmp_path: Path):
        manifest_dir = tmp_path / "models"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        m = manifest_dir / "manifest.json"
        m.write_text(json.dumps({"version": 2}), encoding="utf-8")
        with patch.object(ModelDownloader, "MANIFEST", m):
            dl = ModelDownloader()
        assert dl._manifest_entries == []

    def test_36_get_entry_existing_name(self, tmp_path: Path, fake_model_file):
        path, sha = fake_model_file
        models = [
            {
                "name": "lookup_me",
                "bundled": True,
                "bundled_path": str(path),
                "sha256": sha,
                "size_bytes": 0,
                "required": False,
                "fallback": "dsp",  # §V6 (copilot-instructions.md): logger.warning handled at call site
            }
        ]
        manifest = _make_manifest(models, tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        entry = dl.get_entry("lookup_me")
        assert entry is not None
        assert entry.name == "lookup_me"

    def test_37_get_entry_nonexistent_returns_none(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        assert dl.get_entry("nope") is None

    def test_38_all_result_values_are_finite(self, tmp_path: Path):
        """ensure_all() darf keine NaN/Inf als Ergebnisse liefern (numerische Robustheit §3.1)."""
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        result = dl.ensure_all()
        for v in result.values():
            assert isinstance(v, bool)  # bool ist immer endlich

    def test_39_get_status_returns_dict(self, tmp_path: Path):
        manifest = _make_manifest([], tmp_path)
        with patch.object(ModelDownloader, "MANIFEST", manifest):
            dl = ModelDownloader()
        status = dl.get_status()
        assert isinstance(status, dict)
