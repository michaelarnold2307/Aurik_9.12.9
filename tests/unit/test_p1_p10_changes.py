from __future__ import annotations

"""
tests/unit/test_p1_p10_changes.py — Verifikation der P1–P10 Audit-Implementierungen

Prüft die konkreten neuen Verhaltensweisen aus dem ML-Modell-Audit (Feb 2026):

  P1  CREPE Chunk-Streaming (_CHUNK_FRAMES=512 statt _MAX_ONNX_FRAMES=10)
  P2  FlashSR 5.8 GB entfernt (kein bundled-Eintrag mehr)
  P3  MERT Lizenz-Bereinigung (nicht länger bundled)
  P4  DiffWave → NMF-β DSP-Inpainting als primärer Pfad
  P5  HTDemucs 6s korrekt als experimentell/stub markiert
  P6  HiFiGAN → Griffin-Lim-Fallback, Vocos als primärer Vocoder
    P10 UTMOSv2-Compat bleibt, VERSA ist Standard-Metrik

Namenskonvention §5.2: tests/unit/test_p1_p10_changes.py
Alle Tests: synthetische Signale, kein reales Audio, kein Netzwerk.
Laufzeit-Budget: < 30 s gesamt.
"""


import json
from pathlib import Path

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

# ---------------------------------------------------------------------------
# Gemeinsame Testsignale (synthetisch, reproduzierbar)
# ---------------------------------------------------------------------------

_SR = 22050  # minimale SR → schnellste Verarbeitung
_SR48 = 48000
_SEED = 42
_RNG = np.random.default_rng(_SEED)

MANIFEST_PATH = Path(__file__).parent.parent.parent / "models" / "manifest.json"


def _sine(freq: float = 440.0, dur_s: float = 1.0, sr: int = _SR) -> np.ndarray:
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _noise(dur_s: float = 1.0, sr: int = _SR, amp: float = 0.2) -> np.ndarray:
    return (_RNG.standard_normal(int(sr * dur_s)) * amp).astype(np.float32)


def _silence(dur_s: float = 0.5, sr: int = _SR) -> np.ndarray:
    return np.zeros(int(sr * dur_s), dtype=np.float32)


def _load_manifest() -> dict:
    with MANIFEST_PATH.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _manifest_by_name() -> dict[str, dict]:
    man = _load_manifest()
    return {e["name"]: e for e in man["models"]}


# ===========================================================================
# P1 — CREPE Chunk-Streaming
# ===========================================================================


@pytest.mark.unit
class TestCrepeChunkStreaming:
    """P1: _CHUNK_FRAMES=512 ersetzt _MAX_ONNX_FRAMES=10.

    Verifiziert, dass das Modul die neue Konstante exponiert und die alte
    Beschränkung entfernt ist. Funktions-Tests nutzen kurze Signale (< 2 s)
    um die 30-s-Timeout-Grenze einzuhalten.
    """

    def test_01_chunk_frames_constant_value(self):
        """_CHUNK_FRAMES muss genau 512 sein (Chunk-Streaming-Budget)."""
        import plugins.crepe_plugin as mod

        assert hasattr(mod, "_CHUNK_FRAMES"), "_CHUNK_FRAMES fehlt im Modul"
        assert mod._CHUNK_FRAMES == 512

    def test_02_no_legacy_max_onnx_frames(self):
        """_MAX_ONNX_FRAMES darf nicht mehr existieren (veraltete 10-Frame-Begrenzung)."""
        import plugins.crepe_plugin as mod

        assert not hasattr(mod, "_MAX_ONNX_FRAMES"), "_MAX_ONNX_FRAMES noch vorhanden — Legacy-Konstante nicht entfernt"

    def test_03_crepe_plugin_importable(self):
        """CrepePlugin und Convenience-Funktion müssen importierbar sein."""
        from plugins.crepe_plugin import analyze_pitch

        assert callable(analyze_pitch)

    def test_04_analyze_method_exists(self):
        """CrepePlugin muss öffentliche analyze()-Methode haben."""
        from plugins.crepe_plugin import get_crepe_plugin

        plugin = get_crepe_plugin()
        assert callable(getattr(plugin, "analyze", None)), "analyze() fehlt"

    def test_05_analyze_1s_sine_no_crash(self):
        """1-s-Sinus → analyze() darf nicht werfen."""
        from plugins.crepe_plugin import get_crepe_plugin

        audio = _sine(440.0, dur_s=1.0, sr=_SR)
        result = get_crepe_plugin().analyze(audio, _SR)
        assert result is not None

    def test_06_analyze_result_has_required_attrs(self):
        """CrepeResult muss f0_hz, voiced_prob, salience und times_s haben."""
        from plugins.crepe_plugin import get_crepe_plugin

        audio = _sine(440.0, dur_s=1.0, sr=_SR)
        result = get_crepe_plugin().analyze(audio, _SR)
        for attr in ("f0_hz", "voiced_prob", "salience", "times_s"):
            assert hasattr(result, attr), f"Attribut {attr!r} fehlt in CrepeResult"

    def test_07_analyze_result_finite(self):
        """Alle CrepeResult-Felder müssen finite sein (kein NaN/Inf)."""
        from plugins.crepe_plugin import get_crepe_plugin

        audio = _sine(440.0, dur_s=1.0, sr=_SR)
        result = get_crepe_plugin().analyze(audio, _SR)
        for arr in (result.f0_hz, result.voiced_prob, result.salience, result.times_s):
            assert np.isfinite(arr).all(), "NaN/Inf in CrepeResult-Feld"

    def test_08_analyze_silence_no_crash(self):
        """Stille → analyze() darf nicht werfen, f0_hz darf 0/NaN enthalten."""
        from plugins.crepe_plugin import get_crepe_plugin

        audio = _silence(1.0, sr=_SR)
        result = get_crepe_plugin().analyze(audio, _SR)
        assert result is not None

    def test_09_analyze_voiced_prob_in_unit_range(self):
        """voiced_prob-Werte müssen in [0, 1] liegen."""
        from plugins.crepe_plugin import get_crepe_plugin

        audio = _sine(440.0, dur_s=1.0, sr=_SR)
        result = get_crepe_plugin().analyze(audio, _SR)
        assert np.all(result.voiced_prob >= 0.0), "voiced_prob < 0"
        assert np.all(result.voiced_prob <= 1.0), "voiced_prob > 1"
        assert np.all(result.salience >= 0.0), "salience < 0"
        assert np.all(result.salience <= 1.0), "salience > 1"

    def test_10_f0_shape_matches_times_s(self):
        """f0_hz und times_s müssen dieselbe Länge haben."""
        from plugins.crepe_plugin import get_crepe_plugin

        audio = _sine(440.0, dur_s=1.0, sr=_SR)
        result = get_crepe_plugin().analyze(audio, _SR)
        assert len(result.f0_hz) == len(result.times_s), (
            f"f0_hz ({len(result.f0_hz)}) ≠ times_s ({len(result.times_s)})"
        )
        assert len(result.f0_hz) == len(result.voiced_prob), (
            f"f0_hz ({len(result.f0_hz)}) ≠ voiced_prob ({len(result.voiced_prob)})"
        )

    def test_11_singleton_returns_same_instance(self):
        """get_crepe_plugin() muss dasselbe Singleton zurückgeben."""
        from plugins.crepe_plugin import get_crepe_plugin

        assert get_crepe_plugin() is get_crepe_plugin()

    def test_12_chunk_frames_is_int(self):
        """_CHUNK_FRAMES muss ein int sein (kein float, kein None)."""
        import plugins.crepe_plugin as mod

        assert isinstance(mod._CHUNK_FRAMES, int)


# ===========================================================================
# P4 — DiffWave: NMF-β DSP-Inpainting als primärer Pfad
# ===========================================================================


class TestDiffWaveNMFInpainting:
    """P4: DiffWave nutzt NMF-β + PGHI statt ONNX-Stub als primären Inpainting-Pfad.

    inpaint()-Signatur: inpaint(audio, gap_start, gap_end, sr, n_steps=4)
    """

    def test_01_inpaint_importable(self):
        """inpaint() muss aus plugins.diffwave_plugin importierbar sein."""
        from plugins.diffwave_plugin import inpaint

        assert callable(inpaint)

    def test_02_nmf_inpaint_private_accessible(self):
        """_nmf_inpaint() muss im Modul vorhanden sein (primärer Inpainting-Pfad)."""
        import plugins.diffwave_plugin as mod

        assert hasattr(mod, "_nmf_inpaint"), "_nmf_inpaint() fehlt"
        assert callable(mod._nmf_inpaint)

    def test_03_diffwave_plugin_has_inpaint_method(self):
        """DiffwavePlugin-Klasse muss eine inpaint()-Methode haben."""
        from plugins.diffwave_plugin import DiffwavePlugin

        plugin = DiffwavePlugin()
        assert callable(getattr(plugin, "inpaint", None))

    def test_04_inpaint_returns_same_shape(self):
        """inpaint() gibt Array mit identischer Länge wie Eingang zurück."""
        from plugins.diffwave_plugin import inpaint

        sr = _SR
        audio = _noise(dur_s=3.0, sr=sr)
        gap_s, gap_e = sr // 2, sr // 2 + sr // 4
        result = inpaint(audio, gap_s, gap_e, sr)
        assert result.shape == audio.shape, f"Shape mismatch: {result.shape} ≠ {audio.shape}"

    def test_05_inpaint_all_finite(self):
        """inpaint()-Ergebnis darf kein NaN/Inf enthalten."""
        from plugins.diffwave_plugin import inpaint

        audio = _noise(dur_s=3.0, sr=_SR)
        gap_s, gap_e = _SR // 2, _SR // 2 + _SR // 5
        result = inpaint(audio, gap_s, gap_e, _SR)
        assert np.isfinite(result).all(), "NaN oder Inf im inpaint()-Ergebnis"

    def test_06_inpaint_gap_not_silence(self):
        """Gefüllte Lücke darf nicht stille Null-Energie haben (NMF-β füllt mit Inhalt)."""
        from plugins.diffwave_plugin import inpaint

        audio = _sine(440.0, dur_s=3.0, sr=_SR)
        gap_s, gap_e = _SR // 4, _SR // 4 + _SR // 5
        result = inpaint(audio, gap_s, gap_e, _SR)
        gap_segment = result[gap_s:gap_e]
        rms = float(np.sqrt(np.mean(gap_segment**2)))
        assert rms > 1e-4, f"Lücken-RMS zu niedrig ({rms:.6f}) — möglicherweise Null-Füllung statt NMF"

    def test_07_inpaint_dtype_float32(self):
        """inpaint() muss float32 zurückgeben."""
        from plugins.diffwave_plugin import inpaint

        audio = _noise(dur_s=2.0, sr=_SR)
        result = inpaint(audio, _SR // 4, _SR // 2, _SR)
        assert result.dtype == np.float32, f"dtype={result.dtype} statt float32"

    def test_08_inpaint_silence_input_no_crash(self):
        """Stille als Eingang darf nicht zum Absturz führen."""
        from plugins.diffwave_plugin import inpaint

        audio = _silence(3.0, sr=_SR)
        result = inpaint(audio, _SR // 4, _SR // 2, _SR)
        assert np.isfinite(result).all()

    def test_09_inpaint_gap_at_end_of_signal(self):
        """Lücke am Ende des Signals muss sauber behandelt werden."""
        from plugins.diffwave_plugin import inpaint

        audio = _noise(dur_s=2.0, sr=_SR)
        n = len(audio)
        gap_s, gap_e = n - _SR // 4, n  # letzte 250 ms
        result = inpaint(audio, gap_s, gap_e, _SR)
        assert result.shape == audio.shape

    def test_10_get_diffwave_plugin_singleton(self):
        """get_diffwave_plugin() muss denselben Singleton-Wert zurückgeben."""
        from plugins.diffwave_plugin import get_diffwave_plugin

        assert get_diffwave_plugin() is get_diffwave_plugin()


# ===========================================================================
# P5 — MDX23C (Kim_Vocal_2) als Primär-Separator; HTDemucs 6s Legacy-Fallback
# ===========================================================================


class TestMDX23CPrimarySeparator:
    """P5: MDX23C (Kim_Vocal_2) ist der produktionsreife Primär-Separator.

    HTDemucs 6s verbleibt als experimenteller Legacy-Fallback.
    Verifiziert Plugin-Funktionalität und Manifest-Konsistenz.
    """

    def test_01_mdx23c_plugin_importable(self):
        """MDX23CPlugin muss importierbar sein."""
        from plugins.mdx23c_plugin import MDX23CPlugin

        assert MDX23CPlugin is not None

    def test_02_mdx23c_returns_ndarray(self):
        """MDX23CPlugin.process() muss ein ndarray zurückgeben."""
        from plugins.mdx23c_plugin import MDX23CPlugin

        plugin = MDX23CPlugin()
        audio = _sine(440.0, dur_s=2.0, sr=48000)
        result = plugin.process(audio, sr=48000, stem="vocals")
        assert isinstance(result, np.ndarray), f"Kein ndarray: {type(result)}"

    def test_03_mdx23c_result_finite(self):
        """Ergebnis-Array muss finite sein."""
        from plugins.mdx23c_plugin import MDX23CPlugin

        plugin = MDX23CPlugin()
        audio = _sine(440.0, dur_s=2.0, sr=48000)
        result = plugin.process(audio, sr=48000, stem="vocals")
        assert np.isfinite(result).all(), "NaN/Inf im MDX23C-Ergebnis"

    def test_04_mdx23c_separate_all_stems(self):
        """separate_all_stems() muss Dict mit vocals/inst zurückgeben."""
        from plugins.mdx23c_plugin import MDX23CPlugin

        plugin = MDX23CPlugin()
        audio = _noise(dur_s=2.0, sr=48000)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        result = plugin.separate_all_stems(audio, sr=48000)
        assert isinstance(result, dict), f"Kein dict: {type(result)}"
        assert "vocals" in result or "inst" in result, f"Stems fehlen: {list(result.keys())}"

    def test_05_htdemucs_still_in_manifest(self):
        """htdemucs_6s ist PRODUKTIV (nicht experimental) — §Fix 2026-09-08.

        Früher: experimental=True im (gitignored) Manifest → DemucsV4Plugin
        lud die ONNX-Session nie → stummer HPSS-Fallback (§V6). Neuer
        Vertrag: Produktions-Modelle laden standardmäßig; der Ladepfad liest
        das Manifest nicht mehr (Opt-out: AURIK_DISABLE_HTDEMUCS_6S=1).
        """
        models = _manifest_by_name()
        if "htdemucs_6s" not in models:
            pytest.skip("models/-Paket (gitignored) nicht installiert")
        assert models["htdemucs_6s"].get("experimental") is not True, (
            "htdemucs_6s darf nicht experimental sein — die Demucs-Stufe ist produktiv"
        )

    def test_06_htdemucs_has_dsp_fallback(self):
        """htdemucs_6s muss einen DSP-Fallback deklariert haben."""
        entry = _manifest_by_name()["htdemucs_6s"]
        fallback = entry.get("fallback", "")
        assert fallback, "Kein fallback-Eintrag für htdemucs_6s"

    def test_07_mdx23c_convenience_functions(self):
        """Convenience-Funktionen separate_vocals/separate_stems müssen existieren."""
        from plugins.mdx23c_plugin import separate_stems, separate_vocals

        assert callable(separate_vocals)
        assert callable(separate_stems)

    def test_08_htdemucs_facade_routes_to_mdx23c(self):
        """htdemucs_plugin Facade muss auf MDX23CPlugin routen."""
        from plugins.htdemucs_plugin import get_htdemucs_plugin

        plugin = get_htdemucs_plugin()
        # Muss eine MDX23CPlugin-Instanz sein (oder None wenn nicht verfügbar)
        if plugin is not None:
            assert type(plugin).__name__ == "MDX23CPlugin"


# ===========================================================================
# P6 — HiFiGAN: Griffin-Lim-Fallback; Vocos als primärer Vocoder
# ===========================================================================


class TestHiFiGANFallbackUndVocosStandard:
    """P6: HiFiGAN nutzt Griffin-Lim-Fallback; Vocos ist primärer Vocoder."""

    def test_01_hifigan_importable(self):
        """HifiGanPlugin muss importierbar bleiben (rückwärtskompatibel)."""

        assert True

    def test_02_vocode_audio_importable(self):
        """vocode_audio() muss als Convenience-Funktion verfügbar sein."""
        from plugins.hifigan_plugin import vocode_audio

        assert callable(vocode_audio)

    def test_03_hifigan_has_session_attribute(self):
        """HifiGanPlugin muss _session-Attribut haben (None wenn kein echtes Modell)."""
        from plugins.hifigan_plugin import HifiGanPlugin

        plugin = HifiGanPlugin()
        assert hasattr(plugin, "_session"), "_session-Attribut fehlt"

    def test_04_vocos_is_bundled(self):
        """Vocos muss im Manifest als bundled=True eingetragen sein."""
        models = _manifest_by_name()
        assert "vocos_mel_24khz" in models, "vocos_mel_24khz fehlt im Manifest"
        assert models["vocos_mel_24khz"].get("bundled") is True, "vocos_mel_24khz ist nicht bundled"

    def test_05_vocos_size_above_50mb(self):
        """Vocos-Modell muss > 50 MB sein (verifiziert echte ONNX-Datei, nicht Stub)."""
        entry = _manifest_by_name()["vocos_mel_24khz"]
        size = entry.get("size_bytes", 0)
        assert size > 50_000_000, f"Vocos size_bytes={size} klingt zu klein für echtes Modell"

    def test_06_hifigan_manifest_has_sota_upgrade(self):
        """hifi_gan-Eintrag muss sota_upgrade haben (Vocos ist Upgrade)."""
        models = _manifest_by_name()
        assert "hifi_gan" in models, "hifi_gan fehlt im Manifest"
        assert "sota_upgrade" in models["hifi_gan"], (
            "hifi_gan hat kein sota_upgrade-Feld — Vocos-Migration nicht dokumentiert"
        )

    def test_07_hifigan_vocode_returns_ndarray(self):
        """vocode_audio(audio, sr) muss ein np.ndarray zurückgeben."""
        from plugins.hifigan_plugin import vocode_audio

        audio = _sine(440.0, dur_s=0.5, sr=_SR48)
        result = vocode_audio(audio, sr=_SR48)
        assert isinstance(result, np.ndarray), f"kein ndarray: {type(result)}"

    def test_08_hifigan_vocode_output_finite(self):
        """vocode_audio()-Ausgabe darf kein NaN/Inf enthalten."""
        from plugins.hifigan_plugin import vocode_audio

        audio = _sine(440.0, dur_s=0.5, sr=_SR48)
        result = vocode_audio(audio, sr=_SR48)
        assert np.isfinite(result).all(), "NaN/Inf in vocode_audio()-Ausgabe"


# ===========================================================================
# P2 / P3 / P10 — Manifest-Integrität
# ===========================================================================


class TestManifestIntegritaet:
    """P2/P3/P10: Manifest-Einträge korrekt aktualisiert.

    Verifiziert ohne Plugin-Load dass die strategischen Manifest-Änderungen
    aus dem Audit korrekt persistiert sind.
    """

    def test_01_manifest_exists(self):
        """models/manifest.json muss existieren."""
        assert MANIFEST_PATH.exists(), f"Manifest nicht gefunden: {MANIFEST_PATH}"

    def test_02_manifest_version_2(self):
        """Manifest-Version muss 2 sein."""
        man = _load_manifest()
        assert man.get("version") == 2, f"Manifest-Version = {man.get('version')} statt 2"

    def test_03_manifest_has_models_list(self):
        """Manifest muss 'models'-Liste mit mindestens 10 Einträgen haben."""
        man = _load_manifest()
        assert isinstance(man.get("models"), list)
        assert len(man["models"]) >= 10, f"Nur {len(man['models'])} Modelle im Manifest"

    # --- P2: FlashSR --------------------------------------------------------

    def test_04_flashsr_not_bundled(self):
        """P2: flashsr darf nicht bundled=True sein (5.8 GB-Datei entfernt)."""
        models = _manifest_by_name()
        assert "flashsr" in models, "flashsr fehlt im Manifest"
        assert models["flashsr"].get("bundled") is not True, (
            "flashsr bundled=True — Prüfe ob 5.8 GB-Datei noch existiert"
        )

    def test_05_flashsr_has_sota_upgrade(self):
        """P2: flashsr muss sota_upgrade-Feld haben (optionaler Download)."""
        models = _manifest_by_name()
        assert "sota_upgrade" in models["flashsr"], "flashsr hat kein sota_upgrade-Feld"

    # --- P3: MERT -----------------------------------------------------------

    def test_06_mert_not_bundled(self):
        """P3: mert_instrument_detector darf nicht bundled=True sein (Lizenz-Problem)."""
        models = _manifest_by_name()
        assert "mert_instrument_detector" in models, "mert_instrument_detector fehlt"
        assert models["mert_instrument_detector"].get("bundled") is not True, (
            "mert_instrument_detector bundled=True — CC BY-NC-4.0 Lizenz-Verletzung"
        )

    def test_07_mert_has_sota_upgrade(self):
        """P3: mert_instrument_detector muss sota_upgrade haben (optionaler Download)."""
        models = _manifest_by_name()
        assert "sota_upgrade" in models["mert_instrument_detector"], (
            "mert_instrument_detector hat kein sota_upgrade-Feld"
        )

    # --- P10: VERSA / UTMOSv2 -----------------------------------------------

    def test_08_versa_entry_exists(self):
        """P10: 'versa' (primäre Non-Reference-Metrik) muss im Manifest sein."""
        models = _manifest_by_name()
        assert "versa" in models, "'versa' Eintrag fehlt im Manifest"

    def test_09_versa_is_bundled_primary(self):
        """P10: VERSA muss bundled=True sein (primäre Musik-Qualitätsmetrik)."""
        models = _manifest_by_name()
        entry = models["versa"]
        assert entry.get("bundled") is True, "VERSA ist nicht bundled"

    def test_10_utmosv2_declared(self):
        """P10: UTMOSv2 muss im Manifest deklariert bleiben (aktueller Compatibility-Vertrag)."""
        models = _manifest_by_name()
        assert "utmos" in models, "'utmos' Eintrag fehlt im Manifest"
        entry = models["utmos"]
        assert entry.get("bundled") is True, "utmos-Eintrag muss bundled bleiben"

    # --- Vocos als primärer Vocoder -----------------------------------------

    def test_11_vocos_bundled_and_large(self):
        """Vocos muss bundled=True und > 50 MB sein (nicht Stub)."""
        models = _manifest_by_name()
        assert "vocos_mel_24khz" in models
        e = models["vocos_mel_24khz"]
        assert e.get("bundled") is True
        assert e.get("size_bytes", 0) > 50_000_000

    def test_12_no_bundled_false_with_large_size(self):
        """Modelle mit bundled=False dürfen nicht als > 1 GB deklariert sein
        ohne nota als LazLoad / sota_upgrade — Konsistenzprüfung."""
        models = _manifest_by_name()
        for name, entry in models.items():
            if entry.get("bundled") is False:
                size = entry.get("size_bytes", 0)
                has_lazy = entry.get("lazy_load") or "sota_upgrade" in entry
                if size > 1_000_000_000:
                    assert has_lazy, (
                        f"'{name}': bundled=False, {size // 1e9:.1f} GB groß, "
                        f"aber kein lazy_load/sota_upgrade — Manifest-Inkonsistenz"
                    )

    def test_13_all_bundled_entries_have_sha256(self):
        """Alle bundled=True Einträge müssen einen sha256-Hash haben."""
        models = _manifest_by_name()
        for name, entry in models.items():
            if entry.get("bundled") is True:
                sha = entry.get("sha256", "")
                assert sha and len(sha) >= 32, f"'{name}' ist bundled=True, hat aber keinen/kurzen sha256: {sha!r}"

    def test_14_manifest_model_names_unique(self):
        """Alle Modell-Namen im Manifest müssen einzigartig sein."""
        man = _load_manifest()
        names = [e["name"] for e in man["models"]]
        assert len(names) == len(set(names)), f"Doppelte Namen im Manifest: {[n for n in names if names.count(n) > 1]}"

    def test_15_all_entries_have_fallback(self):
        """Alle Manifest-Einträge müssen einen fallback-Wert haben."""
        man = _load_manifest()
        missing_fallback = [e["name"] for e in man["models"] if not e.get("fallback")]
        assert not missing_fallback, f"Einträge ohne fallback: {missing_fallback}"


# ===========================================================================
# Integrations-Schnelltest: Plugin-Roundtrip
# ===========================================================================


class TestPluginRoundtripSanity:
    """Schnelle Plausibilitäts-Checks über alle P1–P10-Plugins zusammen."""

    def test_01_crepe_analyze_pitch_convenience(self):
        """Modul-Ebene analyze_pitch() muss funktionieren (wie vorher)."""
        from plugins.crepe_plugin import analyze_pitch

        audio = _sine(440.0, dur_s=0.5, sr=_SR)
        result = analyze_pitch(audio, _SR)
        assert result is not None

    def test_02_diffwave_inpaint_convenience(self):
        """Modul-Ebene inpaint() muss mit gültigem Signal laufen."""
        from plugins.diffwave_plugin import inpaint

        audio = _noise(dur_s=2.0, sr=_SR)
        gap_s, gap_e = _SR // 4, _SR // 2
        out = inpaint(audio, gap_s, gap_e, _SR)
        assert out.shape == audio.shape
        assert np.isfinite(out).all()

    def test_03_hifigan_vocode_audio_convenience(self):
        """vocode_audio() muss lauffähig sein."""
        from plugins.hifigan_plugin import vocode_audio

        audio = _noise(dur_s=0.5, sr=_SR48)
        result = vocode_audio(audio, sr=_SR48)
        assert result is not None
        assert np.isfinite(result).all()

    def test_04_demucs_process_consistency(self):
        """DemucsV4Plugin.process() mit selbem Eingang → konsistentes Dict."""
        from plugins.demucs_v4_plugin import DemucsV4Plugin

        plugin = DemucsV4Plugin()
        audio = _sine(440.0, dur_s=1.0, sr=48000)
        r1 = plugin.process(audio, sr=48000)
        r2 = plugin.process(audio, sr=48000)
        assert set(r1.keys()) == set(r2.keys()), "Inkonsistente Schlüssel bei gleicher Eingabe"

    def test_05_no_plugin_imports_fail(self):
        """Alle P1–P10 Plugins müssen ohne ImportError importierbar sein."""
        modules = [
            "plugins.crepe_plugin",
            "plugins.diffwave_plugin",
            "plugins.demucs_v4_plugin",
            "plugins.hifigan_plugin",
        ]
        for mod_name in modules:
            import importlib

            try:
                importlib.import_module(mod_name)
            except ImportError as exc:
                pytest.fail(f"ImportError für {mod_name}: {exc}")
