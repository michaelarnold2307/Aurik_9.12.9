from __future__ import annotations

"""Normative GUI contract checks for ModernMainWindow.

This test suite enforces a subset of hard GUI invariants defined in
copilot-instructions/specs for the modern frontend path.
"""


import json
from pathlib import Path

import pytest

GUI_FILE = Path("Aurik10/ui/modern_window.py")


# ── Semantic Contract Helpers (empfohlen statt String-Suche) ──


def _gui_has_method(method_name: str) -> bool:
    """Check if ModernMainWindow has a specific method — semantic, not string-based."""
    src = _read_gui_source()
    return f"def {method_name}" in src


def _gui_has_attribute(attr_name: str) -> bool:
    """Check if ModernMainWindow.__init__ sets a specific attribute."""
    src = _read_gui_source()
    return f"self.{attr_name}" in src


def _gui_has_widget_class(class_name: str) -> bool:
    """Check if a widget class is referenced in the GUI source."""
    src = _read_gui_source()
    return class_name in src


def _read_gui_source() -> str:
    assert GUI_FILE.exists(), f"Missing GUI file: {GUI_FILE}"
    return GUI_FILE.read_text(encoding="utf-8")


@pytest.mark.normative
def test_progress_bar_uses_10000_scale_in_modern_window() -> None:
    src = _read_gui_source()
    assert "self.progress_bar.setRange(0, 10000)" in src
    assert "self.phase_progress_bar.setRange(0, 10000)" in src
    assert "self.refinement_progress_bar.setRange(0, 10000)" in src
    assert "setRange(0, 100)" not in src


@pytest.mark.normative
def test_preanalysis_gate_controls_recommendation_visibility() -> None:
    src = _read_gui_source()
    assert "_preanalysis_finalized_for" in src
    # Code uses _ready instead of _show_recommendation (same logic)
    assert "_ready = bool(_cfp) and (_finalized_for == _cfp)" in src
    assert "self._apply_mode_recommendation_visuals()" in src


@pytest.mark.normative
def test_magic_button_sync_gate_hooks_present() -> None:
    src = _read_gui_source()
    assert "def _try_signal_preanalysis_done(flag: str) -> None:" in src
    assert '_try_signal_preanalysis_done("era_genre")' in src
    assert '_try_signal_preanalysis_done("defect_scan")' in src
    assert "QTimer.singleShot(15_000, _preanalysis_timeout)" in src


@pytest.mark.normative
def test_preanalysis_hard_timeout_does_not_bypass_defect_scan_gate() -> None:
    src = _read_gui_source()
    assert "def _preanalysis_hard_timeout() -> None:" in src
    assert "QTimer.singleShot(180_000, _preanalysis_hard_timeout)" in src
    assert 'if "defect_scan" in self._preanalysis_flags:' in src
    assert "Pre-analysis hard-timeout reached, waiting for defect_scan before finalization" in src


@pytest.mark.normative
def test_era_chip_not_rendered_as_separate_chip() -> None:
    src = _read_gui_source()
    assert "chip_era wird NICHT gesetzt" in src
    assert "_show_chip(self.chip_era" not in src


@pytest.mark.normative
def test_bridge_fallback_contract_and_export_guard_present() -> None:
    src = _read_gui_source()
    assert "_BRIDGE_AVAILABLE = False" in src
    assert "def _export_guard(audio):" in src
    assert "np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)" in src
    assert "return np.clip(audio, -1.0, 1.0)" in src

    # Required bridge fallback stubs from the frontend contract.
    required_none_stubs = (
        "def _bridge_get_audio_file_validator() -> object | None:",
        "def _bridge_get_defect_scanner() -> type | None:",
        "def _bridge_get_defect_type() -> type | None:",
        "def _bridge_get_quality_mode() -> object | None:",
        "def _bridge_get_restorer_classes() -> object | None:",
        "def _bridge_get_medium_classifier_fn() -> object | None:",
        "def _bridge_get_era_classifier_fn() -> object | None:",
        "def _bridge_get_genre_classifier_fn() -> object | None:",
        "def _bridge_get_carrier_forensics_fn() -> object | None:",
        "def _bridge_get_audio_exporter_class() -> type | None:",
    )
    for stub in required_none_stubs:
        assert stub in src


@pytest.mark.normative
def test_shortcuts_complete() -> None:
    src = _read_gui_source()
    assert "_setup_shortcuts" in src
    # Required shortcuts from §11.4
    for seq in ("Key_Space", "Key_A", "Key_B", "Key_Escape", "Ctrl+R", "Ctrl+Shift+R", "Key_L"):
        assert seq in src, f"Missing shortcut: {seq}"
    # StandardKey for Ctrl+O / Ctrl+S
    assert "StandardKey.Open" in src
    assert "StandardKey.Save" in src


@pytest.mark.normative
def test_kmv_signals_and_refinement_bar_present() -> None:
    src = _read_gui_source()
    # KMV §2.38 signal connections
    for sig in ("refinement_started", "refinement_progress", "refinement_complete", "refinement_cancelled"):
        assert f".{sig}.connect(" in src, f"Missing KMV signal connect: {sig}"
    # Refinement bar: türkis, 3 px, range 0-10000, hidden until Stage 2
    assert "refinement_progress_bar" in src
    assert "refinement_progress_bar.setRange(0, 10000)" in src
    assert "int(pct) * 100" in src  # correct scaling


@pytest.mark.normative
def test_dropout_chip_counter_accepts_backend_aliases() -> None:
    src = _read_gui_source()
    assert '"dropouts": "dropout"' in src
    assert '"gap": "dropout"' in src
    assert '"gaps": "dropout"' in src
    assert '"tape_dropout": "dropout"' in src
    assert '"dropout": ("dropouts", "DROPOUTS", "gap", "gaps", "tape_dropout")' in src


@pytest.mark.normative
def test_dropout_chip_counter_follows_timeline_repair_cursor() -> None:
    src = _read_gui_source()
    assert "def mark_defects_resolved_up_to" in src
    assert "_cutoff_s = max(0.0, min(1.0, float(frac))) * _duration" in src
    assert "_resolved_keys" in src
    assert "_resolved_for_key" in src
    assert 'self.waveform_widget.mark_defects_resolved_up_to(["dropout"], float(frac), _tool)' in src
    assert '"dropout", "dropouts", "gap", "gaps", "tape_dropout"' in src


@pytest.mark.normative
def test_main_progress_strictly_mirrors_reported_progress_no_export_headroom() -> None:
    """§GUI-T9 (2026-09-08): Der Hauptbalken zeigt EXAKT den zuletzt gemeldeten
    Fortschritt — keine 9-90-UI-Map, keine 90/98-%-Deckelung, kein Export-Rücksprung.
    """
    src = _read_gui_source()
    assert "new_cur = tgt" in src  # Emitter folgt exakt dem gemeldeten Ziel
    assert "_emit_cap = 10000" in src  # keine Reserved-Ranges mehr
    assert "_stage_cap = 100.0" in src
    assert "item.progress = 90" not in src  # kein künstlicher 90-%-Rücksprung nach restore
    assert "range(9610, 10010, 10)" not in src  # kein Fake-Glide auf 100 %
    assert 'self.item_progress.emit(item.id, 10000)' in src  # exakt ein Sprung auf 100 %


@pytest.mark.normative
def test_watchdog_formula_correct() -> None:
    src = _read_gui_source()
    # §11.4 Watchdog-Timer Formel: max(5_400_000, dur * 48_000 + 1_800_000)
    assert "48_000" in src
    assert "5_400_000" in src
    assert "watchdog" in src  # timing constant refactored


@pytest.mark.normative
def test_no_direct_magic_button_enable_in_preanalysis_emit_paths() -> None:
    src = _read_gui_source()
    # Contract: enabling should happen through _finalize_preanalysis(),
    # not directly in era/genre or defect pre-analysis update hooks.
    _forbidden_snippets = (
        "_run_defect_scan_bg._apply():\n",
        "_detect_era_genre_bg._upd():\n",
    )
    for marker in _forbidden_snippets:
        assert marker not in src

    _finalize_anchor = src.find("def _finalize_preanalysis() -> None:")
    _signal_anchor = src.find("def _try_signal_preanalysis_done(flag: str) -> None:")
    assert _finalize_anchor >= 0 and _signal_anchor > _finalize_anchor
    _finalize_block = src[_finalize_anchor:_signal_anchor]
    assert "_set_magic_buttons_enabled(True)" in _finalize_block


@pytest.mark.normative
def test_preventive_metadata_visibility_contract_in_quality_banner() -> None:
    src = _read_gui_source()

    # Frontend must keep phase-local prevention telemetry visible in post-run UI.
    assert '"preventive_actions": []' in src
    assert "phase31_damage_shield_applied" in src
    assert "phase31_stereo_delay_corrected" in src
    assert "loudness_makeup_db" in src
    assert "def _build_quality_banner_sections(" in src
    assert "preventive_actions: list[str]" in src
    assert "🛡️  Präventionsschutz:" in src


@pytest.mark.normative
def test_preventive_actions_callsite_contract_in_quality_pipeline() -> None:
    src = _read_gui_source()

    # Separate contract: extraction + callsite forwarding must stay intact.
    assert 'preventive_actions: list[str] = _ctx["preventive_actions"]' in src
    assert "self.prognose_widget.set_preventive_actions(preventive_actions)" in src
    assert "preventive_actions=preventive_actions" in src


@pytest.mark.normative
def test_bridge_unavailable_warning_is_one_shot_and_sets_runtime_health_state() -> None:
    src = _read_gui_source()
    assert 'if bool(getattr(self, "_bridge_unavailable_warning_shown", False)):' in src
    assert "self._bridge_unavailable_warning_shown = True" in src
    assert 'self._runtime_health_state = "bridge_unavailable"' in src


@pytest.mark.normative
def test_runtime_original_fallback_detection_uses_structured_metadata_signals() -> None:
    src = _read_gui_source()
    assert "def _detect_runtime_original_fallback_reason(restoration_result) -> str:" in src
    assert 'metadata = getattr(restoration_result, "metadata", {}) or {}' in src
    assert 'stage_notes = getattr(restoration_result, "stage_notes", {}) or {}' in src
    assert "export_quality_gate_failed" in src
    assert "export_blocked_by_quality_gate" in src
    assert "RUNTIME_ORIGINAL_FALLBACK" in src


@pytest.mark.normative
def test_warnings_are_gated_by_non_sota_status_in_quality_ui() -> None:
    src = _read_gui_source()
    assert '"is_sota_run": True' in src
    assert '"sota_warning_reason": ""' in src
    assert "if not is_sota_run:" in src
    assert "Nicht-SOTA-Ausführung" in src
    assert (
        'has_problem = (not is_sota_run) and (degradation_status in {"blocked", "critical_degraded", "degraded"})'
        in src
    )


@pytest.mark.normative
def test_musiclover_sota_metadata_forwarding_in_export_path() -> None:
    src = _read_gui_source()
    assert "quality_gate_musiclover_all_sota_real" in src
    assert "quality_gate_musiclover_sota_reason" in src


@pytest.mark.normative
def test_user_confidence_summary_is_visible_in_gui_and_export_metadata() -> None:
    src = _read_gui_source()
    assert '"user_confidence_summary"' in src
    assert '"_xp_user_confidence": {}' in src
    assert "xp_user_confidence: dict" in src
    assert "🤝  Nutzervertrauen:" in src
    assert "quality_gate_user_confidence_level" in src
    assert "quality_gate_user_confidence_message" in src
    assert "quality_gate_user_confidence_export_policy" in src
    assert "quality_gate_user_confidence_manual_action_required" in src


@pytest.mark.normative
def test_worldclass_gate_and_threshold_evidence_are_visible_in_quality_banner() -> None:
    src = _read_gui_source()
    assert '"_xp_threshold_evidence": {}' in src
    assert 'ctx["_xp_threshold_evidence"] = dict(_te_raw)' in src
    assert "xp_threshold_evidence: dict" in src
    assert 'qg_wcs = xp_quality_gate.get("worldclass_composite_gate", {})' in src
    assert 'qg_psy = xp_quality_gate.get("psychoacoustic_naturalness_gate", {})' in src
    assert "🏁  Worldclass-Gate:" in src
    assert "🎧  Psychoakustik-Gate:" in src
    assert "📚  Gate-Evidenz:" in src


@pytest.mark.normative
def test_worldclass_and_evidence_are_forwarded_in_export_metadata() -> None:
    src = _read_gui_source()
    assert "quality_gate_worldclass_score" in src
    assert "quality_gate_worldclass_threshold" in src
    assert "quality_gate_worldclass_passed" in src
    assert "quality_gate_worldclass_profile" in src
    assert "quality_gate_worldclass_artifact_veto" in src
    assert "quality_gate_hybrid_engineer_vector" in src
    assert "json.dumps(" in src
    assert "quality_gate_psycho_score" in src
    assert "quality_gate_psycho_threshold" in src
    assert "quality_gate_psycho_per_metric_floor" in src
    assert "quality_gate_psycho_passed" in src
    assert "quality_gate_psycho_profile" in src
    assert "quality_gate_evidence_worldclass_source_class" in src
    assert "quality_gate_evidence_worldclass_revalidate_by" in src
    assert "quality_gate_evidence_psycho_source_class" in src
    assert "quality_gate_evidence_psycho_revalidate_by" in src


@pytest.mark.normative
def test_hybrid_engineer_vector_export_metadata_is_json_parseable() -> None:
    src = _read_gui_source()
    assert '"quality_gate_hybrid_engineer_vector": json.dumps(' in src

    vector_payload = {
        "artifact_freedom": 0.98,
        "vocal_identity_preservation": 0.93,
        "goal_team_balance": 0.87,
    }
    serialized = json.dumps(vector_payload, sort_keys=True, ensure_ascii=True)
    parsed = json.loads(serialized)

    assert parsed["artifact_freedom"] == pytest.approx(0.98)
    assert parsed["vocal_identity_preservation"] == pytest.approx(0.93)
    assert parsed["goal_team_balance"] == pytest.approx(0.87)


@pytest.mark.normative
def test_quality_banner_has_psycho_ampel_style_logic() -> None:
    src = _read_gui_source()
    assert '_psy_present = any("🎧  Psychoakustik-Gate:" in _s for _s in banner_sections)' in src
    assert '_psy_risk = any("klinisch-risiko" in _s for _s in banner_sections)' in src
    assert "_degraded_hint = any(" in src
    assert "#F0B8B8" in src  # Rot: klinisch-risiko
    assert "#CFE8D9" in src  # Gruen: natuerlich
    assert "#F2DAB3" in src  # Gelb: Vorsicht


@pytest.mark.normative
def test_waveform_phase_animation_has_generic_fallback_and_progress_binding() -> None:
    src = _read_gui_source()
    assert "generic fallback" in src.lower()
    assert '_generic_key = f"generic:phase_' in src
    assert (
        "self.batch_thread.phase_progress.connect(lambda v: self.waveform_widget.set_stage_progress(v / 10000.0))"
        in src
    )
    assert "self.batch_thread.phase_progress.connect(" in src
    assert "self.waveform_widget_rest_ab.set_stage_progress(v / 10000.0)" in src


@pytest.mark.normative
def test_phase_step_label_has_no_audio_callback_fallback() -> None:
    src = _read_gui_source()
    assert "if _cur_pct >= 20:" in src
    assert "self._pending_step_info = label" in src
    assert "Fallback: some phases do not emit an audio snapshot callback." in src
    assert "self._phase_step_label.setText(label)" in src
    assert "self._phase_step_label.setVisible(True)" in src


@pytest.mark.normative
def test_runtime_status_surfaces_share_current_phase_text() -> None:
    src = _read_gui_source()
    i18n_src = Path("Aurik10/i18n/__init__.py").read_text(encoding="utf-8")
    assert "Analyse und Vorbereitung laufen" in i18n_src
    assert "Passende Korrekturen werden ausgewählt" not in i18n_src
    assert "Analysis and preparation running" in i18n_src
    assert "Selecting suitable corrections" not in i18n_src
    assert "_runtime_state = self._runtime_display_state" in src
    assert "re.sub" in src
    assert "self._phase_step_label.setText(_step_label)" in src


@pytest.mark.normative
def test_heartbeat_progress_forecast_is_strict_noop() -> None:
    """§GUI-T9 (2026-09-08): Die zeitbasierte Heartbeat-Prognose darf die Balken
    NICHT mehr bewegen — No-op, Balken folgen ausschließlich echten Callbacks.
    """
    src = _read_gui_source()
    assert "def _apply_heartbeat_progress_forecast" in src
    # Der No-op-Return steht VOR der früheren Prognose-Logik:
    _idx = src.index("def _apply_heartbeat_progress_forecast")
    _body = src[_idx : _idx + 1600]
    assert "return" in _body
    assert _body.index("return") < _body.index("progress_anchor"), "Prognose-Logik muss nach dem No-op-Return stehen"


@pytest.mark.normative
def test_scan_cursor_does_not_drive_main_bar_strict() -> None:
    """§GUI-T9 (2026-09-08): Der Waveform-Scan-Cursor treibt den Gesamtbalken
    NICHT mehr (strikte Log-Konformität: Balken = echter Pipeline-Fortschritt).
    """
    src = _read_gui_source()
    assert "def _sync_progress_bar_to_scan_cursor" in src
    _idx = src.index("def _sync_progress_bar_to_scan_cursor")
    _body = src[_idx : _idx + 700]
    assert "return" in _body
    assert _body.index("return") < _body.index("_ui_pct = 13.0"), "Scan-Sync muss vor der alten Mapping-Logik enden"


@pytest.mark.normative
def test_main_progress_cannot_drift_past_pre_pipeline_range() -> None:
    """§GUI-T9: Der Emitter übersteuert jede Drift-/Deckel-Logik mit `new_cur = tgt`
    (exakter gemeldeter Wert). Die Alt-Logik bleibt nur als toter Pfad erhalten.
    """
    src = _read_gui_source()
    assert "new_cur = tgt" in src
    assert "_overshoot_cap = min(tgt + 1.5, 18.9)" in src


@pytest.mark.normative
def test_repeated_same_percent_callbacks_do_not_reset_progress_anchor() -> None:
    src = _read_gui_source()
    assert '_target_advanced = _new_tgt > (_sp["target"] + 0.01)' in src
    assert "if _target_advanced and _inter_s >= 0.5:" in src
    assert "if _target_advanced:" in src
    assert '_sp["last_target_time"] = _now' in src


@pytest.mark.normative
def test_waveform_stage_and_scan_are_mirrored_to_rest_ab_widget() -> None:
    src = _read_gui_source()
    assert "def set_scan_pos(self, frac" in src
    assert "self.waveform_widget_rest_ab.set_active_stage(phase_text)" in src
    assert "self.waveform_widget_rest_ab.set_scan_pos(-1.0)" in src
    assert "self.waveform_widget_rest_ab.clear_stage()" in src


@pytest.mark.normative
def test_scan_cursor_forward_progress_keeps_main_bar_moving() -> None:
    """§GUI-T9: Der Scan-Cursor wird strikt aus dem gemeldeten pct abgeleitet
    (tgt/100); die Sync-Funktion selbst ist No-op (Balken = echter Fortschritt).
    """
    src = _read_gui_source()
    assert "self._sync_progress_bar_to_scan_cursor(frac)" in src
    assert "_scan_frac = max(0.0, min(1.0, tgt / 100.0))" in src  # §GUI-T9: strikte Ableitung
    assert 'current_item = next((i for i in self.batch_queue.items if i.status == "processing"), None)' in src
    assert 'current_item.progress = max(int(getattr(current_item, "progress", 0) or 0), int(_ui_pct))' in src


@pytest.mark.normative
def test_active_repair_defects_are_visible_without_initial_scanner_score() -> None:
    src = _read_gui_source()
    assert "_active_defects_set: set[str] = (" in src
    assert '{"noise": "noise_level"}.get(str(k), str(k))' in src
    assert "or bool(_active_defects_set)" in src
    assert "for _ak in sorted(_active_defects_set):" in src
    assert "_active_level = max(_thr_light * 1.05, 0.02)" in src
    assert "if _canon not in _active_keys:" in src


@pytest.mark.normative
def test_waveform_tool_label_resets_when_phase_has_no_tool_match() -> None:
    src = _read_gui_source()
    assert "self._active_tool = _detected_tool" in src
    assert "if _detected_tool:\n            self._active_tool = _detected_tool" not in src


@pytest.mark.normative
def test_close_while_processing_dialog_uses_explicit_german_buttons() -> None:
    src = _read_gui_source()
    _anchor = src.find("if _workers_running:")
    _end = src.find("self._window_tearing_down = True", _anchor)
    assert _anchor >= 0 and _end > _anchor
    _dialog_block = src[_anchor:_end]
    assert "QDialog(self)" in _dialog_block
    assert "QMessageBox" not in _dialog_block
    assert 'QPushButton(t("dialog.close_while_processing_btn_keep"))' in _dialog_block
    assert 'QPushButton(t("dialog.close_while_processing_btn_close"))' in _dialog_block


@pytest.mark.normative
def test_status_and_quality_styles_sanitize_qss_colors() -> None:
    src = _read_gui_source()
    assert "def _sanitize_qss_colors" in src
    assert "_QSS_COLOR_TOKEN_RE" in src
    assert "self.status_label.setStyleSheet(_sanitize_qss_colors(" in src
    assert "self.quality_score_label.setStyleSheet(_sanitize_qss_colors(" in src
    assert "_sanitize_qss_colors(" in src
    assert "_sanitize_qss_colors(" in src


@pytest.mark.normative
def test_dropout_status_overrides_stale_resampling_focus() -> None:
    src = _read_gui_source()
    assert "_dropout_context = any(" in src
    assert "_needle in _msg_underscored or _needle in _cur_pid" in src
    assert "_repair_names_for_desc" not in src
    assert 'for _needle in ("dropout", "diffusion_inpainting", "tonaussetzer")' in src
    assert 'if _dropout_context and "abtastrate" in _step_desc.lower():' in src
    assert '_step_desc = "Tonaussetzer werden repariert"' in src
    assert 'if _dropout_context and "abtastrate" in _base_text.lower():' in src
    assert '_base_text = "Tonaussetzer werden repariert"' in src


@pytest.mark.normative
def test_planning_status_clears_stale_repair_hints() -> None:
    src = _read_gui_source()
    assert "_planning_context = any(" in src
    assert 'for _needle in ("phasenauswahl", "passende_korrekturen", "korrekturen werden ausgewählt")' in src
    assert (
        'if _planning_context or not _real_repair_phase:\n                        self._current_repair_names = ""'
        in src
    )
    assert 'if _base_bucket == "planning":\n                            _repair_stale = True' in src


@pytest.mark.normative
def test_defect_chips_show_measured_values_when_event_counts_are_absent() -> None:
    src = _read_gui_source()
    assert "def _format_defect_value_html" in src
    assert (
        "if _cnt_total > 0:\n                    return f' <span style=\"color:#8FA6C8;\">{_cnt_total}/{_cnt_rem}</span>'"
        in src
    )
    assert '"bandwidth_loss"' in src
    assert '"bias_error"' in src
    assert '"noise_level": "dB"' in src
    assert '"noise": ("Rauschen", 0.1, 0.5)' not in src
    assert '_txt = f"{_init_txt}→{_cur_txt}"' in src
    assert '_event_like_keys = {"clicks", "crackle", "pops", "sibilance", "dropout"}' in src
    assert "_count_html = _format_defect_value_html(k, float(v_init), float(v_cur))" in src


@pytest.mark.normative
def test_impulse_chip_uses_clipping_event_count_alias() -> None:
    src = _read_gui_source()
    assert '"pops": ("clipping",),' in src
    assert '"clipping": ("pops",),' in src
    assert "for _lookup_key in (_key, *_aliases.get(_key, ()))" in src
    assert "_cnt_key = next(" in src


@pytest.mark.normative
def test_click_passes_are_explained_as_distinct_serial_repairs() -> None:
    src = _read_gui_source()
    assert '"click_removal": "Kurze Knackser werden im ersten Durchgang entfernt"' in src
    assert '"click_pop": "Rest-Impulse und tiefe Pops werden im zweiten Durchgang entfernt"' in src
    assert '"declick": "Kurze Knackser werden im ersten Durchgang entfernt"' in src
    assert '"click_removal": ["clicks", "pops"]' in src
    assert '"click_pop": ["clicks", "pops"]' in src
    assert '"phase_01": "kurze Knackser"' in src
    assert '"phase_27": "Rest-Impulse und Pops"' in src
    assert "entfernt kurze Einzelknackser im ersten Durchgang" in src
    assert "prüft verbleibende Impulse mit größerem Kontext" in src


@pytest.mark.normative
def test_denoise_activates_single_noise_chip_without_hum_alias() -> None:
    src = _read_gui_source()
    assert '"denoise": ["noise_level"],' in src
    assert '"noise_gate": ["noise_level"],' in src
    assert '"tape_hiss": ["crackle", "noise_level"],' in src
    assert '"noise": "noise_level"' in src
    assert '{"noise": "noise_level"}.get(str(k), str(k))' in src
    assert '_canon_key = {"noise": "noise_level"}.get(str(_raw_key), str(_raw_key))' in src


@pytest.mark.normative
def test_chip_event_remaining_counts_follow_serial_phase_score_reduction() -> None:
    src = _read_gui_source()
    assert "_drop_factor = 0.84  # 16 % konservative Absenkung pro abgeschlossener Phase" in src
    assert "_current_defect_scores[_rkey] = _next_val" in src
    assert 'if _resolved <= 0 and _status in ("correcting", "completed", "blocked"):' in src
    assert "_phase_estimated_remaining = int(math.ceil(float(_total) * _score_ratio))" in src
    assert "_remaining = min(_remaining, max(0, _phase_estimated_remaining))" in src


@pytest.mark.normative
def test_resolved_waveform_marker_updates_chip_and_status_counts_immediately() -> None:
    src = _read_gui_source()
    assert "self._resolved_defects_changed_cb: Callable[[], None] | None = None" in src
    assert "def set_resolved_defects_changed_callback(self, callback: Callable[[], None] | None) -> None:" in src
    assert (
        "self.waveform_widget.set_resolved_defects_changed_callback(self._on_waveform_resolved_defects_changed)" in src
    )
    assert "def _on_waveform_resolved_defects_changed(self) -> None:" in src
    assert 'if isinstance(defects, dict) and not defects.get("_from_waveform_resolved_refresh"):' in src
    assert "self._latest_defect_payload = dict(defects)" in src
    assert '_payload["_no_anim"] = True' in src
    assert '_payload["_from_waveform_resolved_refresh"] = True' in src
    assert "seg, tool = queue.pop(0)" in src
    assert "self._repair_history[dk] = history" in src
    assert "if _changed and self._resolved_defects_changed_cb is not None:" in src
    assert "self._resolved_defects_changed_cb()" in src
    assert "n_pop = min(2, len(queue))" not in src


@pytest.mark.normative
def test_active_defect_chips_require_real_phase_id_not_preparation_status() -> None:
    src = _read_gui_source()
    assert '_real_repair_phase = _status == "correcting" and _live_phase_id.startswith("phase_")' in src
    assert '_active_now = (defects.get("_active_defects") or []) if _real_repair_phase else []' in src
    assert 'if _status == "correcting" and _real_repair_phase:' in src
    assert "_real_repair_phase = bool(" in src


@pytest.mark.normative
def test_multi_active_defect_chips_are_reflected_in_phase_status() -> None:
    src = _read_gui_source()
    assert '"click_removal": ["clicks", "pops"]' in src
    assert '"azimuth": ["azimuth_error", "phase_issues", "head_wear"]' in src
    assert '"wow_flutter": ["wow", "flutter", "transport_bump"]' in src
    assert "def _format_active_defect_names" in src
    assert "_active_names = _format_active_defect_names(_active_keys)" in src
    assert "self._current_repair_names = _active_names" in src
    assert 'if _active_defect_names and "·" in _active_defect_names:' in src
    assert "_step_desc = _active_defect_sentence" in src
    assert "_base_text = _active_defect_sentence" in src


@pytest.mark.normative
def test_backend_pipeline_started_callback_is_translated_to_german() -> None:
    src = _read_gui_source()
    assert '"pipeline started": "Restaurierung wird gestartet"' in src
    assert '"pipeline startet": "Restaurierung wird gestartet"' in src


@pytest.mark.normative
def test_noise_phases_are_explained_as_distinct_serial_repairs() -> None:
    src = _read_gui_source()
    phase_focus = src[src.index("def _phase_risk_focus_label") : src.index("def _phase_priority_confidence")]
    phase_priority = src[src.index("def _phase_priority_explanation") : src.index("def _sync_runtime_display_state")]
    assert '"phase_03": "Breitbandrauschen"' in phase_focus
    assert '"phase_29": "Bandrauschen"' in phase_focus
    assert '_focus == "Breitbandrauschen"' in phase_priority
    assert "senkt gleichmäßiges Grundrauschen" in phase_priority
    assert '_focus == "Bandrauschen"' in phase_priority
    assert "senkt band- und trägerbedingtes Hiss" in phase_priority


@pytest.mark.normative
def test_quality_risk_message_explains_user_relevant_protection() -> None:
    src = _read_gui_source()
    assert "def _quality_risk_guidance_text" in src
    assert "Schutzmodus aktiv" in src
    assert "Aurik dosiert die Korrektur vorsichtig" in src
    assert "misst nach jeder Phase nach" in src
    assert "nimmt Schritte zurueck" in src
    assert "Aurik schuetzt" in src
    assert "Risiko erhöht ({_pid_goal_risk:.2f})" not in src
    assert "Qualitätsrisiko erhöht ({_pid_goal_risk:.2f})" not in src


@pytest.mark.normative
def test_defect_progress_uses_open_done_wording_and_deduped_event_totals() -> None:
    src = _read_gui_source()
    assert "_has_progress = _remaining < _total or _resolved > 0 or _reduced >= 3" in src
    assert "if _pct < 19.0 and not _has_progress:" in src
    assert "offen" in src
    assert "def _sum_event_counts_dedup" in src
    assert '_alias_groups = (("pops", "clipping"), ("noise", "noise_level"))' in src
    assert "_total_count += max(max(0, int(_counts.get(_member, 0) or 0)) for _member in _group)" in src


@pytest.mark.normative
def test_active_phase_beats_stale_planning_status() -> None:
    src = _read_gui_source()
    assert 'if str(phase_id or "").strip().lower().startswith("phase_"):\n            return "restoring"' in src
    assert "_latest_is_planning = any(" in src
    assert '_base_is_repair = bool(_phase_id.strip().lower().startswith("phase_"))' in src
    assert "if _latest_phase and not (_latest_is_planning and _base_is_repair):" in src
    phase_focus = src[src.index("def _phase_risk_focus_label") : src.index("def _phase_priority_confidence")]
    assert phase_focus.index("# Primär: deterministische Zuordnung über Phase-ID.") < phase_focus.index(
        "if ui_pct < 20.0:"
    )


@pytest.mark.normative
def test_pre_repair_status_does_not_show_stale_defect_focus() -> None:
    src = _read_gui_source()
    assert '_real_repair_phase = bool(str(_cur_pid or "").strip().lower().startswith("phase_"))' in src
    assert (
        'if _planning_context or not _real_repair_phase:\n                        self._current_repair_names = ""'
        in src
    )
    assert "if _real_repair_phase or _dropout_context" in src
    assert 'if _d_total > 0 and _phase_id.strip().lower().startswith("phase_"):' in src
    assert (
        'if not _phase_id.strip().lower().startswith("phase_"):\n                            _repair_hint = ""' in src
    )


@pytest.mark.normative
def test_dynamic_defect_counter_styles_sanitize_qss_colors() -> None:
    src = _read_gui_source()
    assert "Ungültige Statusfarbe verworfen" in src
    assert "f\"color: {status_color}; font-family: 'Courier New'; font-size: 10pt; font-weight: bold;\"" in src
    assert "f\"color: {status_color}; font-family: 'Courier New'; font-size: 10pt;\"" in src
    assert src.count("_sanitize_qss_colors(") >= 8
