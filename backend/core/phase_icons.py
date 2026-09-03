"""
Phase Icon Registry — Unicode Icons für alle 68 Phasen + System-Stages

Jede Phase und jeder System-Prozess bekommt ein intuitives Icon,
das in Log-Ausgaben vorangestellt wird.

Kategorien:
  🔍 Analyse       🧹 Reinigung      🎚️ Dynamik
  🎤 Gesang        🔊 Stereo         🎛️ EQ/Frequenz
  🔧 Reparatur     ⚡ Speed/Pitch    🛡️ Sicherheit
  🎸 Instrument    💿 Export/Master  📊 Qualität
  🔗 Assembly      🔌 Interface      🤖 AI/ML
  🎵 Musik         ⚠️ Warnung        ❌ Fehler

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

# ── Kategorie-Icons ─────────────────────────────────────────────────────

CATEGORY_ICONS: dict[str, str] = {
    "analyse": "🔍",
    "reinigung": "🧹",
    "dynamik": "🎚️",
    "gesang": "🎤",
    "stereo": "🔊",
    "frequenz": "🎛️",
    "reparatur": "🔧",
    "speed": "⚡",
    "sicherheit": "🛡️",
    "instrument": "🎸",
    "export": "💿",
    "qualitaet": "📊",
    "assembly": "🔗",
    "interface": "🔌",
    "ai": "🤖",
    "musik": "🎵",
    "warnung": "⚠️",
    "fehler": "❌",
    "erfolg": "✅",
    "start": "🚀",
    "ende": "🏁",
    "cd_noise": "💿",
    "parallel": "⚡⚡",
}

# ── Phasen-Icon-Mapping (alle 68 Phasen) ────────────────────────────────

PHASE_ICONS: dict[str, str] = {
    # ── Analyse / Forensik / Vorfilter ──
    "phase_01_click_removal": "🔍🩹",  # Klick-Erkennung & -Entfernung
    "phase_02_hum_removal": "🔍〰️",  # Brumm-Erkennung & -Entfernung
    "phase_correction": "🔍",  # Phasen-Korrektur (generisch)
    "phase_interface": "🔌",  # Phasen-Schnittstelle
    "phase_glue_stage": "🔗",  # Glue-Kompression (Final)
    # ── Reinigung / Denoising ──
    "phase_03_denoise": "🧹🌊",  # Breitband-Entrauschung (ML+DSP)
    "phase_05_rumble_filter": "🧹📉",  # Tieffrequenz-Rumpelfilter
    "phase_09_crackle_removal": "🧹⚡",  # Vinyl-Knistern-Entfernung
    "phase_23_spectral_repair": "🧹🔧",  # Spektrale Inpainting-Reparatur; §4.7c POCS n_iter=2–5 vor PGHI
    "phase_27_click_pop_removal": "🧹🎯",  # Präzise Knackser-Entfernung
    "phase_28_surface_noise_profiling": "🔍🧹",  # Oberflächenrauschen-Profil & NR
    "phase_29_tape_hiss_reduction": "🧹📼",  # Bandrauschen-Unterdrückung
    "phase_30_dc_offset_removal": "🧹📏",  # DC-Offset-Entfernung
    "phase_50_spectral_repair": "🧹🔧",  # Spektrale Nachreparatur
    "phase_57_print_through_reduction": "🧹🔊",  # Kopiereffekt (Print-Through)
    "phase_59_modulation_noise_reduction": "🧹📡",  # Modulationsrauschen
    "phase_62_crosstalk_cancellation": "🧹🔀",  # Übersprech-Kompensation
    "phase_63_intermodulation_reduction": "🧹📊",  # Intermodulations-Verzerrung
    "phase_66_stem_targeted_nr": "🧹🎤",  # Stem-basierte Rauschunterdrückung
    # ── EQ / Frequenz ──
    "phase_04_eq_correction": "🎛️📐",  # Tonlagen-Korrektur-EQ
    "phase_06_frequency_restoration": "🎛️🔧",  # Frequenzgang-Wiederherstellung
    "phase_16_final_eq": "🎛️✨",  # Abschluss-EQ
    "phase_37_bass_enhancement": "🎛️🔊",  # Bass-Anhebung
    "phase_38_presence_boost": "🎛️🎤",  # Präsenz-Anhebung
    "phase_39_air_band_enhancement": "🎛️💨",  # Luftband-Anhebung
    "phase_56_spectral_band_gap_repair": "🎛️🔧",  # Spektrallücken-Reparatur
    # ── Harmonik / Sättigung ──
    "phase_07_harmonic_restoration": "〰️🔧",  # Obertonspektrum-Rekonstruktion
    "phase_22_tape_saturation": "📼✨",  # Band-Sättigung (analog emuliert)
    # ── Dynamik ──
    "phase_08_transient_preservation": "🎚️⚡",  # Transienten-Erhalt
    "phase_10_compression": "🎚️📉",  # Dynamik-Kompression
    "phase_11_limiting": "🎚️🛡️",  # Peak-Limiting
    "phase_18_noise_gate": "🎚️🔇",  # Störschwelle/Noise Gate
    "phase_24_dropout_repair": "🎚️🩹",  # Aussetzer-Reparatur
    "phase_26_dynamic_range_expansion": "🎚️📈",  # Dynamik-Erweiterung
    "phase_35_multiband_compression": "🎚️🎛️",  # Multiband-Kompression
    "phase_36_transient_shaper": "🎚️🔨",  # Transienten-Design
    "phase_47_truepeak_limiter": "🎚️📏",  # True-Peak-Begrenzer (ITU-R)
    "phase_54_transparent_dynamics": "🎚️👻",  # Transparente Dynamik
    # ── Stereo / Räumlichkeit ──
    "phase_13_stereo_enhancement": "🔊✨",  # Stereo-Anreicherung
    "phase_15_stereo_balance": "🔊⚖️",  # Stereo-Balance
    "phase_25_azimuth_correction": "🔊📐",  # Azimut-Korrektur
    "phase_32_mono_to_stereo": "🔊🔀",  # Mono→Stereo
    "phase_33_stereo_width_limiter": "🔊🛡️",  # Stereo-Breiten-Begrenzer
    "phase_34_mid_side_processing": "🔊🎯",  # M/S-Prozessor
    "phase_46_spatial_enhancement": "🔊🏛️",  # Räumlichkeits-Erweiterung
    "phase_48_stereo_width_enhancer": "🔊↔️",  # Stereo-Breiten-Enhancer
    # ── Speed / Pitch ──
    "phase_12_wow_flutter_fix": "⚡🎢",  # Gleichlauf-Korrektur (Wow/Flutter)
    "phase_31_speed_pitch_correction": "⚡🎯",  # Geschwindigkeits- & Pitch-Korrektur
    # ── Gesang ──
    "phase_19_de_esser": "🎤✨",  # De-Esser (Zischlaut-Reduktion)
    "phase_42_vocal_enhancement": "🎤⭐",  # Gesangs-Optimierung
    "phase_43_ml_deesser": "🎤🤖",  # ML-De-Esser (adaptiv)
    "phase_58_lyrics_guided_enhancement": "🎤📝",  # Textgeführte Phonem-Optimierung
    "phase_65_vocal_naturalness_restoration": "🎤🌿",  # Gesangs-Natürlichkeit
    # ── Instrumente ──
    "phase_44_guitar_enhancement": "🎸✨",  # Gitarren-Enhancement
    "phase_45_brass_enhancement": "🎺✨",  # Bläser-Enhancement
    "phase_51_drums_enhancement": "🥁✨",  # Schlagzeug-Enhancement
    "phase_52_piano_restoration": "🎹🔧",  # Klavier-Restaurierung
    # ── Effekte / Raum ──
    "phase_14_phase_correction": "🔄🔊",  # Phasenlage-Korrektur (Stereo)
    "phase_20_reverb_reduction": "🏛️📉",  # Hall-Reduktion
    "phase_21_exciter": "✨⚡",  # Exciter (Oberton-Synthese)
    "phase_49_advanced_dereverb": "🏛️🔧",  # Erweiterte Hall-Entfernung (WPE)
    # ── Mastering / Export ──
    "phase_17_mastering_polish": "💿✨",  # Mastering-Politur
    "phase_40_loudness_normalization": "💿📏",  # Lautheits-Normalisierung (LUFS)
    "phase_41_output_format_optimization": "💿📋",  # Ausgabeformat-Optimierung
    # ── AI / ML ──
    "phase_53_semantic_audio": "🤖📊",  # Semantische Audioanalyse
    "phase_55_diffusion_inpainting": "🤖🎨",  # Diffusions-basiertes Inpainting
    # ── Spezial-Reparatur (Vinyl/Tape) ──
    "phase_60_inner_groove_distortion_repair": "🔧💿",  # Innenrillen-Verzerrung
    "phase_61_groove_echo_cancellation": "🔧🔄",  # Rillenecho-Kompensation
    "phase_64_tape_splice_repair": "🔧📼",  # Band-Klebestelle
}

# ── System-Prozess-Icons ────────────────────────────────────────────────

SYSTEM_ICONS: dict[str, str] = {
    "restoration_start": "🚀",
    "restoration_end": "🏁",
    "song_calibration": "📊",
    "defect_scan": "🔍",
    "era_classification": "🕰️",
    "medium_detection": "💿",
    "export_pipeline": "💿",
    "cd_noise_profile": "💿",
    "dither": "💿",
    "quality_report": "📊",
    "mushra_score": "📊",
    "abx_test": "📊",
    "vocal_repair": "🎤🔧",
    "stem_separation": "🎤",
    "stcg_pre": "🔊🛡️",
    "stcg_post": "🔊🛡️",
    "stcg_chunk": "🔊🛡️",
    "lag_probe": "🔊",
    "graceful_stop": "🛡️",
    "guard_bypass": "⚠️",
    "guard_enforce": "🛡️",
    "feedback_chain": "🔗",
    "parallel_group": "⚡⚡",
    "error_recovery": "⚠️🔧",
}

# ── Deutsche Phasennamen ────────────────────────────────────────────────

PHASE_NAMES_DE: dict[str, str] = {
    # Analyse / Vorfilter
    "phase_01_click_removal": "Klick-Erkennung",
    "phase_02_hum_removal": "Brumm-Entfernung",
    "phase_correction": "Phasen-Korrektur",
    "phase_interface": "Phasen-Schnittstelle",
    "phase_glue_stage": "Glue-Stage",
    # Reinigung / Denoising
    "phase_03_denoise": "Entrauschung",
    "phase_05_rumble_filter": "Rumpelfilter",
    "phase_09_crackle_removal": "Knistern-Entfernung",
    "phase_23_spectral_repair": "Spektrale Reparatur",
    "phase_27_click_pop_removal": "Knackser-Entfernung",
    "phase_28_surface_noise_profiling": "Oberflächenrauschen",
    "phase_29_tape_hiss_reduction": "Band-Rausch-Unterdrückung",
    "phase_30_dc_offset_removal": "DC-Offset-Entfernung",
    "phase_50_spectral_repair": "Spektrale Nachreparatur",
    "phase_57_print_through_reduction": "Kopiereffekt-Reduktion",
    "phase_59_modulation_noise_reduction": "Modulationsrauschen",
    "phase_62_crosstalk_cancellation": "Übersprech-Kompensation",
    "phase_63_intermodulation_reduction": "Intermodulations-Reduktion",
    "phase_66_stem_targeted_nr": "Stem-Rauschunterdrückung",
    # EQ / Frequenz
    "phase_04_eq_correction": "EQ-Korrektur",
    "phase_06_frequency_restoration": "Frequenz-Wiederherstellung",
    "phase_16_final_eq": "Abschluss-EQ",
    "phase_37_bass_enhancement": "Bass-Anhebung",
    "phase_38_presence_boost": "Präsenz-Anhebung",
    "phase_39_air_band_enhancement": "Luftband-Anhebung",
    "phase_56_spectral_band_gap_repair": "Spektrallücken-Reparatur",
    # Harmonik / Sättigung
    "phase_07_harmonic_restoration": "Harmonische Wiederherstellung",
    "phase_22_tape_saturation": "Band-Sättigung",
    # Dynamik
    "phase_08_transient_preservation": "Transienten-Erhalt",
    "phase_10_compression": "Kompression",
    "phase_11_limiting": "Limiting",
    "phase_18_noise_gate": "Noise Gate",
    "phase_24_dropout_repair": "Aussetzer-Reparatur",
    "phase_26_dynamic_range_expansion": "Dynamik-Erweiterung",
    "phase_35_multiband_compression": "Multiband-Kompression",
    "phase_36_transient_shaper": "Transienten-Shaper",
    "phase_47_truepeak_limiter": "True-Peak-Limiter",
    "phase_54_transparent_dynamics": "Transparente Dynamik",
    # Stereo / Räumlichkeit
    "phase_13_stereo_enhancement": "Stereo-Anreicherung",
    "phase_15_stereo_balance": "Stereo-Balance",
    "phase_25_azimuth_correction": "Azimut-Korrektur",
    "phase_32_mono_to_stereo": "Mono-zu-Stereo",
    "phase_33_stereo_width_limiter": "Stereo-Breiten-Begrenzer",
    "phase_34_mid_side_processing": "M/S-Prozessor",
    "phase_46_spatial_enhancement": "Räumlichkeits-Erweiterung",
    "phase_48_stereo_width_enhancer": "Stereo-Breiten-Enhancer",
    # Speed / Pitch
    "phase_12_wow_flutter_fix": "Gleichlauf-Korrektur",
    "phase_31_speed_pitch_correction": "Geschwindigkeits-Korrektur",
    # Gesang
    "phase_19_de_esser": "De-Esser",
    "phase_42_vocal_enhancement": "Gesangs-Verbesserung",
    "phase_43_ml_deesser": "ML-De-Esser",
    "phase_58_lyrics_guided_enhancement": "Textgeführte Optimierung",
    "phase_65_vocal_naturalness_restoration": "Gesangs-Natürlichkeit",
    # Instrumente
    "phase_44_guitar_enhancement": "Gitarren-Enhancement",
    "phase_45_brass_enhancement": "Bläser-Enhancement",
    "phase_51_drums_enhancement": "Schlagzeug-Enhancement",
    "phase_52_piano_restoration": "Klavier-Restaurierung",
    # Effekte / Raum
    "phase_14_phase_correction": "Phasenlage-Korrektur",
    "phase_20_reverb_reduction": "Hall-Reduktion",
    "phase_21_exciter": "Exciter",
    "phase_49_advanced_dereverb": "Erweiterte Hall-Entfernung",
    # Mastering / Export
    "phase_17_mastering_polish": "Mastering-Politur",
    "phase_40_loudness_normalization": "Lautheits-Normalisierung",
    "phase_41_output_format_optimization": "Ausgabeformat-Optimierung",
    # AI / ML
    "phase_53_semantic_audio": "Semantische Audioanalyse",
    "phase_55_diffusion_inpainting": "Diffusions-Inpainting",
    # Spezial-Reparatur (Vinyl/Tape)
    "phase_60_inner_groove_distortion_repair": "Innenrillen-Verzerrung",
    "phase_61_groove_echo_cancellation": "Rillenecho-Kompensation",
    "phase_64_tape_splice_repair": "Band-Klebestelle",
}


# ── API ──────────────────────────────────────────────────────────────────


def phase_icon(phase_id: str) -> str:
    """Gibt das Icon für eine Phase zurück.

    Args:
        phase_id: z.B. 'phase_42_vocal_enhancement' oder 'phase_03_denoise'

    Returns:
        Icon-String oder '🎵' als Fallback.
    """
    # Exakter Match
    if phase_id in PHASE_ICONS:
        return PHASE_ICONS[phase_id]
    # Präfix-Match (z.B. 'phase_42' matcht 'phase_42_vocal_enhancement')
    for key, icon in PHASE_ICONS.items():
        if phase_id.startswith(key.split("_")[0] + "_" + key.split("_")[1]) and len(key.split("_")) >= 3:
            if phase_id.replace("_", "").startswith(key.split("_")[0] + key.split("_")[1]):
                return icon
    return "🎵"


def phase_name_de(phase_id: str) -> str:
    """Gibt den deutschen Anzeigenamen einer Phase zurück."""
    if phase_id in PHASE_NAMES_DE:
        return PHASE_NAMES_DE[phase_id]
    # Präfix-Match
    for key, name in PHASE_NAMES_DE.items():
        base = "_".join(key.split("_")[:2])
        if phase_id.startswith(base):
            return name
    return phase_id.replace("_", " ").title()


def phase_display(phase_id: str) -> str:
    """Gibt Icon + deutschen Namen für Log-Ausgaben zurück."""
    icon = phase_icon(phase_id)
    name = phase_name_de(phase_id)
    return f"{icon} {name}"


def system_icon(process: str) -> str:
    """Gibt das Icon für einen System-Prozess zurück."""
    return SYSTEM_ICONS.get(process, "⚙️")


def icon_log(phase_id: str, message: str, *, is_system: bool = False) -> str:
    """Formatiert eine Log-Nachricht mit vorangestelltem Icon.

    Args:
        phase_id: Phase-ID oder System-Prozess-Name.
        message: Die eigentliche Log-Nachricht.
        is_system: True für System-Prozesse, False für Phasen.

    Returns:
        Formatierte Nachricht: '🔍 message'
    """
    icon = system_icon(phase_id) if is_system else phase_icon(phase_id)
    return f"{icon} {message}"
