"""Experience Runtime Module — Hörer-Komfort & Ermüdungs-Tracking (§v10.706).

Berechnet Hörermüdung, Freude und Frisson basierend auf Wiedergabe-Dauer
und Audio-Segment-Anzahl. Exponentielle Glättung für stabile Werte.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ExperienceRuntime:
    """Listener experience tracking (§v10.706).

    Berechnet Hörermüdung, Freude und Frisson basierend auf Wiedergabe-Dauer
    und Audio-Segment-Anzahl. Exponentielle Glättung für stabile Werte.
    """

    def __init__(self) -> None:
        self.fatigue_index = 0.0
        self.joy_index = 0.0
        self.frisson_index = 0.0
        self._total_segments: int = 0
        self._total_duration_s: float = 0.0

    def update(self, audio_segment: int = 0, duration_s: float = 0.0) -> None:
        """Aktualisiert Hörer-Erfahrungs-Metriken mit exponentieller Glättung.

        Müdigkeit steigt logarithmisch mit der Dauer (α=0.3).
        Freude wächst linear mit Segmenten (α=0.2), gesättigt bei 1.0.
        Frisson wird als Differenz modelliert (Freude − Müdigkeit, clamped [−1, +1]).

        Args:
            audio_segment: Anzahl der verarbeiteten Audio-Segmente
            duration_s: Wiedergabe-Dauer in Sekunden
        """
        self._total_segments += max(audio_segment, 0)
        self._total_duration_s += max(duration_s, 0.0)

        # Exponentielle Glättung (α=0.3 für fatigue, α=0.2 für joy)
        alpha_fatigue = 0.3
        alpha_joy = 0.2

        # Müdigkeit: logarithmisch mit Dauer (simuliert Ermüdungskurve)
        new_fatigue = min(self._total_duration_s / 1800.0, 1.0)  # 30 Min → Sättigung
        self.fatigue_index = alpha_fatigue * new_fatigue + (1.0 - alpha_fatigue) * self.fatigue_index

        # Freude: linear mit Segmenten, gesättigt bei 1.0
        new_joy = min(self._total_segments / 100.0, 1.0)
        self.joy_index = alpha_joy * new_joy + (1.0 - alpha_joy) * self.joy_index

        # Frisson: Differenz zwischen Freude und Müdigkeit (clamped [−1, +1])
        self.frisson_index = float(np.clip(self.joy_index - self.fatigue_index, -1.0, 1.0))


_experience_runtime_instance = ExperienceRuntime()


def get_experience_runtime() -> ExperienceRuntime:
    """Get or create experience runtime singleton."""
    return _experience_runtime_instance


__all__ = ["get_experience_runtime", "ExperienceRuntime"]
