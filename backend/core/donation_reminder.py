"""DonationReminder — §GRATITUDE

Zeigt nach jeder erfolgreichen Restaurierung eine freundliche
Spenden-Erinnerung mit PayPal-Link.
"""

from __future__ import annotations

import json
import logging
import time
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

PAYPAL_URL = "https://www.paypal.com/donate?business=michael.arnold2307@gmail.com&currency_code=EUR"
DONATION_URL = PAYPAL_URL  # Backward-kompatibler Alias (backend/api/bridge.py)
PAYPAL_EMAIL = "michael.arnold2307@gmail.com"
PAYPAL_FALLBACK = "https://paypal.me/michaelarnold2307"

# Rate-limit: maximal alle 24 Stunden (pro Session oder via Disk-Stamp)
_RATE_LIMIT_SECONDS = 86400  # 24 Stunden
_STAMP_FILE = Path(__file__).parent.parent / "logs" / ".donation_last_shown"

# §v10.306 Reziprozitätsprinzip (Cialdini): erst ab der 3. erfolgreichen Nutzung fragen
_MIN_USES_BEFORE_REMINDER = 3
_USAGE_COUNT_FILE = Path(__file__).parent.parent / "logs" / ".donation_usage_count"

# §v10.306 Trust/Grace: nach einer Spende wird die Erinnerung für eine Weile ausgesetzt
_GRACE_FILE = Path(__file__).parent.parent / "logs" / ".donation_grace.json"
_BASE_GRACE_DAYS = 30.0
_GRACE_DAYS_PER_EUR = 6.0
_MAX_GRACE_DAYS = 365.0

_MESSAGES = [
    "🎵 Dein Song wurde erfolgreich restauriert!",
    "",
    "Aurik ist das Ergebnis tausender Stunden Entwicklungsarbeit —",
    "kostenlos, werbefrei und mit Weltspitze-Qualität.",
    "",
    "Wenn Dir Aurik geholfen hat, freue ich mich über Deine Unterstützung:",
    f"👉 {PAYPAL_URL}",
    "",
    "Jeder Betrag hilft, Aurik weiter zu verbessern. Danke! ❤️",
    "",
    "— Michael (Aurik-Entwickler)",
]


def show_reminder(quality_score: float = 0.0) -> str:
    """Zeigt Spenden-Erinnerung mit personalisiertem Qualitäts-Hinweis."""

    if quality_score > 0.8:
        personal = "🌟 Hervorragende Restaurierung! Aurik hat hier ganze Arbeit geleistet."
    elif quality_score > 0.5:
        personal = "✨ Gute Restaurierung! Aurik konnte den Klang spürbar verbessern."
    else:
        personal = "🎧 Dein Song wurde restauriert. Aurik hat sein Bestes gegeben."

    lines = [personal] + _MESSAGES

    message = "\n".join(lines)
    logger.info(message)
    return message


def open_donation_link() -> bool:
    """Öffnet den Spenden-Link im Browser. Fällt auf PAYPAL_FALLBACK zurück, wenn
    `webbrowser.open()` den Fehlschlag per Rückgabewert meldet (nicht nur per Exception —
    fehlender Browser-Handler wirft i.d.R. keine Exception, sondern liefert False).
    """
    try:
        if webbrowser.open(PAYPAL_URL):
            logger.debug("Donation link opened: %s", PAYPAL_URL)
            return True
    except Exception:
        logger.debug("donation_reminder: primary URL open fehlgeschlagen, trying Ersatzpfad", exc_info=True)
    try:
        return bool(webbrowser.open(PAYPAL_FALLBACK))
    except Exception:
        return False


def validate_donation_configuration() -> dict:
    """Prüft die Spenden-Konfiguration (§v10.306 Trust/Transparenz).

    PayPal-Spenden via Browser-Redirect lassen sich serverseitig nicht kryptografisch
    verifizieren — `payment_verification`/`guaranteed_capture` dokumentieren diese
    Grenze transparent statt sie zu verschweigen (§G8 (GEBOTE.md) Transparenz).
    """
    return {
        "primary_ok": PAYPAL_URL.startswith("https://"),
        "fallback_ok": PAYPAL_FALLBACK.startswith("https://"),
        "email_present": bool(PAYPAL_EMAIL),
        "payment_verification": "external_paypal_required",
        "guaranteed_capture": False,
    }


def get_donation_info() -> dict:
    """Gibt Spenden-Informationen als Dict zurück."""
    return {
        "url": PAYPAL_URL,
        "fallback": PAYPAL_FALLBACK,
        "email": PAYPAL_EMAIL,
        "configuration": validate_donation_configuration(),
    }


def extend_grace_period(amount_eur: float) -> float:
    """Verlängert die Reminder-Sperre nach einer Spende (§v10.306 Reziprozität).

    Größere Spenden verdienen eine längere Ruhephase (linear skaliert, gedeckelt).
    Persistiert kumulative Spendenstatistik für Danksagungs-UI/zukünftige Personalisierung.

    Returns:
        Anzahl der Tage, um die die Erinnerung ausgesetzt wird (>= 30).
    """
    days = min(_MAX_GRACE_DAYS, _BASE_GRACE_DAYS + max(0.0, amount_eur) * _GRACE_DAYS_PER_EUR)
    try:
        state: dict = {"count": 0, "total_donated_eur": 0.0}
        if _GRACE_FILE.exists():
            state.update(json.loads(_GRACE_FILE.read_text()))
        state["count"] = int(state.get("count", 0)) + 1
        state["last_donation_eur"] = float(amount_eur)
        state["total_donated_eur"] = float(state.get("total_donated_eur", 0.0)) + float(amount_eur)
        state["grace_until"] = time.time() + days * 86400.0
        _GRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _GRACE_FILE.write_text(json.dumps(state))
    except Exception:
        logger.debug("donation_reminder: grace period persist nicht blockierend Fehlschlag", exc_info=True)
    return days


def record_successful_run(quality_score: float = 0.0) -> None:
    """Zählt eine erfolgreiche Restaurierung für die Reziprozitäts-Schwelle (§v10.306).

    Write-only bzgl. Usage-Count — beeinflusst nicht den Rate-Limit-Stamp
    (der wird ausschließlich von `mark_reminder_shown()` geschrieben, CQRS).
    """
    _ = quality_score  # aktuell nur für zukünftige Personalisierung reserviert
    try:
        count = 0
        if _USAGE_COUNT_FILE.exists():
            count = int(_USAGE_COUNT_FILE.read_text().strip() or "0")
        _USAGE_COUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _USAGE_COUNT_FILE.write_text(str(count + 1))
    except Exception:
        logger.debug("donation_reminder: usage count nicht blockierend Fehlschlag", exc_info=True)


def should_show_reminder() -> bool:
    """Read-only Prüfung (§v10.306 CQRS): Reziprozitäts-Schwelle UND Rate-Limit (24h).

    Schreibt NICHTS — der Anzeige-Stamp wird ausschließlich von
    `mark_reminder_shown()` gesetzt, NACHDEM der Dialog tatsächlich gezeigt wurde.
    """
    try:
        count = 0
        if _USAGE_COUNT_FILE.exists():
            count = int(_USAGE_COUNT_FILE.read_text().strip() or "0")
        if count < _MIN_USES_BEFORE_REMINDER:
            return False

        if _STAMP_FILE.exists():
            last = float(_STAMP_FILE.read_text().strip())
            if time.monotonic() - last < _RATE_LIMIT_SECONDS:
                return False

        if _GRACE_FILE.exists():
            grace_state = json.loads(_GRACE_FILE.read_text())
            if time.monotonic() < float(grace_state.get("grace_until", 0.0)):
                return False

        return True
    except Exception:
        return True  # Bei Fehler lieber anzeigen als nie


def mark_reminder_shown() -> None:
    """Write-only Stamp (§v10.306 CQRS): merkt sich den Zeitpunkt der Anzeige."""
    try:
        _STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STAMP_FILE.write_text(str(time.monotonic()))
    except Exception:
        logger.debug("donation_reminder: stamp write nicht blockierend Fehlschlag", exc_info=True)
