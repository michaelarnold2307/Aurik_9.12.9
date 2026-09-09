from __future__ import annotations

from collections.abc import Generator
from typing import cast

import pytest

pytest.importorskip("PyQt5")  # CI-Minimal-Umgebung (cross-platform) ohne PyQt5
from PyQt5 import QtWidgets

from Aurik10.i18n import set_language, t
from Aurik10.ui.onboarding import OnboardingWizard


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return cast(QtWidgets.QApplication, app)


@pytest.fixture(autouse=True)
def restore_language() -> Generator[None]:
    set_language("de")
    yield
    set_language("de")


@pytest.mark.unit
def test_onboarding_ready_page_uses_finish_button(qapp: QtWidgets.QApplication) -> None:
    set_language("de")
    wizard = OnboardingWizard()
    wizard._current_page = 2
    wizard._stack.setCurrentIndex(2)
    wizard._update_dots()

    assert wizard._btn_next.text() == t("onboarding.finish")


@pytest.mark.unit
@pytest.mark.parametrize("lang", ["de", "en"])
def test_onboarding_i18n_keys_are_resolved(lang: str) -> None:
    set_language(lang)
    for key in (
        "onboarding.welcome.title",
        "onboarding.welcome.body",
        "onboarding.how.title",
        "onboarding.how.step_2",
        "onboarding.ready.body",
        "onboarding.show_again",
    ):
        assert t(key) != key
