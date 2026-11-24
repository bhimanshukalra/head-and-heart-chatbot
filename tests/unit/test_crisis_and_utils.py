import pytest

from head_and_heart.chat import InputTooLongError, validate_input
from head_and_heart.crisis import crisis_message
from head_and_heart.logging import redact


def test_us_self_harm_message_has_988() -> None:
    assert "988" in crisis_message("self_harm", "US")


def test_unknown_region_falls_back_to_generic() -> None:
    msg = crisis_message("self_harm", "ZZ")
    assert "findahelpline" in msg
    assert "local emergency number" in msg


def test_medical_message_points_to_emergency_number() -> None:
    assert "999" in crisis_message("medical_emergency", "uk")


def test_validate_input_strips_and_limits() -> None:
    assert validate_input("  hi  ", 10) == "hi"
    with pytest.raises(InputTooLongError):
        validate_input("x" * 11, 10)


def test_redact_masks_pii() -> None:
    out = redact("mail bob@example.com or call +1 (555) 123-4567, id 123456789012")
    assert "bob@example.com" not in out
    assert "555" not in out
    assert "123456789012" not in out


def test_common_region_aliases_resolve() -> None:
    assert "116 123" in crisis_message("self_harm", "GB")
    assert "14416" in crisis_message("self_harm", "IND")
