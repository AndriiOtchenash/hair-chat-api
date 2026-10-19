import pytest

from app.services.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    context_budget,
    estimate_tokens,
    message_tokens,
)


@pytest.mark.parametrize(
    ("text", "tokens"),
    [("", 0), ("x" * 7, 3), ("x" * 35, 12), ("я" * 350, 115)],  # len / 3.5 * 1.15, rounded up
)
def test_estimate_is_chars_over_3_5_plus_15_percent(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


def test_every_message_carries_role_overhead() -> None:
    assert message_tokens("x" * 35) == 12 + MESSAGE_OVERHEAD_TOKENS


def test_budget_is_our_cap_when_the_window_is_large() -> None:
    assert context_budget(6000, 131_072, 2048) == 6000


def test_budget_leaves_room_for_the_answer_in_a_small_window() -> None:
    assert context_budget(6000, 4096, 2048) == 2048


def test_no_room_left_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="CONTEXT_BUDGET_TOKENS"):
        context_budget(6000, 2048, 2048)
