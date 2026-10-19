import pytest

from app.llm.moderation import ModerationResult
from app.services.moderation_policy import ModerationAction, decide
from tests.llm_fakes import clean, flag

ALLOW, SOFT, BLOCK = ModerationAction.ALLOW, ModerationAction.SOFT, ModerationAction.BLOCK


@pytest.mark.parametrize(
    ("result", "action"),
    [
        (clean(), ALLOW),
        (flag("self-harm", scores={"self-harm": 0.4}), SOFT),
        (flag("self-harm/instructions", "illicit"), SOFT),
        (flag("violence", scores={"violence": 0.2}), ALLOW),
        (
            flag(
                "violence", "violence/graphic", scores={"violence": 0.89, "violence/graphic": 0.1}
            ),
            ALLOW,
        ),
        (flag("violence/graphic", scores={"violence/graphic": 0.9}), BLOCK),
        (flag("violence", "harassment/threatening", scores={"violence": 0.3}), BLOCK),
        (flag("harassment", scores={"harassment": 0.3}), BLOCK),
        (flag("sexual/minors"), BLOCK),
    ],
)
def test_actions(result: ModerationResult, action: ModerationAction) -> None:
    decision = decide(result, violence_threshold=0.9)

    assert decision.action is action
    assert decision.categories == result.categories


def test_threshold_comes_from_the_caller() -> None:
    result = flag("violence", scores={"violence": 0.6})

    assert decide(result, violence_threshold=0.5).action is BLOCK
    assert decide(result, violence_threshold=0.7).action is ALLOW
