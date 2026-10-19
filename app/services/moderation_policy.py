"""What we do with a moderation result: allow, soft (answer with care) or block (ADR 0004).

Shared by ChatService and scripts/moderation_eval.py, so the eval measures exactly this policy.
- self-harm*: never blocked. The message is answered with care and the client shows help lines;
  a block would leave a person in crisis with an error. Wins over every other category.
- violence, violence/graphic: hair-care talk is full of burns, cuts and bleeding scalps, so
  these block only at `violence_threshold`. Together with harassment/threatening they block
  anyway, because that category blocks on its own (next rule).
- anything else flagged by the provider: block.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.llm.moderation import ModerationResult

SELF_HARM_PREFIX = "self-harm"
VIOLENCE = frozenset({"violence", "violence/graphic"})


class ModerationAction(StrEnum):
    ALLOW = "allow"
    SOFT = "soft"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class Decision:
    action: ModerationAction
    # The provider's flagged categories that led to the action (all of them for soft/block).
    categories: tuple[str, ...] = ()


def decide(result: ModerationResult, *, violence_threshold: float) -> Decision:
    flagged = result.categories
    if not result.flagged or not flagged:
        return Decision(ModerationAction.ALLOW)
    if any(c.startswith(SELF_HARM_PREFIX) for c in flagged):
        return Decision(ModerationAction.SOFT, flagged)
    if any(c not in VIOLENCE or result.scores.get(c, 0.0) >= violence_threshold for c in flagged):
        return Decision(ModerationAction.BLOCK, flagged)
    # Only violence below the threshold: a hair-care injury, not a threat.
    return Decision(ModerationAction.ALLOW, flagged)
