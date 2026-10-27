"""Input moderation: one call per user message, before it is saved or sent to the chat chain.

Moderators only classify. What to do with the verdict (allow / soft / block, and letting the
message through when moderation is down) is app.services.moderation_policy, see
docs/adr/0004-moderation-fail-open.md. Results never carry text; scores are read by the policy
but never logged or stored.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import httpx2
import openai

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODERATION_MODEL = "omni-moderation-latest"


@dataclass(frozen=True, slots=True)
class ModerationResult:
    provider: str
    model: str
    flagged: bool
    # Flagged categories by their API names ("self-harm/intent"); empty when not flagged.
    categories: tuple[str, ...] = ()
    # Score per category (API names), for our own thresholds. Never log or store them.
    scores: Mapping[str, float] = field(default_factory=dict, repr=False)


class ModerationUnavailable(Exception):
    """The moderation call failed. `code` is safe to log (no request or response text).

    misconfigured: our side is wrong (key, permissions, bad request) -> logged as error.
    """

    def __init__(self, code: str, *, misconfigured: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.misconfigured = misconfigured


class Moderator(Protocol):
    @property
    def enabled(self) -> bool:
        """False for NoopModerator: messages go through unchecked."""
        ...

    async def check(self, text: str) -> ModerationResult:
        """Raises ModerationUnavailable when the check could not be made."""
        ...

    async def aclose(self) -> None: ...


class OpenAIModerator:
    """OpenAI moderation endpoint (free, multilingual). Text input only."""

    provider = "openai"
    enabled = True

    def __init__(
        self,
        *,
        api_key: str,
        timeout: float,
        model: str = OPENAI_MODERATION_MODEL,
        base_url: str = OPENAI_BASE_URL,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.model = model
        # max_retries=0: a retry would add seconds before the user's answer even starts; when
        # the call fails the message goes through unchecked instead (fail-open).
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=httpx2.Timeout(timeout),
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def check(self, text: str) -> ModerationResult:
        try:
            response = await self._client.moderations.create(model=self.model, input=text)
        except openai.APIStatusError as exc:
            # 429/5xx are outages; any other status means our key or request is wrong.
            status = exc.status_code
            raise ModerationUnavailable(
                f"{type(exc).__name__}:{status}",
                misconfigured=status != 429 and status < 500,
            ) from exc
        except openai.APITimeoutError as exc:
            raise ModerationUnavailable("timeout") from exc
        except openai.APIConnectionError as exc:
            raise ModerationUnavailable("connection_error") from exc
        except openai.APIError as exc:
            raise ModerationUnavailable("api_error") from exc
        if not response.results:
            raise ModerationUnavailable("empty_result")
        result = response.results[0]
        flags = result.categories.model_dump(by_alias=True)
        scores = result.category_scores.model_dump(by_alias=True)
        return ModerationResult(
            provider=self.provider,
            model=response.model,
            flagged=result.flagged,
            categories=tuple(name for name, flagged in flags.items() if flagged),
            scores={name: float(score) for name, score in scores.items() if score is not None},
        )


class NoopModerator:
    """No OPENAI_API_KEY (dev only; staging/prod refuse to start): messages pass unchecked."""

    enabled = False

    async def check(self, text: str) -> ModerationResult:
        return ModerationResult(provider="none", model="none", flagged=False)

    async def aclose(self) -> None:
        return None
