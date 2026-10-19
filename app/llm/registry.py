"""Known providers and models, LLM_CHAIN parsing and assembly of the ResilientLLM.

Model names and context windows were taken from each provider's GET /models (2026-10-19);
add a model here before using it in LLM_CHAIN.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from openai.types.shared import ReasoningEffort
from pydantic import SecretStr

from app.core.config import Settings
from app.db.repositories.llm_calls import LLMCallRecorder
from app.llm.base import LLMProvider, link_key
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.resilient import ResilientLLM


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    base_url: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    max_context_tokens: int
    # Thinking models spend time (first token) and max_tokens on reasoning; keep it low for chat.
    reasoning_effort: ReasoningEffort = None


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"
    ),
    "groq": ProviderSpec("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

MODELS: dict[str, ModelSpec] = {
    "gemini:gemini-3.8-flash": ModelSpec(1_048_576, "low"),
    "gemini:gemini-3.5-flash-lite": ModelSpec(1_048_576, "low"),
    "groq:openai/gpt-oss-120b": ModelSpec(131_072, "low"),
}


def parse_chain(chain: str) -> list[tuple[str, str]]:
    """`provider:model,provider:model` -> [(provider, model)], in priority order.

    Splits on the first `:` only, since model ids may contain `/` or `:` (openai/gpt-oss-120b).
    """
    links: list[tuple[str, str]] = []
    for raw in chain.split(","):
        item = raw.strip()
        if not item:
            continue
        provider, sep, model = item.partition(":")
        if not sep or not provider.strip() or not model.strip():
            raise ValueError(f"LLM_CHAIN item {item!r} is not in provider:model form")
        links.append((provider.strip(), model.strip()))
    if not links:
        raise ValueError("LLM_CHAIN is empty")
    return links


def _api_key(settings: Settings, provider: str) -> SecretStr | None:
    keys: dict[str, SecretStr | None] = {
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
    }
    return keys.get(provider)


def build_providers(settings: Settings, skip: Iterable[str] = ()) -> list[LLMProvider]:
    skipped = set(skip)
    links = parse_chain(settings.llm_chain)
    unknown_skips = skipped - {link_key(p, m) for p, m in links}
    if unknown_skips:
        raise ValueError(f"--skip names links not in LLM_CHAIN: {', '.join(sorted(unknown_skips))}")

    providers: list[LLMProvider] = []
    for provider, model in links:
        key = link_key(provider, model)
        if key in skipped:
            continue
        provider_spec = PROVIDERS.get(provider)
        if provider_spec is None:
            raise ValueError(f"unknown LLM provider {provider!r} (known: {', '.join(PROVIDERS)})")
        model_spec = MODELS.get(key)
        if model_spec is None:
            raise ValueError(f"unknown model {key!r}: add it to app/llm/registry.py MODELS")
        api_key = _api_key(settings, provider)
        if api_key is None:
            raise ValueError(f"{provider_spec.api_key_env} is not set but {key} is in LLM_CHAIN")
        providers.append(
            OpenAICompatibleProvider(
                name=provider,
                model=model,
                max_context_tokens=model_spec.max_context_tokens,
                base_url=provider_spec.base_url,
                api_key=api_key.get_secret_value(),
                connect_timeout=settings.llm_connect_timeout_s,
                read_timeout=settings.llm_read_timeout_s,
                reasoning_effort=model_spec.reasoning_effort,
            )
        )
    if not providers:
        raise ValueError("every link of LLM_CHAIN was skipped")
    return providers


def build_llm(
    settings: Settings, recorder: LLMCallRecorder, skip: Iterable[str] = ()
) -> ResilientLLM:
    return ResilientLLM(
        build_providers(settings, skip),
        recorder,
        first_token_timeout=settings.llm_first_token_timeout_s,
    )
