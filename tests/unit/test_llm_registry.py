import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.llm.registry import MODELS, build_llm, build_providers, parse_chain
from tests.llm_fakes import FakeLLMCallRecorder

CHAIN = "gemini:gemini-3.8-flash, gemini:gemini-3.5-flash-lite ,groq:openai/gpt-oss-120b"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_chain": CHAIN,
        "gemini_api_key": SecretStr("g"),
        "groq_api_key": SecretStr("q"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def keys(s: Settings, skip: tuple[str, ...] = ()) -> list[str]:
    return [f"{p.name}:{p.model}" for p in build_providers(s, skip)]


def test_parse_chain_keeps_order_and_splits_on_first_colon() -> None:
    assert parse_chain(CHAIN) == [
        ("gemini", "gemini-3.8-flash"),
        ("gemini", "gemini-3.5-flash-lite"),
        ("groq", "openai/gpt-oss-120b"),
    ]


@pytest.mark.parametrize("chain", ["", " , ", "gemini", "gemini:", ":model"])
def test_parse_chain_rejects_malformed(chain: str) -> None:
    with pytest.raises(ValueError):
        parse_chain(chain)


def test_default_chain_uses_only_known_models() -> None:
    default = Settings.model_fields["llm_chain"].default
    assert {f"{p}:{m}" for p, m in parse_chain(default)} <= set(MODELS)


def test_builds_chain_with_budget_of_smallest_window() -> None:
    llm = build_llm(settings(), FakeLLMCallRecorder())

    assert [f"{p.name}:{p.model}" for p in llm.providers] == [
        "gemini:gemini-3.8-flash",
        "gemini:gemini-3.5-flash-lite",
        "groq:openai/gpt-oss-120b",
    ]
    assert llm.max_context_tokens == 131_072


def test_skip_removes_links() -> None:
    assert keys(settings(), ("gemini:gemini-3.8-flash", "groq:openai/gpt-oss-120b")) == [
        "gemini:gemini-3.5-flash-lite"
    ]


def test_skip_of_unknown_link_is_an_error() -> None:
    with pytest.raises(ValueError, match="not in LLM_CHAIN"):
        keys(settings(), ("gemini:nope",))


def test_skipping_everything_is_an_error() -> None:
    with pytest.raises(ValueError, match="skipped"):
        keys(settings(llm_chain="groq:openai/gpt-oss-120b"), ("groq:openai/gpt-oss-120b",))


def test_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown LLM provider"):
        keys(settings(llm_chain="deepseek:chat"))


def test_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        keys(settings(llm_chain="gemini:gemini-made-up"))


def test_missing_api_key() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        keys(settings(groq_api_key=None))
