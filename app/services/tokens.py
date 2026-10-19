"""Token arithmetic for the context budget, without provider tokenizers.

The chain mixes providers whose tokenizers differ, so a rough provider-neutral estimate with
headroom is used everywhere instead of an exact count for one of them.
"""

import math

CHARS_PER_TOKEN = 3.5
# Headroom for scripts that tokenize worse than English (Cyrillic, Polish diacritics).
SAFETY_MARGIN = 1.15
# Role markers and separators the chat template adds around every message.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN * SAFETY_MARGIN)


def message_tokens(content: str) -> int:
    return estimate_tokens(content) + MESSAGE_OVERHEAD_TOKENS


def context_budget(configured: int, chain_max_context: int, max_answer_tokens: int) -> int:
    """Input tokens per call: our own cap, and never more than the chain's window leaves for
    the answer (rule 3: chain_max_context is already the min across the chain)."""
    budget = min(configured, chain_max_context - max_answer_tokens)
    if budget <= 0:
        raise ValueError(
            "context budget <= 0: raise CONTEXT_BUDGET_TOKENS or lower CHAT_MAX_ANSWER_TOKENS"
        )
    return budget
