"""Response agent nodes (therapist and logical)."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from head_and_heart.prompts import BLENDED_PROMPT, LOGICAL_PROMPT, THERAPIST_PROMPT
from head_and_heart.state import State

DEFAULT_MAX_CONTEXT_TOKENS = 6000


def _make_agent(
    llm: BaseChatModel, system_prompt: str, max_tokens: int
) -> Callable[[State], dict[str, Any]]:
    def agent(state: State) -> dict[str, Any]:
        history: Sequence[BaseMessage] = trim_messages(
            state["messages"],
            max_tokens=max_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        # Always keep the latest message even if it alone exceeds the budget.
        if not history:
            history = state["messages"][-1:]
        reply = llm.invoke([SystemMessage(system_prompt), *history])
        return {"messages": [reply]}

    return agent


def make_therapist(
    llm: BaseChatModel, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
) -> Callable[[State], dict[str, Any]]:
    return _make_agent(llm, THERAPIST_PROMPT, max_tokens)


def make_logical(
    llm: BaseChatModel, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
) -> Callable[[State], dict[str, Any]]:
    return _make_agent(llm, LOGICAL_PROMPT, max_tokens)


def make_blended(
    llm: BaseChatModel, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
) -> Callable[[State], dict[str, Any]]:
    return _make_agent(llm, BLENDED_PROMPT, max_tokens)
