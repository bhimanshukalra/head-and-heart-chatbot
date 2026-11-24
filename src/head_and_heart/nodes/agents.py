"""Response agent nodes (therapist and logical)."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from head_and_heart.prompts import LOGICAL_PROMPT, THERAPIST_PROMPT
from head_and_heart.state import State


def _make_agent(llm: BaseChatModel, system_prompt: str) -> Callable[[State], dict[str, Any]]:
    def agent(state: State) -> dict[str, Any]:
        reply = llm.invoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [reply]}

    return agent


def make_therapist(llm: BaseChatModel) -> Callable[[State], dict[str, Any]]:
    return _make_agent(llm, THERAPIST_PROMPT)


def make_logical(llm: BaseChatModel) -> Callable[[State], dict[str, Any]]:
    return _make_agent(llm, LOGICAL_PROMPT)
