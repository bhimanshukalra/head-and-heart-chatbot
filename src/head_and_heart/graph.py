"""Graph assembly."""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from head_and_heart.nodes.agents import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    make_blended,
    make_logical,
    make_therapist,
)
from head_and_heart.nodes.classifier import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TYPE,
    make_classifier,
)
from head_and_heart.state import State


def route(state: State) -> Literal["therapist", "logical", "blended"]:
    """Pick the agent node for the classified message type."""
    message_type = state.get("message_type") or DEFAULT_TYPE
    if message_type == "emotional":
        return "therapist"
    if message_type == "logical":
        return "logical"
    return "blended"


def build_graph(
    *,
    classifier_llm: BaseChatModel,
    therapist_llm: BaseChatModel,
    logical_llm: BaseChatModel,
    blended_llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> CompiledStateGraph[State, None, State, State]:
    builder = StateGraph(State)
    builder.add_node("classifier", make_classifier(classifier_llm, confidence_threshold))  # type: ignore[call-overload]
    builder.add_node("therapist", make_therapist(therapist_llm, max_context_tokens))  # type: ignore[call-overload]
    builder.add_node(
        "blended",
        make_blended(blended_llm or therapist_llm, max_context_tokens),  # type: ignore[call-overload]
    )
    builder.add_node("logical", make_logical(logical_llm, max_context_tokens))  # type: ignore[call-overload]

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier", route, {"therapist": "therapist", "logical": "logical", "blended": "blended"}
    )
    builder.add_edge("therapist", END)
    builder.add_edge("logical", END)
    builder.add_edge("blended", END)
    return builder.compile(checkpointer=checkpointer)
