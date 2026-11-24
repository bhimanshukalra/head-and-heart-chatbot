"""Graph assembly."""

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from head_and_heart.config import Settings
from head_and_heart.llm import get_chat_model
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
from head_and_heart.nodes.safety import make_crisis_response, make_safety_check
from head_and_heart.observability import instrument
from head_and_heart.state import State


def route_after_safety(state: State) -> Literal["crisis_response", "classifier"]:
    """Short-circuit to the fixed crisis reply when the safety check flags the message."""
    return "crisis_response" if state.get("safety") == "crisis" else "classifier"


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
    safety_llm: BaseChatModel | None = None,
    crisis_region: str = "US",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> CompiledStateGraph[State, None, State, State]:
    builder = StateGraph(State)
    nodes: dict[str, Callable[[State], dict[str, Any]]] = {
        "safety_check": make_safety_check(safety_llm or classifier_llm),
        "crisis_response": make_crisis_response(crisis_region),
        "classifier": make_classifier(classifier_llm, confidence_threshold),
        "therapist": make_therapist(therapist_llm, max_context_tokens),
        "blended": make_blended(blended_llm or therapist_llm, max_context_tokens),
        "logical": make_logical(logical_llm, max_context_tokens),
    }
    for name, node in nodes.items():
        builder.add_node(name, instrument(name, node))  # type: ignore[call-overload]

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {"crisis_response": "crisis_response", "classifier": "classifier"},
    )
    builder.add_edge("crisis_response", END)
    builder.add_conditional_edges(
        "classifier", route, {"therapist": "therapist", "logical": "logical", "blended": "blended"}
    )
    builder.add_edge("therapist", END)
    builder.add_edge("logical", END)
    builder.add_edge("blended", END)
    return builder.compile(checkpointer=checkpointer)


def build_default_graph(
    settings: Settings, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[State, None, State, State]:
    """Build the production graph with Groq models taken from settings."""
    return build_graph(
        classifier_llm=get_chat_model(
            settings, model_name=settings.classifier_model_name, temperature=0
        ),
        therapist_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.7),
        logical_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.2),
        blended_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.5),
        checkpointer=checkpointer,
        max_context_tokens=settings.max_context_tokens,
        confidence_threshold=settings.classifier_confidence_threshold,
        crisis_region=settings.crisis_region,
    )
