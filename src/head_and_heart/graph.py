"""Graph assembly."""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from head_and_heart.nodes.agents import make_logical, make_therapist
from head_and_heart.nodes.classifier import DEFAULT_TYPE, make_classifier
from head_and_heart.state import State


def route(state: State) -> Literal["therapist", "logical"]:
    """Pick the agent node for the classified message type."""
    message_type = state.get("message_type") or DEFAULT_TYPE
    return "therapist" if message_type == "emotional" else "logical"


def build_graph(
    *,
    classifier_llm: BaseChatModel,
    therapist_llm: BaseChatModel,
    logical_llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[State, None, State, State]:
    builder = StateGraph(State)
    builder.add_node("classifier", make_classifier(classifier_llm))  # type: ignore[call-overload]
    builder.add_node("therapist", make_therapist(therapist_llm))  # type: ignore[call-overload]
    builder.add_node("logical", make_logical(logical_llm))  # type: ignore[call-overload]

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier", route, {"therapist": "therapist", "logical": "logical"}
    )
    builder.add_edge("therapist", END)
    builder.add_edge("logical", END)
    return builder.compile(checkpointer=checkpointer)
