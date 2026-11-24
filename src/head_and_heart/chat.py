"""Helpers for running one chat turn."""

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

AGENT_NODES = {"therapist", "logical", "blended"}


def stream_reply(graph: Any, user_text: str, config: RunnableConfig) -> Iterator[str]:
    """Yield the assistant reply token by token (agent nodes only, not the classifier)."""
    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(user_text)]}, config, stream_mode="messages"
    ):
        is_agent_chunk = metadata.get("langgraph_node") in AGENT_NODES and isinstance(
            chunk, AIMessageChunk
        )
        if is_agent_chunk and chunk.text:
            yield chunk.text
