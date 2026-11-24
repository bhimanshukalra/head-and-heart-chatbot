"""Helpers for running one chat turn."""

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

AGENT_NODES = {"therapist", "logical", "blended"}


class InputTooLongError(ValueError):
    """Raised when a user message exceeds the configured limit."""


def validate_input(text: str, max_chars: int) -> str:
    """Strip and bound user input."""
    text = text.strip()
    if len(text) > max_chars:
        raise InputTooLongError(f"Message is too long (max {max_chars} characters).")
    return text


def stream_reply(graph: Any, user_text: str, config: RunnableConfig) -> Iterator[str]:
    """Yield the assistant reply: LLM tokens from agent nodes, or the fixed crisis message."""
    for mode, payload in graph.stream(
        {"messages": [HumanMessage(user_text)]}, config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, metadata = payload
            is_agent_chunk = metadata.get("langgraph_node") in AGENT_NODES and isinstance(
                chunk, AIMessageChunk
            )
            if is_agent_chunk and chunk.text:
                yield chunk.text
        elif mode == "updates" and "crisis_response" in payload:
            for message in payload["crisis_response"]["messages"]:
                yield message.text
