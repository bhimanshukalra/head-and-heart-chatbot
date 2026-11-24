"""Helpers for running one chat turn."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

AGENT_NODES = {"therapist", "logical", "blended"}
STREAM_MODES = ["messages", "updates"]


class InputTooLongError(ValueError):
    """Raised when a user message exceeds the configured limit."""


def validate_input(text: str, max_chars: int) -> str:
    """Strip and bound user input."""
    text = text.strip()
    if len(text) > max_chars:
        raise InputTooLongError(f"Message is too long (max {max_chars} characters).")
    return text


def _texts(mode: str, payload: Any) -> Iterator[str]:
    """Extract reply text from one stream event: LLM tokens or the fixed crisis message."""
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


def stream_reply(graph: Any, user_text: str, config: RunnableConfig) -> Iterator[str]:
    for mode, payload in graph.stream(
        {"messages": [HumanMessage(user_text)]}, config, stream_mode=STREAM_MODES
    ):
        yield from _texts(mode, payload)


async def astream_reply(graph: Any, user_text: str, config: RunnableConfig) -> AsyncIterator[str]:
    async for mode, payload in graph.astream(
        {"messages": [HumanMessage(user_text)]}, config, stream_mode=STREAM_MODES
    ):
        for text in _texts(mode, payload):
            yield text
