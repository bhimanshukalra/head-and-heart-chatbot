"""Graph state."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

MessageType = Literal["emotional", "logical", "mixed"]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_type: NotRequired[MessageType | None]
    confidence: NotRequired[float | None]
