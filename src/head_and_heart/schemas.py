"""Structured-output schemas."""

from pydantic import BaseModel, Field

from head_and_heart.state import MessageType


class MessageClassifier(BaseModel):
    message_type: MessageType = Field(
        ...,
        description="Whether the message needs an emotional (therapist) or logical response.",
    )
