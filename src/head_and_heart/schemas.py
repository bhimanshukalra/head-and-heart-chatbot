"""Structured-output schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from head_and_heart.state import MessageType


class MessageClassifier(BaseModel):
    message_type: MessageType = Field(
        ...,
        description=(
            "'emotional' for feelings/support, 'logical' for facts/solutions, "
            "'mixed' when the message clearly needs both."
        ),
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in the classification, from 0 to 1."
    )


class SafetyAssessment(BaseModel):
    category: Literal["none", "self_harm", "abuse", "medical_emergency"] = Field(
        ...,
        description=(
            "Pick a category only if the user describes a CURRENT, PERSONAL risk "
            "(to themselves or from someone else). Use 'none' for general or informational talk."
        ),
    )
