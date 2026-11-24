"""Message classifier node."""

import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from head_and_heart.prompts import CLASSIFIER_PROMPT
from head_and_heart.schemas import MessageClassifier
from head_and_heart.state import MessageType, State

logger = logging.getLogger(__name__)

DEFAULT_TYPE: MessageType = "mixed"
CONTEXT_MESSAGES = 5  # latest message plus up to two earlier turns
MAX_CHARS_PER_MESSAGE = 500
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def format_transcript(messages: Sequence[BaseMessage]) -> str:
    """Render recent messages as a labelled transcript, marking the one to classify."""
    recent = list(messages)[-CONTEXT_MESSAGES:]
    lines = []
    for i, msg in enumerate(recent):
        role = "User" if msg.type == "human" else "Assistant"
        text = msg.text[:MAX_CHARS_PER_MESSAGE]
        lines.append(f"{role}: {text}")
        if i == len(recent) - 1:
            lines[-1] = f"LATEST {lines[-1]}"
    return "\n".join(lines)


def make_classifier(
    llm: BaseChatModel, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Callable[[State], dict[str, Any]]:
    structured = llm.with_structured_output(MessageClassifier)

    def classify_message(state: State) -> dict[str, Any]:
        try:
            result = structured.invoke(
                [
                    SystemMessage(CLASSIFIER_PROMPT),
                    HumanMessage(format_transcript(state["messages"])),
                ]
            )
            if not isinstance(result, MessageClassifier):
                result = MessageClassifier.model_validate(result)
        except Exception:
            logger.warning("Classification failed; defaulting to %s", DEFAULT_TYPE, exc_info=True)
            return {"message_type": DEFAULT_TYPE, "confidence": None}

        message_type = result.message_type
        if result.confidence < confidence_threshold:
            logger.info("Low confidence (%.2f); treating as %s", result.confidence, DEFAULT_TYPE)
            message_type = DEFAULT_TYPE
        return {"message_type": message_type, "confidence": result.confidence}

    return classify_message
