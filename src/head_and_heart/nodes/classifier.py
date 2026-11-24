"""Message classifier node."""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from head_and_heart.prompts import CLASSIFIER_PROMPT
from head_and_heart.schemas import MessageClassifier
from head_and_heart.state import MessageType, State

logger = logging.getLogger(__name__)

DEFAULT_TYPE: MessageType = "logical"


def make_classifier(llm: BaseChatModel) -> Callable[[State], dict[str, Any]]:
    structured = llm.with_structured_output(MessageClassifier)

    def classify_message(state: State) -> dict[str, Any]:
        try:
            result = structured.invoke([SystemMessage(CLASSIFIER_PROMPT), state["messages"][-1]])
            if not isinstance(result, MessageClassifier):
                result = MessageClassifier.model_validate(result)
            return {"message_type": result.message_type}
        except Exception:
            logger.warning("Classification failed; defaulting to %s", DEFAULT_TYPE, exc_info=True)
            return {"message_type": DEFAULT_TYPE}

    return classify_message
