from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from head_and_heart.schemas import MessageClassifier


class FakeClassifierLLM:
    """Stands in for a chat model whose structured output is a fixed classification."""

    def __init__(self, result: Any) -> None:
        self.result = result

    def with_structured_output(self, schema: type) -> "FakeClassifierLLM":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def make_classifier_llm() -> Any:
    def _make(message_type: str | Exception) -> FakeClassifierLLM:
        if isinstance(message_type, Exception):
            return FakeClassifierLLM(message_type)
        return FakeClassifierLLM(MessageClassifier(message_type=message_type))

    return _make


@pytest.fixture
def make_agent_llm() -> Any:
    def _make(text: str) -> GenericFakeChatModel:
        return GenericFakeChatModel(messages=iter([AIMessage(text) for _ in range(10)]))

    return _make
