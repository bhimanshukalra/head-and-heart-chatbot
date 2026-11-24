from langchain_core.messages import AIMessage, HumanMessage

from head_and_heart.nodes.classifier import CONTEXT_MESSAGES, format_transcript
from head_and_heart.prompts import PROMPT_VERSION


def test_transcript_limits_context_and_marks_latest() -> None:
    msgs = [HumanMessage(f"m{i}") if i % 2 == 0 else AIMessage(f"m{i}") for i in range(10)]
    lines = format_transcript(msgs).splitlines()
    assert len(lines) == CONTEXT_MESSAGES
    assert lines[-1].startswith("LATEST ")
    assert "m0" not in format_transcript(msgs)


def test_long_messages_are_truncated() -> None:
    assert len(format_transcript([HumanMessage("x" * 5000)])) < 600


def test_prompt_version_is_set() -> None:
    assert PROMPT_VERSION
