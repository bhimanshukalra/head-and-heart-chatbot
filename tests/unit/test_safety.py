from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from head_and_heart.nodes.safety import make_safety_check


def _run(llm: Any, text: str) -> dict[str, Any]:
    return make_safety_check(llm)({"messages": [HumanMessage(text)]})


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I want to kill myself", "self_harm"),
        ("honestly i don't want to be alive anymore", "self_harm"),
        ("I've been cutting myself again", "self_harm"),
        ("my dad beats me", "abuse"),
        ("I think I overdosed", "medical_emergency"),
        ("I can't breathe", "medical_emergency"),
    ],
)
def test_strong_patterns_skip_the_llm(make_safety_llm: Any, text: str, category: str) -> None:
    # An LLM that raises proves the LLM was never consulted.
    out = _run(make_safety_llm(AssertionError("LLM must not be called")), text)
    assert (out["safety"], out["crisis_category"]) == ("crisis", category)


@pytest.mark.parametrize(
    "text", ["What is the capital of France?", "I feel a bit sad today", "Help me plan a trip"]
)
def test_ordinary_messages_pass(make_safety_llm: Any, text: str) -> None:
    out = _run(make_safety_llm(AssertionError("LLM must not be called")), text)
    assert out["safety"] == "ok"


def test_weak_match_cleared_by_llm(make_safety_llm: Any) -> None:
    out = _run(make_safety_llm("none"), "What are the suicide statistics for 2020?")
    assert out["safety"] == "ok"


def test_weak_match_confirmed_by_llm(make_safety_llm: Any) -> None:
    out = _run(make_safety_llm("self_harm"), "I feel hopeless and nothing matters")
    assert (out["safety"], out["crisis_category"]) == ("crisis", "self_harm")


def test_weak_match_fails_safe_when_llm_errors(make_safety_llm: Any) -> None:
    out = _run(make_safety_llm(RuntimeError("down")), "I have chest pain")
    assert (out["safety"], out["crisis_category"]) == ("crisis", "medical_emergency")
