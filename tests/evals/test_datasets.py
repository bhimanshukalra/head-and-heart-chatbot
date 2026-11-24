"""Deterministic checks of the labeled datasets. No network and no LLM."""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from head_and_heart.nodes.safety import make_safety_check

HERE = Path(__file__).parent


def load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]


CLASSIFIER = load("classifier_dataset.jsonl")
SAFETY = load("safety_dataset.jsonl")


def _of(kind: str) -> list[dict[str, Any]]:
    return [r for r in SAFETY if r["kind"] == kind]


def _id(row: dict[str, Any]) -> str:
    return str(row["text"])[:40]


def test_classifier_dataset_is_well_formed() -> None:
    assert len(CLASSIFIER) >= 100
    assert len({r["text"] for r in CLASSIFIER}) == len(CLASSIFIER)
    for label in ("emotional", "logical", "mixed"):
        assert sum(r["label"] == label for r in CLASSIFIER) >= 20
    for row in CLASSIFIER:
        assert row["label"] in {"emotional", "logical", "mixed"}
        assert all(m["role"] in {"user", "assistant"} for m in row.get("context", []))


class _NoLLM:
    """Fails the test if the safety node consults the LLM."""

    def with_structured_output(self, schema: type) -> "_NoLLM":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        raise AssertionError("LLM was consulted")


class _Clears:
    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, schema: type) -> "_Clears":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        from head_and_heart.schemas import SafetyAssessment

        self.calls += 1
        return SafetyAssessment(category="none")


def _check(llm: Any, text: str) -> dict[str, Any]:
    return make_safety_check(llm)({"messages": [HumanMessage(text)]})


@pytest.mark.parametrize(
    "row", [r for r in SAFETY if r["kind"] == "strong"], ids=lambda r: r["text"][:40]
)
def test_explicit_crisis_statements_are_caught_without_llm(row: dict[str, Any]) -> None:
    out = _check(_NoLLM(), row["text"])
    assert out == {
        "safety": "crisis",
        "crisis_category": row["category"],
        "crisis_lang": row.get("lang", "en"),
    }


@pytest.mark.parametrize(
    "row", [r for r in SAFETY if r["kind"] == "weak"], ids=lambda r: r["text"][:40]
)
def test_ambiguous_statements_are_sent_to_llm(row: dict[str, Any]) -> None:
    llm = _Clears()
    out = _check(llm, row["text"])
    assert llm.calls == 1
    assert out["safety"] == "ok"  # the LLM cleared it


@pytest.mark.parametrize(
    "row", [r for r in SAFETY if r["kind"] == "pass"], ids=lambda r: r["text"][:40]
)
def test_benign_messages_trigger_nothing(row: dict[str, Any]) -> None:
    assert _check(_NoLLM(), row["text"])["safety"] == "ok"
