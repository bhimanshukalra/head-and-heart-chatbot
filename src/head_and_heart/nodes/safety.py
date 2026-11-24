"""Safety check node: deterministic regex first, LLM confirmation for ambiguous cases."""

import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from head_and_heart.crisis import crisis_message
from head_and_heart.prompts import SAFETY_PROMPT
from head_and_heart.schemas import SafetyAssessment
from head_and_heart.state import CrisisCategory, State

logger = logging.getLogger(__name__)


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Explicit first-person statements: go straight to the crisis response (no LLM).
STRONG: dict[CrisisCategory, list[re.Pattern[str]]] = {
    "self_harm": _compile(
        [
            r"\b(kill|end|take)\s+(myself|my own life|my life)\b",
            r"\bwant(ed)?\s+to\s+die\b",
            r"\b(don'?t|do not)\s+want\s+to\s+(live|be alive|be here)\b",
            r"\bbetter off dead\b",
            r"\b(hurt|cut|harm)(t?ing)?\s+myself\b",
            r"\bend it all\b",
        ]
    ),
    "abuse": _compile(
        [
            r"\b(he|she|they|my\s+(dad|father|mom|mother|parents?|partner|husband|wife|boyfriend|"
            r"girlfriend|stepfather|stepmother|uncle|brother|sister))\s+"
            r"(hits?|beats?|abuses?|hurts?|rap(ed|es)|molests?)\s+me\b",
            r"\b(is|are|keeps?)\s+(hitting|beating|abusing)\s+me\b",
            r"\bi(\'m| am)\s+being\s+(abused|beaten)\b",
            r"\bi\s+was\s+(raped|sexually assaulted)\b",
        ]
    ),
    "medical_emergency": _compile(
        [
            r"\bi\s+(just\s+)?overdosed\b",
            r"\bi\s+took\s+too\s+many\s+(pills|tablets)\b",
            r"\bi(\'m| am)\s+having\s+a\s+(heart attack|stroke)\b",
            r"\bi\s+(can\'?t|cannot)\s+breathe\b",
        ]
    ),
}

# Ambiguous terms: a hit triggers an LLM check. If that check fails, assume crisis.
WEAK: dict[CrisisCategory, list[re.Pattern[str]]] = {
    "self_harm": _compile(
        [
            r"\bsuicid",
            r"\bself[- ]?harm",
            r"\bhopeless\b",
            r"\bcan\'?t go on\b",
            r"\bno reason to live\b",
            r"\boverdose\b",
        ]
    ),
    "abuse": _compile([r"\babus(e|ed|ive)\b", r"\bdomestic violence\b", r"\bassault"]),
    "medical_emergency": _compile(
        [
            r"\bchest pain\b",
            r"\bheart attack\b",
            r"\bstroke\b",
            r"\bunconscious\b",
            r"\bsevere bleeding\b",
        ]
    ),
}


def _first_match(
    text: str, table: dict[CrisisCategory, list[re.Pattern[str]]]
) -> CrisisCategory | None:
    for category, patterns in table.items():
        if any(p.search(text) for p in patterns):
            return category
    return None


def make_safety_check(llm: BaseChatModel) -> Callable[[State], dict[str, Any]]:
    structured = llm.with_structured_output(SafetyAssessment)

    def safety_check(state: State) -> dict[str, Any]:
        # Phones often type curly apostrophes, which would defeat the patterns.
        text = state["messages"][-1].text.replace("\u2019", "'")

        if category := _first_match(text, STRONG):
            logger.warning("Safety: explicit crisis pattern (%s)", category)
            return {"safety": "crisis", "crisis_category": category}

        if suspected := _first_match(text, WEAK):
            try:
                result = structured.invoke(
                    [SystemMessage(SAFETY_PROMPT), HumanMessage(text[:2000])]
                )
                if not isinstance(result, SafetyAssessment):
                    result = SafetyAssessment.model_validate(result)
            except Exception:
                logger.warning("Safety confirmation failed; failing safe (%s)", suspected)
                return {"safety": "crisis", "crisis_category": suspected}
            if result.category != "none":
                logger.warning("Safety: confirmed crisis (%s)", result.category)
                return {"safety": "crisis", "crisis_category": result.category}

        return {"safety": "ok", "crisis_category": None}

    return safety_check


def make_crisis_response(region: str) -> Callable[[State], dict[str, Any]]:
    def crisis_response(state: State) -> dict[str, Any]:
        category = state.get("crisis_category") or "self_harm"
        return {"messages": [AIMessage(crisis_message(category, region))]}

    return crisis_response
