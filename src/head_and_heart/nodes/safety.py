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
from head_and_heart.state import CrisisCategory, Language, State

logger = logging.getLogger(__name__)


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Spanish "me quiero morir de vergüenza/risa/..." is a common idiom, not a crisis statement.
_NOT_IDIOM = (
    r"(?!\s+de\s+(risa|verg|pena|hambre|sue|calor|fr[ií]o|miedo|celos|envidia|aburr|cansanc))"
)


Table = dict[CrisisCategory, list[re.Pattern[str]]]

# Explicit first-person statements: go straight to the crisis response (no LLM).
STRONG: dict[Language, Table] = {
    "en": {
        "self_harm": _compile(
            [
                r"\b(kill|end)\s+(myself|my\s?self|myslef|mysefl|my\s+(own\s+)?life)\b",
                r"\b(take|taking|took)\s+my\s+own\s+life\b",
                r"\b(take|taking|took)\s+my\s+life\b"
                r"(?!\s+(back|in|into|to|and|more|seriously|from|off)\b)",
                r"\b(want(ed)?\s+to|wanna|wan\s+na)\s+die\b",
                r"\b(don'?t|do not)\s+want\s+to\s+(live|be alive|be here|wake up)\b",
                r"\bwish\s+i\s+(was|were)\s+dead\b",
                r"\bbetter off dead\b",
                r"\b(hurt|cut|harm)(t?ing)?\s+myself\b",
                r"\bend it all\b",
                r"\bun-?alive\s+(myself|me)\b",
                r"\boff\s+myself\b",
                r"\bno\s+point\s+(in|to)\s+(living|going on|life)\b",
            ]
        ),
        "abuse": _compile(
            [
                r"\b(he|she|they|my\s+(dad|father|mom|mother|parents?|partner|husband|wife|boyfriend|"
                r"girlfriend|stepfather|stepmother|uncle|brother|sister))\s+"
                r"(hits?|beats?|abuses?|hurts?|chokes?|strangles?|rap(ed|es)|molests?)\s+me\b",
                r"\b(he|she|they)('s| is| has| have|'ve)?\s+(been\s+)?"
                r"(hitting|beating|choking|strangling)\s+me\b",
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
    },
    "es": {
        "self_harm": _compile(
            [
                r"\bquiero\s+(morir|morirme|matarme|suicidarme|acabar\s+con\s+(mi\s+vida|todo))\b"
                + _NOT_IDIOM,
                r"\bme\s+quiero\s+(morir|matar|suicidar)\b" + _NOT_IDIOM,
                r"\bvoy\s+a\s+(matarme|suicidarme|quitarme\s+la\s+vida)\b",
                r"\bquitarme\s+la\s+vida\b",
                r"\bno\s+quiero\s+(vivir|seguir\s+viviendo|estar\s+vivo|estar\s+viva)\b",
                r"\b(mejor|estar[ií]a\s+mejor)\s+muert[oa]\b",
                r"\bhacerme\s+da[ñn]o\b",
            ]
        ),
        "abuse": _compile(
            [
                r"\bmi\s+(pap[aá]|padre|mam[aá]|madre|pareja|esposo|esposa|novio|novia|padrastro|"
                r"t[ií]o|hermano)\s+me\s+(pega|golpea|maltrata|viola)\b",
                r"\bme\s+(pega|golpea|maltrata)\b",
                r"\bme\s+(violaron|abusaron)\b",
            ]
        ),
        "medical_emergency": _compile(
            [
                r"\bno\s+puedo\s+respirar\b",
                r"\btuve\s+una\s+sobredosis\b",
                r"\btom[eé]\s+demasiadas\s+pastillas\b",
                r"\bestoy\s+teniendo\s+un\s+(infarto|derrame)\b",
            ]
        ),
    },
}

# Ambiguous terms: a hit triggers an LLM check. If that check fails, assume crisis.
WEAK: dict[Language, Table] = {
    "en": {
        "self_harm": _compile(
            [
                r"\bsuicid",
                r"\b(sucide|suicde|suiside|sucidal|suiscide)\b",
                r"\bself[- ]?harm",
                r"(?<![\d.])(?<!\d\s)\bkms\b",
                r"\bhopeless\b",
                r"\bcan\'?t go on\b",
                r"\bcan\'?t (take|do) (this|it) (anymore|any more)\b",
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
    },
    "es": {
        "self_harm": _compile(
            [r"\bsuicid(io|arme|arse)\b", r"\bsin\s+esperanza\b", r"\bsobredosis\b"]
        ),
        "abuse": _compile([r"\bviolencia\s+dom[eé]stica\b", r"\babuso\b"]),
        "medical_emergency": _compile([r"\bdolor\s+de\s+pecho\b"]),
    },
}


def _first_match(
    text: str, tables: dict[Language, Table]
) -> tuple[CrisisCategory, Language] | None:
    for lang, table in tables.items():
        for category, patterns in table.items():
            if any(p.search(text) for p in patterns):
                return category, lang
    return None


def make_safety_check(llm: BaseChatModel) -> Callable[[State], dict[str, Any]]:
    structured = llm.with_structured_output(SafetyAssessment)

    def safety_check(state: State) -> dict[str, Any]:
        # Phones often type curly apostrophes, which would defeat the patterns.
        text = state["messages"][-1].text.replace("\u2019", "'")

        if hit := _first_match(text, STRONG):
            category, lang = hit
            logger.warning("Safety: explicit crisis pattern (%s, %s)", category, lang)
            return {"safety": "crisis", "crisis_category": category, "crisis_lang": lang}

        if suspected := _first_match(text, WEAK):
            suspected_category, lang = suspected
            try:
                result = structured.invoke(
                    [SystemMessage(SAFETY_PROMPT), HumanMessage(text[:2000])]
                )
                if not isinstance(result, SafetyAssessment):
                    result = SafetyAssessment.model_validate(result)
            except Exception:
                logger.warning("Safety confirmation failed; failing safe (%s)", suspected_category)
                return {
                    "safety": "crisis",
                    "crisis_category": suspected_category,
                    "crisis_lang": lang,
                }
            if result.category != "none":
                logger.warning("Safety: confirmed crisis (%s)", result.category)
                return {"safety": "crisis", "crisis_category": result.category, "crisis_lang": lang}

        return {"safety": "ok", "crisis_category": None, "crisis_lang": "en"}

    return safety_check


def make_crisis_response(region: str) -> Callable[[State], dict[str, Any]]:
    def crisis_response(state: State) -> dict[str, Any]:
        category = state.get("crisis_category") or "self_harm"
        lang = state.get("crisis_lang") or "en"
        return {"messages": [AIMessage(crisis_message(category, region, lang))]}

    return crisis_response
