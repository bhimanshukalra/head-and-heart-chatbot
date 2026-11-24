"""Fixed crisis responses. No LLM is involved, so wording is vetted and predictable.

Verify every number below for your target regions before production use.
"""

from head_and_heart.state import CrisisCategory

# region -> (crisis/suicide line, emergency number)
SELF_HARM_LINES: dict[str, tuple[str, str]] = {
    "US": ("call or text 988 (Suicide & Crisis Lifeline)", "911"),
    "CA": ("call or text 988 (Suicide Crisis Helpline)", "911"),
    "UK": ("call Samaritans on 116 123", "999"),
    "IN": ("call Tele-MANAS on 14416", "112"),
    "AU": ("call Lifeline on 13 11 14", "000"),
}
GENERIC_LINE = "find a local helpline at https://findahelpline.com"
GENERIC_EMERGENCY = "your local emergency number"

DOMESTIC_ABUSE_LINES: dict[str, str] = {
    "US": "the National Domestic Violence Hotline: 1-800-799-7233",
}

_LEAD: dict[CrisisCategory, str] = {
    "self_harm": (
        "I'm really sorry you're going through this, and I'm glad you told me. "
        "You deserve support right now, and I'm an AI, so I can't be the help you need."
    ),
    "abuse": (
        "I'm so sorry this is happening. What you describe is serious, "
        "and you deserve to be safe. I'm an AI, so I can't step in, but people can."
    ),
    "medical_emergency": (
        "This sounds like it could be a medical emergency. I'm an AI and can't help in person."
    ),
}


def crisis_message(category: CrisisCategory, region: str = "US") -> str:
    """Build the fixed crisis reply for a category and region."""
    line, emergency = SELF_HARM_LINES.get(region.upper(), (GENERIC_LINE, GENERIC_EMERGENCY))
    lead = _LEAD[category]
    if category == "medical_emergency":
        action = f"Please call {emergency} now, or ask someone nearby to help you."
    elif category == "abuse":
        hotline = DOMESTIC_ABUSE_LINES.get(region.upper(), "a local domestic violence hotline")
        action = f"If you are in immediate danger, call {emergency}. You can also reach {hotline}."
    else:
        action = (
            f"Please reach out right now: {line}. "
            f"If you might act on these thoughts or are in immediate danger, call {emergency}. "
            "If you can, tell someone you trust so you aren't alone."
        )
    return f"{lead}\n\n{action}"
