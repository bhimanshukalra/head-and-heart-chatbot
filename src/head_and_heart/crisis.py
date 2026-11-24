"""Fixed crisis responses. No LLM is involved, so wording is vetted and predictable.

Numbers were checked against official sources on 2026-11-24 (988lifeline.org, 988.ca,
samaritans.org, telemanas / PIB, lifeline.org.au, thehotline.org). The emergency numbers
(911, 999, 112, 000) are the standard national ones. Re-check them periodically and before
adding a region.
"""

from head_and_heart.state import CrisisCategory, Language

# region -> (crisis/suicide line, emergency number)
SELF_HARM_LINES: dict[str, tuple[str, str]] = {
    "US": ("call or text 988 (Suicide & Crisis Lifeline)", "911"),
    "CA": ("call or text 988 (Suicide Crisis Helpline)", "911"),
    "UK": ("call Samaritans on 116 123", "999"),
    "IN": ("call Tele-MANAS on 14416", "112"),
    "AU": ("call Lifeline on 13 11 14", "000"),
}
REGION_ALIASES = {"GB": "UK", "IND": "IN", "AUS": "AU", "USA": "US", "CAN": "CA"}
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


_LEAD_ES: dict[CrisisCategory, str] = {
    "self_harm": (
        "Lamento mucho que estés pasando por esto, y me alegra que me lo hayas contado. "
        "Mereces apoyo ahora mismo; soy una IA y no puedo ser la ayuda que necesitas."
    ),
    "abuse": (
        "Lamento mucho que esto esté pasando. Lo que describes es grave y mereces estar a salvo. "
        "Soy una IA y no puedo intervenir, pero otras personas sí."
    ),
    "medical_emergency": (
        "Esto podría ser una emergencia médica. Soy una IA y no puedo ayudarte en persona."
    ),
}
_US_LINE_ES = "llama o envía un mensaje de texto al 988 (oprime 2 para español)"
_GENERIC_LINE_ES = "busca una línea de ayuda local en https://findahelpline.com"
_GENERIC_EMERGENCY_ES = "tu número local de emergencias"


def _crisis_message_es(category: CrisisCategory, region: str) -> str:
    _, emergency = SELF_HARM_LINES.get(region, (GENERIC_LINE, GENERIC_EMERGENCY))
    if emergency == GENERIC_EMERGENCY:
        emergency = _GENERIC_EMERGENCY_ES
    lead = _LEAD_ES[category]
    if category == "medical_emergency":
        action = f"Llama ahora al {emergency}, o pide ayuda a alguien cercano."
    elif category == "abuse":
        hotline = (
            "la Línea Nacional de Violencia Doméstica de EE. UU. (1-800-799-7233)"
            if region == "US"
            else "una línea local de violencia doméstica"
        )
        action = (
            f"Si estás en peligro inmediato, llama al {emergency}. "
            f"También puedes contactar a {hotline}."
        )
    else:
        line = _US_LINE_ES if region == "US" else _GENERIC_LINE_ES
        action = (
            f"Busca ayuda ahora mismo: {line}. "
            "Si podrías actuar según estos pensamientos o estás en peligro inmediato, "
            f"llama al {emergency}. "
            "Si puedes, cuéntaselo a alguien de confianza para no estar solo o sola."
        )
    return f"{lead}\n\n{action}"


def crisis_message(category: CrisisCategory, region: str = "US", lang: Language = "en") -> str:
    """Build the fixed crisis reply for a category, region and language."""
    region = REGION_ALIASES.get(region.upper(), region.upper())
    if lang == "es":
        return _crisis_message_es(category, region)
    line, emergency = SELF_HARM_LINES.get(region, (GENERIC_LINE, GENERIC_EMERGENCY))
    lead = _LEAD[category]
    if category == "medical_emergency":
        action = f"Please call {emergency} now, or ask someone nearby to help you."
    elif category == "abuse":
        hotline = DOMESTIC_ABUSE_LINES.get(region, "a local domestic violence hotline")
        action = f"If you are in immediate danger, call {emergency}. You can also reach {hotline}."
    else:
        action = (
            f"Please reach out right now: {line}. "
            f"If you might act on these thoughts or are in immediate danger, call {emergency}. "
            "If you can, tell someone you trust so you aren't alone."
        )
    return f"{lead}\n\n{action}"
