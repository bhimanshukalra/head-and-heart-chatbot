"""Logging setup with PII redaction."""

import logging
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d")
_LONG_NUMBER = re.compile(r"\b\d{9,}\b")


def redact(text: str) -> str:
    """Mask emails, phone-like numbers and long digit strings."""
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_NUMBER.sub("[number]", text)


class RedactingFormatter(logging.Formatter):
    """Redacts the final formatted record, including exception text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
