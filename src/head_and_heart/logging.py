"""Logging setup: PII redaction, optional JSON output, per-request thread id."""

import copy
import json
import logging
import re
from contextvars import ContextVar
from datetime import UTC, datetime

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d")
_LONG_NUMBER = re.compile(r"\b\d{9,}\b")

thread_id_var: ContextVar[str | None] = ContextVar("thread_id", default=None)

# Structured fields that callers may attach with `extra=`.
EXTRA_FIELDS = (
    "node",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "outcome",
)


def redact(text: str) -> str:
    """Mask emails, phone-like numbers and long digit strings."""
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_NUMBER.sub("[number]", text)


class RedactingFormatter(logging.Formatter):
    """Redacts the message and exception text only, so timestamps stay intact."""

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info:
            record.exc_text = redact(self.formatException(record.exc_info))
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with redacted message and exception text."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if thread_id := thread_id_var.get():
            payload["thread_id"] = thread_id
        for field in EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload)


def configure_logging(level: str, log_format: str = "text") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if log_format == "json"
        else RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
