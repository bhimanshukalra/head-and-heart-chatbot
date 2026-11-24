import json
import logging

from head_and_heart.logging import JsonFormatter, RedactingFormatter, thread_id_var


def _record(msg: str, **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("x", logging.INFO, "f", 1, msg, None, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_text_formatter_redacts_message_but_keeps_timestamp() -> None:
    fmt = RedactingFormatter("%(asctime)s %(message)s")
    out = fmt.format(_record("contact bob@example.com"))
    assert "bob@example.com" not in out
    assert "[phone]" not in out  # the timestamp must not be mistaken for a phone number
    assert out[:4].isdigit()


def test_json_formatter_has_context_and_extras() -> None:
    thread_id_var.set("t-1")
    line = JsonFormatter().format(_record("node completed", node="logical", latency_ms=12.5))
    data = json.loads(line)
    assert data["thread_id"] == "t-1"
    assert data["node"] == "logical"
    assert data["latency_ms"] == 12.5
    thread_id_var.set(None)


def test_json_formatter_redacts_pii() -> None:
    data = json.loads(JsonFormatter().format(_record("call +1 555 123 4567")))
    assert "555" not in data["message"]
