import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from head_and_heart import config
from head_and_heart.config import Settings
from head_and_heart.crisis import crisis_message
from head_and_heart.llm import get_chat_model
from head_and_heart.logging import JsonFormatter, RedactingFormatter, configure_logging
from head_and_heart.observability import (
    CLASSIFIER_FALLBACKS,
    NODE_ERRORS,
    _tokens,
    instrument,
)


def test_get_chat_model_builds_groq_model_from_settings() -> None:
    settings = Settings(_env_file=None, groq_api_key="x", max_retries=5)
    model = get_chat_model(settings, model_name="openai/gpt-oss-120b", temperature=0.3)
    assert isinstance(model, ChatGroq)
    assert model.temperature == 0.3
    assert model.max_retries == 5


def test_get_settings_is_cached_and_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k")
    config.get_settings.cache_clear()
    try:
        assert config.get_settings() is config.get_settings()
    finally:
        config.get_settings.cache_clear()


def test_abuse_message_mentions_emergency_and_hotline() -> None:
    assert "911" in crisis_message("abuse", "US")
    assert "1-800-799-7233" in crisis_message("abuse", "US")
    assert "local domestic violence hotline" in crisis_message("abuse", "UK")


def _exc_record() -> logging.LogRecord:
    try:
        raise ValueError("bad bob@example.com")
    except ValueError:
        import sys

        return logging.LogRecord("x", logging.ERROR, "f", 1, "failed", None, sys.exc_info())


def test_formatters_redact_exception_text() -> None:
    assert "bob@example.com" not in RedactingFormatter("%(message)s").format(_exc_record())
    assert "bob@example.com" not in JsonFormatter().format(_exc_record())


def test_configure_logging_selects_format() -> None:
    configure_logging("INFO", "json")
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)
    configure_logging("INFO", "text")
    assert isinstance(logging.getLogger().handlers[0].formatter, RedactingFormatter)


def test_token_usage_extracted_from_ai_message() -> None:
    msg = AIMessage("x", usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})
    assert _tokens({"messages": [msg]}) == (3, 4)
    assert _tokens({"messages": [AIMessage("no usage")]}) == (0, 0)


def test_instrument_counts_classifier_fallback() -> None:
    before = CLASSIFIER_FALLBACKS._value.get()
    instrument("classifier", lambda s: {"message_type": "mixed", "confidence": None})({})
    assert CLASSIFIER_FALLBACKS._value.get() == before + 1


def test_instrument_counts_and_reraises_errors() -> None:
    def boom(_state: Any) -> dict[str, Any]:
        raise RuntimeError("x")

    before = NODE_ERRORS.labels("boom_node")._value.get()
    with pytest.raises(RuntimeError):
        instrument("boom_node", boom)({})
    assert NODE_ERRORS.labels("boom_node")._value.get() == before + 1


def test_parse_prices_handles_slashes_and_rejects_garbage() -> None:
    from head_and_heart.observability import parse_prices

    assert parse_prices("openai/gpt-oss-120b:0.15:0.60, m2:1:2") == {
        "openai/gpt-oss-120b": (0.15, 0.60),
        "m2": (1.0, 2.0),
    }
    assert parse_prices("") == {}
    for bad in ("model:x:y", "nocolons", ":1:2"):
        with pytest.raises(ValueError):
            parse_prices(bad)


def test_cost_is_computed_from_tokens_and_model_prices() -> None:
    from head_and_heart.observability import LLM_COST, configure_pricing

    configure_pricing("test-model:2:10")
    msg = AIMessage(
        "x",
        usage_metadata={
            "input_tokens": 1_000_000,
            "output_tokens": 500_000,
            "total_tokens": 1_500_000,
        },
        response_metadata={"model_name": "test-model"},
    )
    before = LLM_COST.labels("costnode")._value.get()
    instrument("costnode", lambda s: {"messages": [msg]})({})
    assert LLM_COST.labels("costnode")._value.get() == pytest.approx(before + 7.0)
    configure_pricing("")  # restore global state


def test_unknown_model_costs_nothing() -> None:
    from head_and_heart.observability import _usage, configure_pricing

    configure_pricing("")
    msg = AIMessage(
        "x",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "unpriced"},
    )
    assert _usage({"messages": [msg]}) == (10, 5, 0.0)
