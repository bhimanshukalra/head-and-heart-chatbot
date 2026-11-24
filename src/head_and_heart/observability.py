"""Metrics and node instrumentation.

Message content is never logged or recorded here: only node names, timings and token counts.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from prometheus_client import Counter, Histogram

logger = logging.getLogger("head_and_heart.node")

NODE_LATENCY = Histogram("node_latency_seconds", "Graph node latency.", ["node"])
NODE_ERRORS = Counter("node_errors_total", "Unhandled errors raised by a node.", ["node"])
MESSAGES_CLASSIFIED = Counter(
    "messages_classified_total", "Messages by classification.", ["message_type"]
)
CLASSIFIER_FALLBACKS = Counter(
    "classifier_fallbacks_total", "Classifier failures that used the default route."
)
CRISIS_TRIGGERS = Counter("crisis_triggers_total", "Crisis responses by category.", ["category"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens used.", ["node", "direction"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM spend in USD.", ["node"])
CHAT_REQUESTS = Counter("chat_requests_total", "HTTP chat requests by outcome.", ["outcome"])


# model name -> (USD per 1M input tokens, USD per 1M output tokens). Empty disables cost tracking.
PRICES: dict[str, tuple[float, float]] = {}


def parse_prices(raw: str) -> dict[str, tuple[float, float]]:
    """Parse 'model:input:output,...' (USD per 1M tokens). Model names may contain '/' and ':'."""
    prices: dict[str, tuple[float, float]] = {}
    for item in filter(None, (p.strip() for p in raw.split(","))):
        model, _, rest = item.rpartition(":")
        model, _, price_in = model.rpartition(":")
        try:
            prices[model] = (float(price_in), float(rest))
        except ValueError as exc:
            raise ValueError("MODEL_PRICES must look like 'model:input:output,...'.") from exc
        if not model:
            raise ValueError("MODEL_PRICES must look like 'model:input:output,...'.")
    return prices


def configure_pricing(raw: str) -> None:
    PRICES.clear()
    PRICES.update(parse_prices(raw))


def _usage(result: dict[str, Any]) -> tuple[int, int, float]:
    """Return (input tokens, output tokens, estimated USD) for the first AI message with usage."""
    for message in result.get("messages", []):
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        if usage:
            tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            model = (message.response_metadata or {}).get("model_name", "")
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            return tokens_in, tokens_out, (tokens_in * price_in + tokens_out * price_out) / 1e6
    return 0, 0, 0.0


def _tokens(result: dict[str, Any]) -> tuple[int, int]:
    tokens_in, tokens_out, _ = _usage(result)
    return tokens_in, tokens_out


def _record(name: str, result: dict[str, Any]) -> tuple[int, int, float]:
    if name == "classifier":
        if result.get("confidence") is None:
            CLASSIFIER_FALLBACKS.inc()
        if message_type := result.get("message_type"):
            MESSAGES_CLASSIFIED.labels(message_type).inc()
    elif name == "safety_check" and result.get("safety") == "crisis":
        CRISIS_TRIGGERS.labels(result.get("crisis_category") or "unknown").inc()
    tokens_in, tokens_out, cost = _usage(result)
    if tokens_in or tokens_out:
        LLM_TOKENS.labels(name, "input").inc(tokens_in)
        LLM_TOKENS.labels(name, "output").inc(tokens_out)
    if cost:
        LLM_COST.labels(name).inc(cost)
    return tokens_in, tokens_out, cost


def instrument(name: str, node: Callable[[Any], dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    """Wrap a graph node with timing, metrics and a content-free structured log line."""

    @functools.wraps(node)
    def wrapper(state: Any) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = node(state)
        except Exception:
            NODE_ERRORS.labels(name).inc()
            logger.error("node failed", extra={"node": name, "outcome": "error"})
            raise
        elapsed = time.perf_counter() - start
        NODE_LATENCY.labels(name).observe(elapsed)
        tokens_in, tokens_out, cost = _record(name, result)
        logger.info(
            "node completed",
            extra={
                "node": name,
                "latency_ms": round(elapsed * 1000, 1),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                **({"cost_usd": round(cost, 6)} if cost else {}),
                "outcome": "ok",
            },
        )
        return result

    return wrapper
