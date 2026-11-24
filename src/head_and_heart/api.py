"""FastAPI service: streaming chat over SSE with auth, rate limiting and thread isolation."""

import json
import logging
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from head_and_heart.chat import InputTooLongError, astream_reply, validate_input
from head_and_heart.config import Settings, get_settings
from head_and_heart.graph import build_default_graph
from head_and_heart.logging import configure_logging, thread_id_var
from head_and_heart.observability import CHAT_REQUESTS

logger = logging.getLogger(__name__)

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
GENERIC_ERROR = "Something went wrong. Please try again."


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)
    debug: bool = Field(default=False, description="Include routing details in the meta event.")


class RateLimiter:
    """Sliding-window limiter, per process. Use a shared store or gateway for multiple replicas."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parse 'user:key,user2:key2' into {key: user}. Raises ValueError on bad input."""
    keys: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        user, sep, key = pair.partition(":")
        if not sep or not user or not key:
            raise ValueError("API_KEYS must look like 'user:key,user2:key2'.")
        keys[key] = user
    return keys


async def _meta(graph: Any, config: RunnableConfig, debug: bool) -> dict[str, Any] | None:
    """Non-sensitive facts about how the turn was handled, read from the checkpoint."""
    try:
        values = (await graph.aget_state(config)).values
    except Exception:
        logger.warning("Could not read turn metadata", exc_info=True)
        return None
    meta: dict[str, Any] = {"crisis": values.get("safety") == "crisis"}
    if debug:
        meta["message_type"] = values.get("message_type")
        meta["confidence"] = values.get("confidence")
    return meta


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(settings: Settings, graph: Any | None = None) -> FastAPI:
    """Build the app. Pass `graph` to inject a prebuilt graph (used by tests)."""
    keys = parse_api_keys(settings.api_keys.get_secret_value())
    if not keys and not settings.api_auth_disabled:
        raise ValueError("Set API_KEYS, or API_AUTH_DISABLED=true for local development.")
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if graph is not None:
                app.state.graph = graph
            else:
                saver = (
                    await stack.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path)
                    )
                    if settings.persist_history
                    else InMemorySaver()
                )
                app.state.graph = build_default_graph(settings, saver)
            yield

    app = FastAPI(title="Head & Heart chatbot", lifespan=lifespan)

    async def authenticate(x_api_key: str | None = Header(default=None)) -> str:
        if settings.api_auth_disabled:
            return "anonymous"
        if x_api_key:
            for key, user in keys.items():
                if secrets.compare_digest(key, x_api_key):
                    return user
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        # No message content is recorded. Restrict access at the network level.
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/chat")
    async def chat(
        body: ChatRequest, request: Request, user: str = Depends(authenticate)
    ) -> StreamingResponse:
        if not limiter.allow(user):
            CHAT_REQUESTS.labels("rate_limited").inc()
            raise HTTPException(
                status_code=429, detail="Too many requests.", headers={"Retry-After": "60"}
            )
        try:
            message = validate_input(body.message, settings.max_input_chars)
        except InputTooLongError as exc:
            CHAT_REQUESTS.labels("rejected").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not message:
            CHAT_REQUESTS.labels("rejected").inc()
            raise HTTPException(status_code=422, detail="Message must not be blank.")

        thread_id = body.thread_id or uuid.uuid4().hex
        # The user id is part of the key, so users can never read each other's threads.
        config: RunnableConfig = {"configurable": {"thread_id": f"{user}:{thread_id}"}}
        current_graph = request.app.state.graph

        async def events() -> AsyncIterator[str]:
            thread_id_var.set(thread_id)
            try:
                async for text in astream_reply(current_graph, message, config):
                    yield _sse("token", {"text": text})
                if (meta := await _meta(current_graph, config, body.debug)) is not None:
                    yield _sse("meta", meta)
                yield _sse("done", {"thread_id": thread_id})
                CHAT_REQUESTS.labels("ok").inc()
            except Exception:
                logger.exception("Chat request failed")
                CHAT_REQUESTS.labels("error").inc()
                yield _sse("error", {"message": GENERIC_ERROR})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/threads/{thread_id}", status_code=204)
    async def delete_thread(
        thread_id: str, request: Request, user: str = Depends(authenticate)
    ) -> Response:
        if not _valid_thread_id(thread_id):
            raise HTTPException(status_code=422, detail="Invalid thread id.")
        await request.app.state.graph.checkpointer.adelete_thread(f"{user}:{thread_id}")
        return Response(status_code=204)

    return app


def _valid_thread_id(thread_id: str) -> bool:
    return re.fullmatch(THREAD_ID_PATTERN, thread_id) is not None


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory head_and_heart.api:app_factory`."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return create_app(settings)


__all__ = ["RateLimiter", "app_factory", "create_app", "parse_api_keys"]
