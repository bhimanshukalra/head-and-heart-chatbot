import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from head_and_heart.api import RateLimiter, create_app, parse_api_keys
from head_and_heart.config import Settings
from head_and_heart.graph import build_graph

HEADERS_A = {"X-API-Key": "ka"}
HEADERS_B = {"X-API-Key": "kb"}


def _settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {"groq_api_key": "x", "api_keys": "alice:ka,bob:kb"}
    return Settings(_env_file=None, **{**base, **kw})


@pytest.fixture
def make_client(make_classifier_llm: Any, make_agent_llm: Any) -> Any:
    def _make(**kw: Any) -> TestClient:
        graph = build_graph(
            classifier_llm=make_classifier_llm("logical"),
            therapist_llm=make_agent_llm("THERAPIST"),
            logical_llm=make_agent_llm("LOGICAL"),
            blended_llm=make_agent_llm("BLENDED"),
            checkpointer=InMemorySaver(),
        )
        return TestClient(create_app(_settings(**kw), graph=graph))

    return _make


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in filter(None, text.split("\n\n")):
        event, data = block.split("\n")
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_health_needs_no_auth(make_client: Any) -> None:
    with make_client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_chat_requires_valid_key(make_client: Any) -> None:
    with make_client() as c:
        assert c.post("/chat", json={"message": "hi"}).status_code == 401
        assert (
            c.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "no"}).status_code == 401
        )


def test_chat_streams_tokens_then_done(make_client: Any) -> None:
    with make_client() as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "t1"}, headers=HEADERS_A)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert "LOGICAL" in "".join(d["text"] for e, d in events if e == "token")
    assert events[-1] == ("done", {"thread_id": "t1"})


def test_crisis_message_gets_fixed_reply(make_client: Any) -> None:
    with make_client(crisis_region="US") as c:
        r = c.post("/chat", json={"message": "I want to kill myself"}, headers=HEADERS_A)
    assert "988" in "".join(d["text"] for e, d in _events(r.text) if e == "token")


def test_validation_errors(make_client: Any) -> None:
    with make_client(max_input_chars=10) as c:
        assert c.post("/chat", json={"message": "x" * 11}, headers=HEADERS_A).status_code == 422
        assert c.post("/chat", json={"message": "   "}, headers=HEADERS_A).status_code == 422
        bad = {"message": "hi", "thread_id": "a:b"}
        assert c.post("/chat", json=bad, headers=HEADERS_A).status_code == 422


def test_rate_limit(make_client: Any) -> None:
    with make_client(rate_limit_per_minute=2) as c:
        codes = [
            c.post("/chat", json={"message": "hi"}, headers=HEADERS_A).status_code for _ in range(3)
        ]
        assert codes == [200, 200, 429]
        # another user is unaffected
        assert c.post("/chat", json={"message": "hi"}, headers=HEADERS_B).status_code == 200


def test_threads_are_isolated_per_user_and_deletable(make_client: Any) -> None:
    with make_client() as c:
        for headers in (HEADERS_A, HEADERS_B):
            c.post("/chat", json={"message": "hi", "thread_id": "same"}, headers=headers)
        graph = c.app.state.graph
        for user in ("alice", "bob"):
            values = graph.get_state({"configurable": {"thread_id": f"{user}:same"}}).values
            assert len(values["messages"]) == 2

        assert c.delete("/threads/same", headers=HEADERS_A).status_code == 204
        assert not graph.get_state({"configurable": {"thread_id": "alice:same"}}).values
        assert graph.get_state({"configurable": {"thread_id": "bob:same"}}).values


def test_metrics_endpoint(make_client: Any) -> None:
    with make_client() as c:
        c.post("/chat", json={"message": "hi"}, headers=HEADERS_A)
        body = c.get("/metrics").text
    assert "messages_classified_total" in body
    assert "node_latency_seconds" in body


def test_startup_requires_keys_unless_auth_disabled() -> None:
    with pytest.raises(ValueError):
        create_app(_settings(api_keys=""))
    create_app(_settings(api_keys="", api_auth_disabled=True))


def test_parse_api_keys() -> None:
    assert parse_api_keys("a:k1, b:k2") == {"k1": "a", "k2": "b"}
    with pytest.raises(ValueError):
        parse_api_keys("nocolon")


def test_rate_limiter_window() -> None:
    now = [0.0]
    limiter = RateLimiter(1, window_s=10, clock=lambda: now[0])
    assert limiter.allow("u")
    assert not limiter.allow("u")
    now[0] = 11
    assert limiter.allow("u")


def test_stream_failure_emits_error_event_without_leaking_details() -> None:
    class Broken:
        async def astream(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("secret internal detail")
            yield  # pragma: no cover

    with TestClient(create_app(_settings(), graph=Broken())) as c:
        r = c.post("/chat", json={"message": "hi"}, headers=HEADERS_A)
    events = _events(r.text)
    assert events[-1][0] == "error"
    assert "secret" not in r.text


def test_delete_rejects_bad_thread_id(make_client: Any) -> None:
    with make_client() as c:
        assert c.delete("/threads/bad%3Aid", headers=HEADERS_A).status_code == 422


@pytest.mark.parametrize("persist", [False, True])
def test_default_lifespan_builds_graph(
    persist: bool,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    make_classifier_llm: Any,
    make_agent_llm: Any,
) -> None:
    from head_and_heart import api

    monkeypatch.setattr(
        api,
        "build_default_graph",
        lambda settings, saver: build_graph(
            classifier_llm=make_classifier_llm("logical"),
            therapist_llm=make_agent_llm("THERAPIST"),
            logical_llm=make_agent_llm("LOGICAL"),
            checkpointer=saver,
        ),
    )
    settings = _settings(persist_history=persist, checkpoint_db_path=str(tmp_path / "t.sqlite"))
    with TestClient(create_app(settings)) as c:
        r = c.post("/chat", json={"message": "hi", "thread_id": "p"}, headers=HEADERS_A)
        assert "LOGICAL" in r.text
        assert c.delete("/threads/p", headers=HEADERS_A).status_code == 204


def test_meta_event_reports_crisis_and_hides_routing_by_default(make_client: Any) -> None:
    with make_client() as c:
        normal = c.post("/chat", json={"message": "hi"}, headers=HEADERS_A)
        crisis = c.post("/chat", json={"message": "I want to kill myself"}, headers=HEADERS_A)
    assert dict(_events(normal.text))["meta"] == {"crisis": False}
    assert dict(_events(crisis.text))["meta"] == {"crisis": True}


def test_debug_flag_adds_routing_details(make_client: Any) -> None:
    with make_client() as c:
        r = c.post("/chat", json={"message": "hi", "debug": True}, headers=HEADERS_A)
    meta = dict(_events(r.text))["meta"]
    assert meta["message_type"] == "logical"
    assert meta["confidence"] == 0.95
