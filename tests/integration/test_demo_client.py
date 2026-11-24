from typing import Any

import httpx
import pytest
from demo.client import DemoError, delete_thread, stream_chat
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from head_and_heart.api import create_app
from head_and_heart.config import Settings
from head_and_heart.graph import build_graph


@pytest.fixture
def api(make_classifier_llm: Any, make_agent_llm: Any) -> Any:
    graph = build_graph(
        classifier_llm=make_classifier_llm("logical"),
        therapist_llm=make_agent_llm("THERAPIST"),
        logical_llm=make_agent_llm("LOGICAL"),
        checkpointer=InMemorySaver(),
    )
    settings = Settings(
        _env_file=None, groq_api_key="x", api_keys="alice:ka", rate_limit_per_minute=3
    )
    with TestClient(create_app(settings, graph=graph)) as client:
        yield client


def test_events_are_parsed_in_order(api: Any) -> None:
    events = list(stream_chat(api, "hi", "t1", "ka", debug=True))
    assert [e.name for e in events][-2:] == ["meta", "done"]
    assert "LOGICAL" in "".join(e.data["text"] for e in events if e.name == "token")
    assert events[-2].data["message_type"] == "logical"
    assert events[-1].data == {"thread_id": "t1"}


def test_crisis_is_flagged_in_meta(api: Any) -> None:
    events = list(stream_chat(api, "I want to kill myself", "t2", "ka"))
    assert next(e for e in events if e.name == "meta").data == {"crisis": True}


def test_bad_key_gives_friendly_error(api: Any) -> None:
    with pytest.raises(DemoError, match="API key"):
        list(stream_chat(api, "hi", "t3", "wrong"))


def test_rate_limit_gives_friendly_error(api: Any) -> None:
    for _ in range(3):
        list(stream_chat(api, "hi", "t4", "ka"))
    with pytest.raises(DemoError, match="Too many"):
        list(stream_chat(api, "hi", "t4", "ka"))


def test_rejected_message_surfaces_server_reason(api: Any) -> None:
    with pytest.raises(DemoError, match="too long"):
        list(stream_chat(api, "x" * 5000, "t5", "ka"))


def test_unreachable_service_gives_friendly_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(refuse))
    with pytest.raises(DemoError, match="Could not reach"):
        list(stream_chat(client, "hi", "t6", "ka"))


def test_delete_thread_removes_server_state(api: Any) -> None:
    list(stream_chat(api, "hi", "t7", "ka"))
    graph = api.app.state.graph
    assert graph.get_state({"configurable": {"thread_id": "alice:t7"}}).values
    delete_thread(api, "t7", "ka")
    assert not graph.get_state({"configurable": {"thread_id": "alice:t7"}}).values
