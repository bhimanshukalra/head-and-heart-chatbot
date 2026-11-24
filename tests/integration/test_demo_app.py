"""Headless UI tests. Skipped when the optional `demo` extra is not installed."""

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
from demo.client import Event
from streamlit.testing.v1 import AppTest

DEMO_DIR = Path(__file__).parents[2] / "demo"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    sys.modules.pop("client", None)
    import client  # type: ignore[import-not-found]  # the module the script imports

    monkeypatch.setattr(client, "make_client", lambda base_url: object())
    monkeypatch.setattr(client, "delete_thread", lambda *a, **k: None)
    return client


def _run(events: list[Event] | Exception) -> AppTest:
    import client

    def fake_stream(*args: Any, **kwargs: Any) -> Any:
        if isinstance(events, Exception):
            raise events
        yield from events

    client.stream_chat = fake_stream
    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=10).run()
    at.sidebar.text_input[1].set_value("key").run()
    at.chat_input[0].set_value("hello").run()
    return at


def test_shows_disclaimer_and_asks_for_key_first(app: Any) -> None:
    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=10).run()
    assert not at.exception
    assert "not a licensed therapist" in at.info[0].value
    assert at.chat_input[0].disabled
    assert at.warning


def test_streams_a_reply_into_history(app: Any) -> None:
    at = _run(
        [
            Event("token", {"text": "Hi "}),
            Event("token", {"text": "there"}),
            Event("meta", {"crisis": False}),
            Event("done", {"thread_id": "x"}),
        ]
    )
    assert not at.exception
    assert [m["content"] for m in at.session_state.messages] == ["hello", "Hi there"]


def test_crisis_reply_is_called_out(app: Any) -> None:
    at = _run(
        [
            Event("token", {"text": "Call 988"}),
            Event("meta", {"crisis": True}),
            Event("done", {"thread_id": "x"}),
        ]
    )
    assert any("emergency services" in e.value for e in at.error)
    assert at.session_state.messages[-1]["crisis"] is True


def test_service_errors_are_shown_not_stored(app: Any) -> None:
    # The app catches the DemoError class from the module it imported, so raise that one.
    at = _run(app.DemoError("Too many requests."))
    assert any("Too many requests." in e.value for e in at.error)
    assert [m["role"] for m in at.session_state.messages] == ["user"]
