"""Small SSE client for the chat API. Used by the Streamlit demo and tested without a UI."""

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx


class DemoError(Exception):
    """An error with a message that is safe to show to the person using the demo."""


@dataclass(frozen=True)
class Event:
    name: str
    data: dict[str, Any]


def make_client(base_url: str, timeout_s: float = 60.0) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=5.0))


def _explain(response: httpx.Response) -> str:
    if response.status_code == 401:
        return "The API key was rejected. Check the key in the sidebar."
    if response.status_code == 429:
        return "Too many requests. Please wait a minute and try again."
    if response.status_code == 422:
        detail = response.json().get("detail") if response.content else None
        return detail if isinstance(detail, str) else "That message was not accepted."
    return f"The service returned an unexpected error ({response.status_code})."


def stream_chat(
    client: httpx.Client,
    message: str,
    thread_id: str,
    api_key: str,
    *,
    debug: bool = False,
) -> Iterator[Event]:
    """POST /chat and yield each server-sent event. Raises DemoError for friendly failures."""
    body = {"message": message, "thread_id": thread_id, "debug": debug}
    try:
        with client.stream("POST", "/chat", json=body, headers={"X-API-Key": api_key}) as response:
            if response.status_code != 200:
                response.read()
                raise DemoError(_explain(response))
            event: str | None = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line.removeprefix("event: ")
                elif line.startswith("data: ") and event:
                    yield Event(event, json.loads(line.removeprefix("data: ")))
                    event = None
    except httpx.HTTPError as exc:
        raise DemoError("Could not reach the chat service. Is it running?") from exc


def delete_thread(client: httpx.Client, thread_id: str, api_key: str) -> None:
    """Best-effort deletion of the server-side thread."""
    with contextlib.suppress(httpx.HTTPError):
        client.delete(f"/threads/{thread_id}", headers={"X-API-Key": api_key})
