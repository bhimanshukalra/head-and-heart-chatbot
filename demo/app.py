"""Streamlit demo for the Head & Heart chatbot.

For demos and internal testing only: Streamlit re-runs this script on every interaction and does
not isolate users. It talks to the HTTP API, so auth, rate limits and safety checks all apply.

Run:  uv sync --extra demo && uv run streamlit run demo/app.py
"""

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from client import DemoError, delete_thread, make_client, stream_chat

DISCLAIMER = (
    "This is an AI assistant, not a licensed therapist or medical professional. "
    "Conversations are kept in memory only and are not saved by this demo."
)
CRISIS_NOTE = (
    "If you are in immediate danger, please contact your local emergency services "
    "or a crisis line right now."
)


@st.cache_resource
def _client(base_url: str) -> Any:
    return make_client(base_url)


def _debug_caption(meta: dict[str, Any]) -> str | None:
    if meta.get("crisis"):
        return "route: crisis response (fixed text, no LLM)"
    if "message_type" in meta:
        confidence = meta.get("confidence")
        shown = "n/a" if confidence is None else f"{confidence:.2f}"
        return f"route: {meta['message_type']} · confidence: {shown}"
    return None


def _render(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("crisis"):
            st.error(CRISIS_NOTE)
        if message.get("caption"):
            st.caption(message["caption"])


def _sidebar() -> tuple[str, str, bool]:
    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("API URL", os.environ.get("DEMO_API_URL", "http://localhost:8000"))
        api_key = st.text_input("API key", os.environ.get("DEMO_API_KEY", ""), type="password")
        debug = st.toggle("Show routing details", value=False)
        if st.button("New conversation"):
            delete_thread(_client(base_url), st.session_state.thread_id, api_key)
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.rerun()
    return base_url, api_key, debug


def _reply(base_url: str, api_key: str, debug: bool, prompt: str) -> None:
    meta: dict[str, Any] = {}

    def tokens() -> Iterator[str]:
        for event in stream_chat(
            _client(base_url), prompt, st.session_state.thread_id, api_key, debug=debug
        ):
            if event.name == "token":
                yield event.data["text"]
            elif event.name == "meta":
                meta.update(event.data)
            elif event.name == "error":
                raise DemoError(event.data.get("message", "Something went wrong."))

    with st.chat_message("assistant"):
        try:
            text = st.write_stream(tokens())
        except DemoError as exc:
            st.error(str(exc))
            return
        crisis = bool(meta.get("crisis"))
        caption = _debug_caption(meta) if debug else None
        if crisis:
            st.error(CRISIS_NOTE)
        if caption:
            st.caption(caption)
    st.session_state.messages.append(
        {"role": "assistant", "content": str(text), "crisis": crisis, "caption": caption}
    )


def main() -> None:
    st.set_page_config(page_title="Head & Heart", page_icon="💬")
    st.title("Head & Heart")
    st.info(DISCLAIMER)
    st.session_state.setdefault("thread_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])

    base_url, api_key, debug = _sidebar()
    for message in st.session_state.messages:
        _render(message)

    if not api_key:
        st.warning("Enter an API key in the sidebar to start chatting.")
    prompt = st.chat_input("Type a message", disabled=not api_key, max_chars=4000)
    if prompt:
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        _render(user_message)
        _reply(base_url, api_key, debug, prompt)


main()
