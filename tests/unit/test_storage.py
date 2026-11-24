import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from head_and_heart.config import Settings
from head_and_heart.graph import build_graph
from head_and_heart.storage import make_serde, purge_idle_threads, purge_idle_threads_async

SECRET = "my-very-private-sentence"
KEY = "k" * 32


def _cfg(thread: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}}


@pytest.fixture
def graph_for(make_classifier_llm: Any, make_agent_llm: Any) -> Any:
    def _make(saver: Any) -> Any:
        return build_graph(
            classifier_llm=make_classifier_llm("logical"),
            therapist_llm=make_agent_llm("T"),
            logical_llm=make_agent_llm("L"),
            checkpointer=saver,
        )

    return _make


def test_purge_removes_only_idle_threads(graph_for: Any) -> None:
    saver = InMemorySaver()
    graph = graph_for(saver)
    graph.invoke({"messages": [HumanMessage("a")]}, _cfg("t1"))
    graph.invoke({"messages": [HumanMessage("b")]}, _cfg("t2"))

    soon = datetime.now(UTC) + timedelta(days=1)
    assert purge_idle_threads(saver, timedelta(days=5), now=soon) == 0  # nothing is 5 days idle yet
    later = datetime.now(UTC) + timedelta(days=10)
    assert purge_idle_threads(saver, timedelta(days=5), now=later) == 2
    assert not graph.get_state(_cfg("t1")).values


async def test_async_purge(graph_for: Any) -> None:
    saver = InMemorySaver()
    graph_for(saver).invoke({"messages": [HumanMessage("a")]}, _cfg("t1"))
    later = datetime.now(UTC) + timedelta(days=10)
    assert await purge_idle_threads_async(saver, timedelta(days=5), now=later) == 1
    assert await purge_idle_threads_async(saver, timedelta(days=5), now=later) == 0


def _raw_bytes(db: str) -> bytes:
    conn = sqlite3.connect(db)
    try:
        dump = "\n".join(conn.iterdump())
        blobs = b"".join(
            bytes(v)
            for table in ("checkpoints", "writes")
            for row in conn.execute(f"SELECT * FROM {table}")
            for v in row
            if isinstance(v, (bytes, memoryview))
        )
    finally:
        conn.close()
    return dump.encode() + blobs


def test_encryption_hides_content_and_round_trips(graph_for: Any, tmp_path: Any) -> None:
    serde = EncryptedSerializer.from_pycryptodome_aes(key=KEY.encode())
    enc_db, plain_db = str(tmp_path / "enc.sqlite"), str(tmp_path / "plain.sqlite")

    for db, s in ((enc_db, serde), (plain_db, None)):
        conn = sqlite3.connect(db, check_same_thread=False)
        graph_for(SqliteSaver(conn, serde=s)).invoke(
            {"messages": [HumanMessage(SECRET)]}, _cfg("t1")
        )
        conn.close()

    assert SECRET.encode() in _raw_bytes(plain_db)  # control: plaintext is visible without a key
    assert SECRET.encode() not in _raw_bytes(enc_db)

    conn = sqlite3.connect(enc_db, check_same_thread=False)
    state = graph_for(SqliteSaver(conn, serde=serde)).get_state(_cfg("t1"))
    assert state.values["messages"][0].content == SECRET
    conn.close()


def test_make_serde_follows_settings() -> None:
    assert make_serde(Settings(_env_file=None, groq_api_key="x")) is None
    on = Settings(_env_file=None, groq_api_key="x", checkpoint_encryption_key=KEY)
    assert isinstance(make_serde(on), EncryptedSerializer)


def test_encryption_key_length_is_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, groq_api_key="x", checkpoint_encryption_key="too-short")


async def test_retention_loop_purges_repeatedly_and_survives_errors() -> None:
    import asyncio

    from head_and_heart.api import _retention_loop

    calls = {"n": 0}

    class FlakySaver:
        async def alist(self, config: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return
            yield  # pragma: no cover

    task = asyncio.create_task(_retention_loop(FlakySaver(), timedelta(days=1), 0))
    await asyncio.sleep(0.05)
    task.cancel()
    assert calls["n"] >= 2  # kept going after the first failure
