from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from head_and_heart.graph import build_graph


def _graph(classifier: Any, make_agent_llm: Any, **kw: Any) -> Any:
    return build_graph(
        classifier_llm=classifier,
        therapist_llm=make_agent_llm("THERAPIST"),
        logical_llm=make_agent_llm("LOGICAL"),
        blended_llm=make_agent_llm("BLENDED"),
        **kw,
    )


def test_emotional_routes_to_therapist(make_classifier_llm: Any, make_agent_llm: Any) -> None:
    g = _graph(make_classifier_llm("emotional"), make_agent_llm)
    out = g.invoke({"messages": [HumanMessage("I feel so alone")]})
    assert out["message_type"] == "emotional"
    assert out["messages"][-1].text == "THERAPIST"


def test_logical_routes_to_logical(make_classifier_llm: Any, make_agent_llm: Any) -> None:
    g = _graph(make_classifier_llm("logical"), make_agent_llm)
    out = g.invoke({"messages": [HumanMessage("What is 2+2?")]})
    assert out["messages"][-1].text == "LOGICAL"


def test_classifier_failure_defaults_to_blended(
    make_classifier_llm: Any, make_agent_llm: Any
) -> None:
    g = _graph(make_classifier_llm(RuntimeError("api down")), make_agent_llm)
    out = g.invoke({"messages": [HumanMessage("hello")]})
    assert out["message_type"] == "mixed"
    assert out["messages"][-1].text == "BLENDED"


def test_mixed_routes_to_blended(make_classifier_llm: Any, make_agent_llm: Any) -> None:
    g = _graph(make_classifier_llm("mixed"), make_agent_llm)
    out = g.invoke({"messages": [HumanMessage("I'm stressed, how do I study?")]})
    assert out["messages"][-1].text == "BLENDED"


def test_low_confidence_routes_to_blended(make_classifier_llm: Any, make_agent_llm: Any) -> None:
    g = _graph(make_classifier_llm("logical", confidence=0.3), make_agent_llm)
    out = g.invoke({"messages": [HumanMessage("hmm")]})
    assert out["message_type"] == "mixed"
    assert out["messages"][-1].text == "BLENDED"


def test_classifier_sees_recent_context(make_classifier_llm: Any, make_agent_llm: Any) -> None:
    clf = make_classifier_llm("emotional")
    g = _graph(clf, make_agent_llm, checkpointer=MemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "c1"}}
    g.invoke({"messages": [HumanMessage("I feel lonely")]}, cfg)
    g.invoke({"messages": [HumanMessage("yes, exactly")]}, cfg)
    transcript = clf.last_input[-1].content
    assert "I feel lonely" in transcript
    assert "LATEST User: yes, exactly" in transcript


def test_history_is_kept_across_turns(make_classifier_llm: Any, make_agent_llm: Any) -> None:
    g = _graph(make_classifier_llm("logical"), make_agent_llm, checkpointer=MemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    g.invoke({"messages": [HumanMessage("first")]}, cfg)
    out = g.invoke({"messages": [HumanMessage("second")]}, cfg)
    assert [m.type for m in out["messages"]] == ["human", "ai", "human", "ai"]


def test_agent_receives_full_history(make_classifier_llm: Any) -> None:
    seen: list[Any] = []

    class Recorder:
        def invoke(self, messages: list[Any]) -> Any:
            from langchain_core.messages import AIMessage

            seen.append(messages)
            return AIMessage("ok")

    g = build_graph(
        classifier_llm=make_classifier_llm("logical"),
        therapist_llm=Recorder(),  # type: ignore[arg-type]
        logical_llm=Recorder(),  # type: ignore[arg-type]
        checkpointer=MemorySaver(),
    )
    cfg: RunnableConfig = {"configurable": {"thread_id": "t2"}}
    g.invoke({"messages": [HumanMessage("my name is Sam")]}, cfg)
    g.invoke({"messages": [HumanMessage("what is my name?")]}, cfg)
    contents = [m.content for m in seen[-1]]
    assert "my name is Sam" in contents


def test_stream_reply_yields_agent_tokens_only(
    make_classifier_llm: Any, make_agent_llm: Any
) -> None:
    from head_and_heart.chat import stream_reply

    g = _graph(make_classifier_llm("emotional"), make_agent_llm, checkpointer=MemorySaver())
    cfg: RunnableConfig = {"configurable": {"thread_id": "s1"}}
    assert "".join(stream_reply(g, "I feel sad", cfg)).strip() == "THERAPIST"


def test_sqlite_checkpoint_persists_across_graph_instances(
    make_classifier_llm: Any, make_agent_llm: Any, tmp_path: Any
) -> None:
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    db = str(tmp_path / "t.sqlite")
    cfg: RunnableConfig = {"configurable": {"thread_id": "p1"}}
    for text, expected in [("first", 2), ("second", 4)]:
        conn = sqlite3.connect(db, check_same_thread=False)
        g = _graph(make_classifier_llm("logical"), make_agent_llm, checkpointer=SqliteSaver(conn))
        out = g.invoke({"messages": [HumanMessage(text)]}, cfg)
        assert len(out["messages"]) == expected
        conn.close()
