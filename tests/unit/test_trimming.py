from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from head_and_heart.nodes.agents import make_logical


class Recorder:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.seen = messages
        return AIMessage("ok")


def test_old_messages_are_dropped_but_latest_kept() -> None:
    rec = Recorder()
    agent = make_logical(rec, max_tokens=50)  # type: ignore[arg-type]
    history: list[Any] = []
    for i in range(20):
        history += [HumanMessage(f"question number {i} " * 3), AIMessage(f"answer {i} " * 3)]
    history.append(HumanMessage("latest question"))
    agent({"messages": history})
    contents = [m.content for m in rec.seen]
    assert contents[-1] == "latest question"
    assert len(rec.seen) < len(history)
    assert rec.seen[1].type == "human"  # trimmed history starts on a human turn


def test_latest_message_kept_when_over_budget() -> None:
    rec = Recorder()
    agent = make_logical(rec, max_tokens=1)  # type: ignore[arg-type]
    agent({"messages": [HumanMessage("a very long message " * 50)]})
    assert rec.seen[-1].type == "human"
