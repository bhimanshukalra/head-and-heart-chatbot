import sys
from collections.abc import Iterator
from typing import Any

import pytest

from head_and_heart import cli
from head_and_heart.config import Settings
from head_and_heart.graph import build_graph


def _settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, groq_api_key="x", **kw)


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch, make_classifier_llm: Any, make_agent_llm: Any) -> Any:
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)

    def _run(
        inputs: list[str], argv: list[str] | None = None, graph: Any = None, **kw: Any
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["head-and-heart", *(argv or [])])
        monkeypatch.setattr(cli, "get_settings", lambda: _settings(**kw))
        monkeypatch.setattr(
            cli,
            "build_default_graph",
            lambda settings, saver: (
                graph
                or build_graph(
                    classifier_llm=make_classifier_llm("logical"),
                    therapist_llm=make_agent_llm("THERAPIST"),
                    logical_llm=make_agent_llm("LOGICAL"),
                    checkpointer=saver,
                )
            ),
        )
        feed: Iterator[str] = iter(inputs)

        def fake_input(_prompt: str = "") -> str:
            try:
                return next(feed)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr("builtins.input", fake_input)
        cli.main()

    return _run


def test_one_turn_then_exit(run_cli: Any, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(["hello", "exit"])
    out = capsys.readouterr().out
    assert "LOGICAL" in out
    assert "not saved" in out
    assert "not a licensed therapist" in out


def test_blank_and_too_long_input(run_cli: Any, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(["", "x" * 20, "quit"], max_input_chars=10)
    assert "too long" in capsys.readouterr().out


def test_eof_exits_cleanly(run_cli: Any, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli([])  # an exhausted input feed raises EOFError
    assert "Bye" in capsys.readouterr().out


def test_graph_failure_is_reported_and_loop_continues(
    run_cli: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    class Broken:
        def stream(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("boom")

    run_cli(["hi", "exit"], graph=Broken())
    assert "something went wrong" in capsys.readouterr().out.lower()


def test_thread_id_requires_persistence(run_cli: Any) -> None:
    with pytest.raises(SystemExit):
        run_cli([], argv=["--thread-id", "abc"])


def test_persistent_session_and_delete(run_cli: Any, tmp_path: Any, capsys: Any) -> None:
    db = str(tmp_path / "chat.sqlite")
    run_cli(["hello", "exit"], persist_history=True, checkpoint_db_path=db)
    assert "Thread:" in capsys.readouterr().out
    run_cli([], argv=["--delete-thread", "abc"], persist_history=True, checkpoint_db_path=db)
    assert "Deleted thread abc" in capsys.readouterr().out
