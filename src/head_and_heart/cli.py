"""Command-line entrypoint."""

import argparse
import logging
import sqlite3
import uuid
from contextlib import closing
from datetime import timedelta

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from head_and_heart.chat import InputTooLongError, stream_reply, validate_input
from head_and_heart.config import Settings, get_settings
from head_and_heart.graph import build_default_graph
from head_and_heart.logging import configure_logging, thread_id_var
from head_and_heart.storage import make_serde, purge_idle_threads

EXIT_COMMANDS = {"exit", "quit"}
DISCLAIMER = (
    "Note: this is an AI assistant, not a licensed therapist or medical professional. "
    "If you are in crisis, please contact local emergency services or a crisis line."
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="head-and-heart")
    parser.add_argument("--thread-id", help="Resume a stored conversation (needs PERSIST_HISTORY).")
    parser.add_argument("--delete-thread", metavar="ID", help="Delete a stored conversation.")
    parser.add_argument(
        "--purge-older-than",
        type=int,
        metavar="DAYS",
        help="Delete stored conversations idle for more than DAYS days, then exit.",
    )
    return parser.parse_args()


def delete_thread(settings: Settings, thread_id: str) -> None:
    with closing(sqlite3.connect(settings.checkpoint_db_path)) as conn:
        SqliteSaver(conn).delete_thread(thread_id)
    print(f"Deleted thread {thread_id}.")


def purge_old(settings: Settings, days: int) -> int:
    with closing(sqlite3.connect(settings.checkpoint_db_path)) as conn:
        return purge_idle_threads(
            SqliteSaver(conn, serde=make_serde(settings)), timedelta(days=days)
        )


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    if args.delete_thread:
        delete_thread(settings, args.delete_thread)
        return
    if args.purge_older_than is not None:
        print(f"Deleted {purge_old(settings, args.purge_older_than)} idle conversations.")
        return
    if args.thread_id and not settings.persist_history:
        raise SystemExit("--thread-id requires PERSIST_HISTORY=true.")

    conn = (
        sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
        if settings.persist_history
        else None
    )
    if conn and settings.retention_days > 0:
        saver = SqliteSaver(conn, serde=make_serde(settings))
        purge_idle_threads(saver, timedelta(days=settings.retention_days))
    graph = build_default_graph(
        settings, SqliteSaver(conn, serde=make_serde(settings)) if conn else InMemorySaver()
    )
    thread_id = args.thread_id or str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    thread_id_var.set(thread_id)
    if settings.persist_history:
        print(f"Thread: {thread_id} (resume with --thread-id; delete with --delete-thread)")
    else:
        print("Conversation is not saved (set PERSIST_HISTORY=true to keep it).")
    if not args.thread_id:
        print(DISCLAIMER)

    try:
        while True:
            try:
                user_input = input("Message: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye")
                return
            if not user_input:
                continue
            if user_input.lower() in EXIT_COMMANDS:
                print("Bye")
                return
            try:
                user_input = validate_input(user_input, settings.max_input_chars)
            except InputTooLongError as exc:
                print(f"Assistant: {exc}")
                continue
            try:
                print("Assistant: ", end="", flush=True)
                for token in stream_reply(graph, user_input, config):
                    print(token, end="", flush=True)
                print()
            except Exception:
                logger.exception("Request failed")
                print("\nAssistant: Sorry, something went wrong. Please try again.")
    finally:
        if conn:
            conn.close()
