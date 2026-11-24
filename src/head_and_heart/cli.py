"""Command-line entrypoint."""

import argparse
import logging
import sqlite3
import uuid
from contextlib import closing

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from head_and_heart.chat import InputTooLongError, stream_reply, validate_input
from head_and_heart.config import Settings, get_settings
from head_and_heart.graph import build_default_graph
from head_and_heart.logging import configure_logging, thread_id_var

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
    return parser.parse_args()


def delete_thread(settings: Settings, thread_id: str) -> None:
    with closing(sqlite3.connect(settings.checkpoint_db_path)) as conn:
        SqliteSaver(conn).delete_thread(thread_id)
    print(f"Deleted thread {thread_id}.")


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    if args.delete_thread:
        delete_thread(settings, args.delete_thread)
        return
    if args.thread_id and not settings.persist_history:
        raise SystemExit("--thread-id requires PERSIST_HISTORY=true.")

    conn = (
        sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
        if settings.persist_history
        else None
    )
    graph = build_default_graph(settings, SqliteSaver(conn) if conn else InMemorySaver())
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
