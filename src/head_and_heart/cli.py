"""Command-line entrypoint."""

import argparse
import logging
import sqlite3
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from head_and_heart.chat import stream_reply
from head_and_heart.config import get_settings
from head_and_heart.graph import build_graph
from head_and_heart.llm import get_chat_model

EXIT_COMMANDS = {"exit", "quit"}
DISCLAIMER = (
    "Note: this is an AI assistant, not a licensed therapist or medical professional. "
    "If you are in crisis, please contact local emergency services or a crisis line."
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="head-and-heart")
    parser.add_argument("--thread-id", help="Resume a previous conversation by its thread id.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    graph = build_graph(
        classifier_llm=get_chat_model(
            settings, model_name=settings.classifier_model_name, temperature=0
        ),
        therapist_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.7),
        logical_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.2),
        blended_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.5),
        checkpointer=SqliteSaver(conn),
        max_context_tokens=settings.max_context_tokens,
        confidence_threshold=settings.classifier_confidence_threshold,
    )
    thread_id = args.thread_id or str(uuid.uuid4())
    if not args.thread_id:
        print(DISCLAIMER)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print(f"Thread: {thread_id} (resume with --thread-id)")

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
                print("Assistant: ", end="", flush=True)
                for token in stream_reply(graph, user_input, config):
                    print(token, end="", flush=True)
                print()
            except Exception:
                logger.exception("Request failed")
                print("\nAssistant: Sorry, something went wrong. Please try again.")
    finally:
        conn.close()
