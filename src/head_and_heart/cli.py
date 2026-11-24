"""Command-line entrypoint."""

import logging
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from head_and_heart.config import get_settings
from head_and_heart.graph import build_graph
from head_and_heart.llm import get_chat_model

EXIT_COMMANDS = {"exit", "quit"}


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    graph = build_graph(
        classifier_llm=get_chat_model(
            settings, model_name=settings.classifier_model_name, temperature=0
        ),
        therapist_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.7),
        logical_llm=get_chat_model(settings, model_name=settings.model_name, temperature=0.2),
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

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
            result = graph.invoke({"messages": [HumanMessage(user_input)]}, config)
        except Exception:
            logging.getLogger(__name__).exception("Request failed")
            print("Assistant: Sorry, something went wrong. Please try again.")
            continue
        print(f"Assistant: {result['messages'][-1].text}")
