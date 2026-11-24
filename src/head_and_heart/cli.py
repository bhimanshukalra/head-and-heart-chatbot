"""Command-line entrypoint."""

from head_and_heart.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"Head & Heart chatbot ready (model: {settings.model_name}).")
