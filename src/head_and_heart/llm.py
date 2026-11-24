"""Chat model factory."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from head_and_heart.config import Settings


def get_chat_model(settings: Settings, *, model_name: str, temperature: float) -> BaseChatModel:
    """Build a chat model with timeout and retries taken from settings."""
    return init_chat_model(
        model_name,
        model_provider=settings.llm_provider,
        temperature=temperature,
        timeout=settings.request_timeout_s,
        max_retries=settings.max_retries,
        api_key=settings.groq_api_key.get_secret_value(),
    )
