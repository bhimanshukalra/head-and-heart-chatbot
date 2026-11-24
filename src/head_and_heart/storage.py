"""Conversation storage helpers: optional encryption at rest and retention-based expiry."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

from head_and_heart.config import Settings

logger = logging.getLogger(__name__)


def make_serde(settings: Settings) -> EncryptedSerializer | None:
    """AES-encrypting serializer when CHECKPOINT_ENCRYPTION_KEY is set, otherwise None."""
    key = settings.checkpoint_encryption_key.get_secret_value()
    return EncryptedSerializer.from_pycryptodome_aes(key=key.encode()) if key else None


def _newest_by_thread(checkpoints: list[Any]) -> dict[str, datetime]:
    newest: dict[str, datetime] = {}
    for item in checkpoints:
        thread_id = item.config["configurable"]["thread_id"]
        ts = datetime.fromisoformat(item.checkpoint["ts"])
        if thread_id not in newest or ts > newest[thread_id]:
            newest[thread_id] = ts
    return newest


def _idle_threads(
    newest: dict[str, datetime], max_age: timedelta, now: datetime | None
) -> list[str]:
    cutoff = (now or datetime.now(UTC)) - max_age
    return [thread for thread, ts in newest.items() if ts < cutoff]


def purge_idle_threads(saver: Any, max_age: timedelta, now: datetime | None = None) -> int:
    """Delete every thread whose newest checkpoint is older than `max_age`. Returns the count."""
    idle = _idle_threads(_newest_by_thread(list(saver.list(None))), max_age, now)
    for thread_id in idle:
        saver.delete_thread(thread_id)
    logger.info("Retention purge removed %d idle conversations", len(idle))
    return len(idle)


async def purge_idle_threads_async(
    saver: Any, max_age: timedelta, now: datetime | None = None
) -> int:
    """Async version of `purge_idle_threads` for the API's async checkpointers."""
    items = [item async for item in saver.alist(None)]
    idle = _idle_threads(_newest_by_thread(items), max_age, now)
    for thread_id in idle:
        await saver.adelete_thread(thread_id)
    logger.info("Retention purge removed %d idle conversations", len(idle))
    return len(idle)
