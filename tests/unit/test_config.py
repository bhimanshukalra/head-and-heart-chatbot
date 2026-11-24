import pytest
from pydantic import ValidationError

from head_and_heart.config import Settings


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults_and_secret_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.max_retries == 3
    assert "sk-test" not in repr(s)
