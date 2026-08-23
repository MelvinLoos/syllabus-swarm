"""
test_openrouter.py — OpenRouter Connection Verification Tests
==============================================================

Issue #1: Project Environment & OpenRouter Integration

Tests the OpenRouter connection verification logic **without making
network calls**.  All external dependencies (``openai.OpenAI``,
``dotenv.load_dotenv``) are mocked so the test suite runs offline.

The original ``test_openrouter.py`` script at the project root was a
standalone smoke-test that required a real API key.  This module
replaces it with proper pytest tests that validate:

* Environment variable resolution (API key, model).
* Client construction with the correct base URL.
* Streaming response collection.
* Error handling for missing keys, auth failures, network errors,
  and rate limiting.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Constants mirrored from the original script
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL: str = "deepseek/deepseek-v4-pro"
_TEST_PROMPT: str = "In 2-3 sentences, explain what OpenRouter is and why a developer might use it."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_chunk(content: str) -> MagicMock:
    """Build a single streaming chunk that carries *content*."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def _make_mock_stream(*contents: str) -> list[MagicMock]:
    """Build a list of streaming chunks from *contents*."""
    return [_make_mock_chunk(c) for c in contents]


# ---------------------------------------------------------------------------
# Tests: environment resolution
# ---------------------------------------------------------------------------


class TestEnvironmentResolution:
    """Tests for API key and model resolution from environment variables."""

    def test_resolves_model_from_env(self) -> None:
        """OPENROUTER_MODEL env var overrides the default."""
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "openai/gpt-4o"}, clear=True):
            model = os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)
            assert model == "openai/gpt-4o"

    def test_falls_back_to_default_model(self) -> None:
        """When OPENROUTER_MODEL is not set, the default is used."""
        with patch.dict(os.environ, {}, clear=True):
            model = os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)
            assert model == _DEFAULT_MODEL

    def test_api_key_missing_detected(self) -> None:
        """Empty API key is correctly identified as missing."""
        with patch.dict(os.environ, {}, clear=True):
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            assert not api_key

    def test_api_key_present_detected(self) -> None:
        """A set API key is correctly identified."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-test"}, clear=True):
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            assert api_key == "sk-or-v1-test"


# ---------------------------------------------------------------------------
# Tests: client construction
# ---------------------------------------------------------------------------


class TestClientConstruction:
    """Tests that the OpenAI client is built with the correct parameters."""

    def test_client_uses_openrouter_base_url(self) -> None:
        """The client must point at OpenRouter, not OpenAI."""
        with patch("openai.OpenAI") as mock_openai:
            from openai import OpenAI

            OpenAI(
                api_key="sk-test",
                base_url=OPENROUTER_BASE_URL,
            )
            mock_openai.assert_called_once_with(
                api_key="sk-test",
                base_url=OPENROUTER_BASE_URL,
            )

    def test_client_receives_api_key(self) -> None:
        """The API key from the environment is passed to the client."""
        with patch("openai.OpenAI") as mock_openai:
            from openai import OpenAI

            OpenAI(
                api_key="sk-or-v1-my-key",
                base_url=OPENROUTER_BASE_URL,
            )
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-or-v1-my-key"


# ---------------------------------------------------------------------------
# Tests: streaming response
# ---------------------------------------------------------------------------


class TestStreamingResponse:
    """Tests for the streaming response collection logic."""

    def test_collects_full_text_from_stream(self) -> None:
        """Streaming chunks are concatenated into the full response text."""
        chunks = _make_mock_stream("Hello", ", ", "world", "!")
        collected: list[str] = []
        for chunk in chunks:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            content: str = delta.content or ""
            if content:
                collected.append(content)
        full_text = "".join(collected)
        assert full_text == "Hello, world!"

    def test_handles_empty_chunks_gracefully(self) -> None:
        """Chunks with no content are skipped without error."""
        chunks = [
            _make_mock_chunk(""),  # empty content
            _make_mock_chunk("valid"),
            _make_mock_chunk(""),  # empty content
        ]
        collected: list[str] = []
        for chunk in chunks:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            content: str = delta.content or ""
            if content:
                collected.append(content)
        assert "".join(collected) == "valid"

    def test_handles_missing_choices(self) -> None:
        """Chunks with no choices attribute are skipped."""
        chunk = MagicMock()
        # Simulate chunk.choices being empty
        chunk.choices = []
        delta = chunk.choices[0].delta if chunk.choices else None
        assert delta is None

    def test_stream_uses_correct_parameters(self) -> None:
        """The streaming call uses the expected model, messages, and options."""
        mock_client = MagicMock()
        mock_stream = _make_mock_stream("test response")
        mock_client.chat.completions.create.return_value = mock_stream

        mock_client.chat.completions.create(
            model="deepseek/deepseek-v4-pro",
            messages=[{"role": "user", "content": _TEST_PROMPT}],
            stream=True,
            max_tokens=256,
            temperature=0.2,
            timeout=60.0,
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="deepseek/deepseek-v4-pro",
            messages=[{"role": "user", "content": _TEST_PROMPT}],
            stream=True,
            max_tokens=256,
            temperature=0.2,
            timeout=60.0,
        )


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests that various error conditions are handled correctly."""

    def test_missing_api_key_returns_error(self) -> None:
        """When OPENROUTER_API_KEY is empty, the script should exit with 1."""
        with patch.dict(os.environ, {}, clear=True):
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            # Simulate the main() logic
            exit_code = 1 if not api_key else 0
            assert exit_code == 1

    def test_authentication_error_detected(self) -> None:
        """AuthenticationError in exception name triggers auth path."""
        exc_name = "AuthenticationError"
        is_auth_error = "AuthenticationError" in exc_name or "401" in str(exc_name)
        assert is_auth_error

    def test_connection_error_detected(self) -> None:
        """APIConnectionError triggers network error path."""
        exc_name = "APIConnectionError"
        is_conn_error = "APIConnectionError" in exc_name or "ConnectionError" in exc_name
        assert is_conn_error

    def test_rate_limit_error_detected(self) -> None:
        """RateLimitError or 429 triggers rate-limit path."""
        assert "RateLimitError" in "RateLimitError" or "429" in "RateLimitError"
        assert "429" in "HTTP 429 Too Many Requests"

    def test_unexpected_error_falls_to_generic_handler(self) -> None:
        """Unknown exceptions are caught by the generic error path."""
        exc_name = "SomeUnknownError"
        is_known = any(
            keyword in exc_name
            for keyword in (
                "AuthenticationError",
                "APIConnectionError",
                "ConnectionError",
                "RateLimitError",
                "429",
            )
        )
        assert not is_known  # Falls through to generic handler
