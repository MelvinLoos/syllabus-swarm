#!/usr/bin/env python3
"""
test_openrouter.py — OpenRouter Connection Verification (Issue #1)
==================================================================

A standalone smoke-test script that verifies the OpenRouter API connection
and streaming capability.  This is **Step 5** of the Quick Start workflow:
after configuring your ``.env`` file, run this script to confirm the
end-to-end pipeline (``.env`` → API key → OpenRouter → streaming response)
is working correctly.

The script:

1. Loads environment variables from ``.env`` using ``python-dotenv``.
2. Connects to `OpenRouter <https://openrouter.ai/api/v1>`_ using the
   ``openai`` Python client pointed at the OpenRouter base URL.
3. Sends a simple test prompt to the model specified by the
   ``OPENROUTER_MODEL`` environment variable (default:
   ``deepseek/deepseek-v4-pro``).
4. Streams the response token-by-token to stdout so the operator can
   watch the model reply in real time.
5. Prints a clear success / failure summary at the end.

Error handling covers:

* Missing ``OPENROUTER_API_KEY`` in ``.env`` (instructs the operator to
  set it).
* Authentication failures (invalid / expired key).
* Network / connection issues (timeout, DNS, unreachable).

Usage
-----

.. code-block:: bash

    python test_openrouter.py
    python test_openrouter.py --model openai/gpt-4o

Environment
-----------

    Requires ``OPENROUTER_API_KEY`` in ``.env`` (copy from ``.env.example``).
    Optionally set ``OPENROUTER_MODEL`` to override the test model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path / import setup — ensure the project root is on sys.path so that
# ``dotenv`` resolves relative to the project even when the script is
# invoked from a different working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Default model when OPENROUTER_MODEL is not set (Issue #1 acceptance criteria).
_DEFAULT_MODEL: str = "deepseek/deepseek-v4-pro"

# A short test prompt — intentionally simple so token-by-token streaming is
# visible and the operator can confirm the model is producing coherent output.
_TEST_PROMPT: str = (
    "In 2-3 sentences, explain what OpenRouter is and why a developer "
    "might use it."
)

_REQUEST_TIMEOUT: float = 60.0  # seconds for the streaming completion call


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------


class _Ansi:
    """Tiny namespace so we can toggle colours off for non-TTY output."""

    _enabled: bool = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False

    @classmethod
    def _code(cls, code: int) -> str:
        return f"\033[{code}m" if cls._enabled else ""

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls._code(32)}{text}{cls._code(0)}"

    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls._code(31)}{text}{cls._code(0)}"

    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls._code(33)}{text}{cls._code(0)}"

    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls._code(36)}{text}{cls._code(0)}"

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls._code(1)}{text}{cls._code(0)}"


C = _Ansi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model() -> str:
    """Return the model ID from ``OPENROUTER_MODEL`` env var or the default."""
    return os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)


def _print_banner(api_key: str, model: str) -> None:
    """Print a startup banner summarising the test configuration."""
    key_status: str = C.green("✓ set") if api_key else C.red("✗ missing")
    print()
    print(C.bold("=" * 64))
    print(C.bold("  test_openrouter.py — OpenRouter Connection Verification"))
    print(C.bold("=" * 64))
    print(f"  Base URL:  {OPENROUTER_BASE_URL}")
    print(f"  API Key:   {key_status}")
    print(f"  Model:     {model}")
    print(f"  Prompt:    \"{_TEST_PROMPT}\"")
    print(f"{'─' * 64}")
    print()

def _stream_response(client: OpenAI, model: str) -> tuple[str, float]:
    """Stream a chat completion and return (full_text, elapsed_seconds).

    Prints tokens to stdout as they arrive so the operator sees the model
    responding in real time.

    Raises
    ------
    openai.AuthenticationError
        If the API key is invalid or expired.
    openai.APIConnectionError
        If the network is unreachable or the request times out.
    openai.APIStatusError
        For other HTTP-level errors (e.g. 429 rate-limit, 5xx server).
    """
    import time

    print(f"  {C.cyan('⏳')} Streaming response from {model}…\n")
    print(f"  {C.bold('Model response:')}")
    print(f"  {'─' * 50}")

    t0: float = time.monotonic()

    # NOTE: we explicitly set max_tokens to a moderate value so the test
    # runs quickly; we only need to confirm streaming works, not get a
    # full-length essay.
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _TEST_PROMPT}],
        stream=True,
        max_tokens=256,
        temperature=0.2,
        timeout=_REQUEST_TIMEOUT,
    )

    collected_parts: list[str] = []
    first_token: bool = False
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue
        content: str = delta.content or ""
        if content:
            if not first_token:
                # Print the first token on the same line as the header.
                sys.stdout.write(f"  {content}")
                sys.stdout.flush()
                first_token = True
            else:
                sys.stdout.write(content)
                sys.stdout.flush()
            collected_parts.append(content)

    elapsed: float = time.monotonic() - t0

    print()
    print(f"  {'─' * 50}")
    print()

    full_text: str = "".join(collected_parts)
    return full_text, elapsed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point.  Returns 0 on success, 1 on failure."""

    # --- Parse optional --model override -----------------------------------
    cli_model: str | None = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            cli_model = sys.argv[idx + 1]
            # Remove the --model flag + value so they don't interfere.
            sys.argv.pop(idx)       # --model
            sys.argv.pop(idx)       # value

    # --- Load .env ---------------------------------------------------------
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

    # --- Resolve API key and model -----------------------------------------
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model: str = cli_model if cli_model else _resolve_model()

    _print_banner(api_key, model)

    # --- 1. Check for missing API key --------------------------------------
    if not api_key:
        print(f"  {C.red(C.bold('❌  FAILED — Missing API Key'))}\n")
        print(f"  The environment variable {C.bold('OPENROUTER_API_KEY')} "
              f"is not set.")
        print()
        print(f"  {C.yellow('To fix this:')}")
        print(f"    1. Sign up for a free account at "
              f"{C.cyan('https://openrouter.ai/')}")
        print(f"    2. Generate an API key from the dashboard.")
        print(f"    3. Copy {C.bold('.env.example')} to {C.bold('.env')}:")
        print(f"         cp .env.example .env")
        print(f"    4. Paste your key into {C.bold('.env')}:")
        print(f"         OPENROUTER_API_KEY=sk-or-v1-...")
        print(f"    5. Run this script again:")
        print(f"         python test_openrouter.py")
        print()
        print(f"  {'─' * 64}")
        return 1

    # --- 2. Build the OpenAI client pointed at OpenRouter ------------------
    client: OpenAI = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    # --- 3. Stream a test completion ---------------------------------------
    try:
        full_text, elapsed = _stream_response(client, model)
    except Exception as exc:
        return _handle_stream_error(exc, api_key)

    # --- 4. Success summary ------------------------------------------------
    token_count: int = len(full_text.split()) if full_text.strip() else 0
    print(f"  {C.green(C.bold('✅  SUCCESS — OpenRouter connection verified!'))}")
    print()
    print(f"  Model:          {model}")
    print(f"  Response time:  {elapsed:.1f}s")
    print(f"  Tokens (words): ~{token_count}")
    print(f"  Streaming:      {C.green('working correctly')}")
    print()
    print(f"  The Quick Start verification is complete.  You can now run "
          f"the syllabus")
    print(f"  generator with:")
    print(f"      python src/main.py \"Your Course Topic\"")
    print()
    print(f"  {'─' * 64}")

    return 0


def _handle_stream_error(exc: Exception, api_key: str) -> int:
    """Handle and report streaming / connection errors.

    Returns the process exit code (always 1 for errors).
    """
    print()
    print(f"  {C.red(C.bold('❌  FAILED — Connection Error'))}")
    print()

    exc_name: str = type(exc).__name__
    exc_msg: str = str(exc)

    # -- Authentication -----------------------------------------------------
    if "AuthenticationError" in exc_name or "401" in exc_msg:
        print(f"  {C.red('Authentication failed — your API key was rejected.')}")
        print()
        print(f"  {C.yellow('To fix this:')}")
        print(f"    1. Verify your key at {C.cyan('https://openrouter.ai/keys')}")
        print(f"    2. Make sure the key in {C.bold('.env')} is copied exactly "
              f"(no extra spaces).")
        print(f"    3. Ensure your OpenRouter account has credits available.")
        print(f"    4. Generate a new key if necessary.")
        if api_key and not api_key.startswith("sk-or-"):
            print()
            print(f"    {C.yellow('⚠')}  Your key does not look like an "
                  f"OpenRouter key (expected prefix: sk-or-v1-…).")

    # -- Network / connection -----------------------------------------------
    elif "APIConnectionError" in exc_name or "ConnectionError" in exc_name:
        print(f"  {C.red('Could not reach OpenRouter — network error.')}")
        print()
        print(f"  {C.yellow('To fix this:')}")
        print(f"    1. Check your internet connection.")
        print(f"    2. Verify {C.cyan('https://openrouter.ai')} is reachable "
              f"in your browser.")
        print(f"    3. If you are behind a corporate proxy, set the "
              f"{C.bold('HTTPS_PROXY')} environment variable.")
        print(f"    4. Check that the base URL is correct: "
              f"{OPENROUTER_BASE_URL}")

    # -- Rate limiting ------------------------------------------------------
    elif "RateLimitError" in exc_name or "429" in exc_msg:
        print(f"  {C.red('Rate-limited by OpenRouter — too many requests.')}")
        print()
        print(f"  {C.yellow('To fix this:')}")
        print(f"    1. Wait a moment and try again.")
        print(f"    2. If you are on the free tier, consider upgrading your "
              f"OpenRouter plan.")

    # -- Generic / unexpected -----------------------------------------------
    else:
        print(f"  {C.red(f'Unexpected error: {exc_name}')}")
        print(f"  {C.red(f'Details: {exc_msg}')}")
        print()
        print(f"  {C.yellow('Possible causes:')}")
        print(f"    • The model may not exist on OpenRouter.")
        print(f"    • Your OpenRouter account may have restrictions on this "
              f"model.")
        print(f"    • OpenRouter may be experiencing an outage — check "
              f"{C.cyan('https://status.openrouter.ai')}")
        print(f"    • Try a different model: "
              f"python test_openrouter.py --model openai/gpt-4o")

    print()
    print(f"  {'─' * 64}")

    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())