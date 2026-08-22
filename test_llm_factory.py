"""
test_llm_factory.py — Unit tests for the shared LLM factory
============================================================

Issue #5: Per-Agent Model Configuration — Specialized LLMs for Each Agent

Validates the 4-tier fallback chain implemented by
:func:`src.llm_factory.build_llm_for_agent`:

    1. Per-agent override   -> ``AGENT_{ROLE}_{PROPERTY}``
    2. Agent-wide default   -> ``AGENT_DEFAULT_{PROPERTY}``
    3. Legacy globals       -> ``OPENROUTER_MODEL`` / ``AGENT_TEMPERATURE`` /
                               ``AGENT_MAX_TOKENS`` (deprecated, kept for
                               backward compatibility)
    4. Hardcoded defaults   -> values baked into ``llm_factory.py``

Every test isolates the environment with ``unittest.mock.patch.dict`` using
``clear=True`` so that the ``.env`` values loaded at import time (via
``load_dotenv``) cannot leak into the assertions.

Run with::

    python -m unittest test_llm_factory -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.llm_factory import (
    CURRICULUM_ARCHITECT,
    LAB_DEVELOPER,
    OUTPUT_EXPORTER,
    build_llm_for_agent,
    get_effective_config,
)

# ---------------------------------------------------------------------------
# Expected hardcoded defaults (tier 4) — mirrored from src/llm_factory.py.
# NOTE: crewai.LLM strips the provider prefix from model strings (e.g.
# "deepseek/deepseek-r1" becomes "deepseek-r1").  Tests that inspect the
# raw resolved value use get_effective_config(); tests that inspect the
# constructed LLM use the stripped form.
# ---------------------------------------------------------------------------
HARDCODED_MODEL: str = "deepseek/deepseek-r1"
HARDCODED_MODEL_STRIPPED: str = "deepseek-r1"
HARDCODED_TEMPERATURE: float = 0.2
HARDCODED_TOP_P: float = 0.1
HARDCODED_MAX_TOKENS: int = 8192
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

_DUMMY_API_KEY: str = "sk-test-dummy-key-for-unit-tests"
_BASE_ENV: dict[str, str] = {"OPENROUTER_API_KEY": _DUMMY_API_KEY}

ALL_ROLES: tuple[str, ...] = (
    CURRICULUM_ARCHITECT,
    LAB_DEVELOPER,
    OUTPUT_EXPORTER,
)


def _env(*extra: dict[str, str]) -> dict[str, str]:
    """Merge *extra* dicts on top of the base env (API key only)."""
    result: dict[str, str] = dict(_BASE_ENV)
    for d in extra:
        result.update(d)
    return result


class TestModelFallbackChain(unittest.TestCase):
    """Tests for the MODEL property's 4-tier fallback chain."""

    def test_per_agent_override_takes_highest_priority(self) -> None:
        """Tier 1: AGENT_{ROLE}_MODEL wins over default & legacy values."""
        env = _env({
            "AGENT_CURRICULUM_ARCHITECT_MODEL": "openai/per-agent-model",
            "AGENT_DEFAULT_MODEL": "openai/default-model",
            "OPENROUTER_MODEL": "openai/legacy-model",
        })
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            self.assertEqual(config["model"], "openai/per-agent-model")

    def test_fallback_to_agent_default_model(self) -> None:
        """Tier 2: AGENT_DEFAULT_MODEL used when no per-agent override set."""
        env = _env({
            "AGENT_DEFAULT_MODEL": "openai/default-model",
            "OPENROUTER_MODEL": "openai/legacy-model",
        })
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            self.assertEqual(config["model"], "openai/default-model")

    def test_fallback_to_legacy_openrouter_model(self) -> None:
        """Tier 3: legacy OPENROUTER_MODEL used when no per-agent/default."""
        with patch.dict(os.environ, _env({"OPENROUTER_MODEL": "openai/legacy-model"}), clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            self.assertEqual(config["model"], "openai/legacy-model")

    def test_fallback_to_hardcoded_default_model(self) -> None:
        """Tier 4: hardcoded default used when nothing is configured."""
        with patch.dict(os.environ, _env(), clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            self.assertEqual(config["model"], HARDCODED_MODEL)


class TestNumericFallbackChain(unittest.TestCase):
    """Temperature and max_tokens follow the exact same chain as MODEL."""

    def test_temperature_follows_fallback_chain(self) -> None:
        # Tier 1: per-agent override
        with patch.dict(os.environ, _env({
            "AGENT_CURRICULUM_ARCHITECT_TEMPERATURE": "0.7",
            "AGENT_DEFAULT_TEMPERATURE": "0.5",
            "AGENT_TEMPERATURE": "0.3",
        }), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).temperature, 0.7)

        # Tier 2: agent-wide default
        with patch.dict(os.environ, _env({
            "AGENT_DEFAULT_TEMPERATURE": "0.5",
            "AGENT_TEMPERATURE": "0.3",
        }), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).temperature, 0.5)

        # Tier 3: legacy global
        with patch.dict(os.environ, _env({"AGENT_TEMPERATURE": "0.3"}), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).temperature, 0.3)

        # Tier 4: hardcoded default
        with patch.dict(os.environ, _env(), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).temperature, HARDCODED_TEMPERATURE)

    def test_max_tokens_follows_fallback_chain(self) -> None:
        # Tier 1: per-agent override
        with patch.dict(os.environ, _env({
            "AGENT_CURRICULUM_ARCHITECT_MAX_TOKENS": "2048",
            "AGENT_DEFAULT_MAX_TOKENS": "4096",
            "AGENT_MAX_TOKENS": "8192",
        }), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens, 2048)

        # Tier 2: agent-wide default
        with patch.dict(os.environ, _env({
            "AGENT_DEFAULT_MAX_TOKENS": "4096",
            "AGENT_MAX_TOKENS": "8192",
        }), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens, 4096)

        # Tier 3: legacy global
        with patch.dict(os.environ, _env({"AGENT_MAX_TOKENS": "8192"}), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens, 8192)

        # Tier 4: hardcoded default
        with patch.dict(os.environ, _env(), clear=True):
            self.assertEqual(build_llm_for_agent(CURRICULUM_ARCHITECT).max_tokens, HARDCODED_MAX_TOKENS)


class TestBuildLLMNoEnvironment(unittest.TestCase):
    """build_llm_for_agent always returns a usable LLM with only API key set."""

    def test_returns_llm_with_all_defaults_when_only_api_key_set(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            llm = build_llm_for_agent(CURRICULUM_ARCHITECT)
            self.assertIsNotNone(llm)
            self.assertEqual(llm.model, HARDCODED_MODEL_STRIPPED)
            self.assertEqual(llm.temperature, HARDCODED_TEMPERATURE)
            self.assertEqual(llm.top_p, HARDCODED_TOP_P)
            self.assertEqual(llm.max_tokens, HARDCODED_MAX_TOKENS)
            self.assertEqual(llm.base_url, OPENROUTER_BASE_URL)

    def test_all_known_agents_build_for_every_role(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            for role in ALL_ROLES:
                with self.subTest(role=role):
                    llm = build_llm_for_agent(role)
                    self.assertIsNotNone(llm)
                    self.assertEqual(llm.base_url, OPENROUTER_BASE_URL)


class TestBackwardCompatibility(unittest.TestCase):
    """Legacy globals keep existing agents working unchanged."""

    def test_legacy_globals_drive_all_agents_without_per_agent_vars(self) -> None:
        env = _env({
            "OPENROUTER_MODEL": "openai/legacy-model",
            "AGENT_TEMPERATURE": "0.4",
            "AGENT_TOP_P": "0.2",
            "AGENT_MAX_TOKENS": "4096",
        })
        with patch.dict(os.environ, env, clear=True):
            for role in ALL_ROLES:
                with self.subTest(role=role):
                    llm = build_llm_for_agent(role)
                    self.assertEqual(llm.model, "legacy-model")
                    self.assertEqual(llm.temperature, 0.4)
                    self.assertEqual(llm.top_p, 0.2)
                    self.assertEqual(llm.max_tokens, 4096)

    def test_per_agent_vars_do_not_leak_across_agents(self) -> None:
        """Per-agent override for one role does not affect other roles."""
        env = _env({
            "AGENT_CURRICULUM_ARCHITECT_MODEL": "openai/architect-only",
            "OPENROUTER_MODEL": "openai/legacy-model",
        })
        with patch.dict(os.environ, env, clear=True):
            architect = build_llm_for_agent(CURRICULUM_ARCHITECT)
            lab_dev = build_llm_for_agent(LAB_DEVELOPER)
            exporter = build_llm_for_agent(OUTPUT_EXPORTER)

            self.assertEqual(architect.model, "architect-only")
            self.assertEqual(lab_dev.model, "legacy-model")
            self.assertEqual(exporter.model, "legacy-model")


class TestGetEffectiveConfig(unittest.TestCase):
    """get_effective_config() mirrors build_llm_for_agent resolution."""

    def test_effective_config_resolves_the_same_chain(self) -> None:
        env = _env({
            "AGENT_CURRICULUM_ARCHITECT_MODEL": "openai/per-agent-model",
            "AGENT_DEFAULT_TEMPERATURE": "0.6",
        })
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(CURRICULUM_ARCHITECT)
            self.assertEqual(config["role"], CURRICULUM_ARCHITECT)
            self.assertEqual(config["model"], "openai/per-agent-model")
            self.assertEqual(config["temperature"], 0.6)
            self.assertEqual(config["max_tokens"], HARDCODED_MAX_TOKENS)
            self.assertEqual(config["base_url"], OPENROUTER_BASE_URL)

    def test_effective_config_matches_built_llm(self) -> None:
        env = _env({"AGENT_DEFAULT_MODEL": "openai/default-model"})
        with patch.dict(os.environ, env, clear=True):
            config = get_effective_config(LAB_DEVELOPER)
            llm = build_llm_for_agent(LAB_DEVELOPER)
            self.assertEqual(config["model"], "openai/default-model")
            self.assertEqual(llm.model, "default-model")
            self.assertEqual(llm.temperature, config["temperature"])
            self.assertEqual(llm.top_p, config["top_p"])
            self.assertEqual(llm.max_tokens, config["max_tokens"])


class TestInputValidation(unittest.TestCase):
    """build_llm_for_agent rejects invalid agent_role inputs."""

    def test_empty_role_raises_value_error(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            with self.assertRaises(ValueError):
                build_llm_for_agent("")

    def test_non_string_role_raises_value_error(self) -> None:
        with patch.dict(os.environ, _env(), clear=True):
            with self.assertRaises(ValueError):
                build_llm_for_agent(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
