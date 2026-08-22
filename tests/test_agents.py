"""
test_agents.py — Tests for agent factory functions
===================================================

Validates that ``create_curriculum_architect`` and
``create_lab_developer`` produce correctly-configured CrewAI Agent
objects with the expected roles, goals, backstories, and settings.

All tests mock ``build_llm_for_agent`` so no network calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, LLM

from src.agents.curriculum_architect import (
    create_curriculum_architect,
    get_architect,
)
from src.agents.lab_developer import (
    create_lab_developer,
    get_lab_developer,
)
from src.llm_factory import CURRICULUM_ARCHITECT, LAB_DEVELOPER


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """A MagicMock spec'd to crewai.LLM."""
    llm = MagicMock(spec=LLM)
    llm.model = "deepseek-v4-pro"
    llm.temperature = 0.2
    llm.top_p = 0.1
    llm.max_tokens = 8192
    llm.base_url = "https://openrouter.ai/api/v1"
    return llm


# ===================================================================
# create_curriculum_architect
# ===================================================================


class TestCreateCurriculumArchitect:
    """Tests for the Curriculum Architect agent factory."""

    def test_role_contains_curriculum_architect(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the Curriculum Architect."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_curriculum_architect()
            assert "Curriculum Architect" in agent.role
            mock_build.assert_called_once_with(CURRICULUM_ARCHITECT)

    def test_goal_contains_humanics_literacies(self, mock_llm: MagicMock) -> None:
        """The goal references all three Humanics literacies."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Technological Literacy" in agent.goal
            assert "Data Literacy" in agent.goal
            assert "Human Literacy" in agent.goal

    def test_backstory_mentions_joseph_aoun(self, mock_llm: MagicMock) -> None:
        """The backstory references Joseph Aoun and Robot-Proof."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Joseph Aoun" in agent.backstory
            assert "Robot-Proof" in agent.backstory

    def test_backstory_mentions_humanics(self, mock_llm: MagicMock) -> None:
        """The backstory references the Humanics framework."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Humanics" in agent.backstory

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        """The agent does not allow delegation."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.allow_delegation is False

    def test_max_iter_is_five(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 5."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.max_iter == 5

    def test_max_rpm_is_twenty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 20."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.max_rpm == 20

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent"
        ) as mock_build:
            agent = create_curriculum_architect(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_goal_mentions_markdown_output(self, mock_llm: MagicMock) -> None:
        """The goal specifies structured Markdown output."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "Markdown" in agent.goal

    def test_backstory_mentions_backward_design(self, mock_llm: MagicMock) -> None:
        """The backstory references backward design methodology."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_curriculum_architect()
            assert "backward design" in agent.backstory.lower()


# ===================================================================
# create_lab_developer
# ===================================================================


class TestCreateLabDeveloper:
    """Tests for the Lab & Project Developer agent factory."""

    def test_role_contains_lab_developer(self, mock_llm: MagicMock) -> None:
        """The agent's role identifies it as the Lab Developer."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ) as mock_build:
            agent = create_lab_developer()
            assert "Lab & Project Developer" in agent.role
            mock_build.assert_called_once_with(LAB_DEVELOPER)

    def test_goal_contains_three_tiers(self, mock_llm: MagicMock) -> None:
        """The goal references all three lab tiers."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Tier 1 — Foundations" in agent.goal
            assert "Tier 2 — Application" in agent.goal
            assert "Tier 3 — Architecture" in agent.goal

    def test_goal_contains_humanics_literacies(self, mock_llm: MagicMock) -> None:
        """The goal references all three Humanics literacies."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Technological [T]" in agent.goal
            assert "Data [D]" in agent.goal
            assert "Human [H]" in agent.goal

    def test_backstory_mentions_software_engineer(self, mock_llm: MagicMock) -> None:
        """The backstory establishes the agent as a seasoned engineer."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "software engineer" in agent.backstory.lower()

    def test_backstory_mentions_docker(self, mock_llm: MagicMock) -> None:
        """The backstory references Docker as a learning objective."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Docker" in agent.backstory

    def test_backstory_mentions_humanics(self, mock_llm: MagicMock) -> None:
        """The backstory references the Humanics framework."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Humanics" in agent.backstory

    def test_allow_delegation_is_false(self, mock_llm: MagicMock) -> None:
        """The agent does not allow delegation."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.allow_delegation is False

    def test_max_iter_is_seven(self, mock_llm: MagicMock) -> None:
        """The agent has max_iter set to 7 (higher than architect for complex labs)."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.max_iter == 7

    def test_max_rpm_is_twenty(self, mock_llm: MagicMock) -> None:
        """The agent has max_rpm set to 20."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.max_rpm == 20

    def test_verbose_defaults_to_false(self, mock_llm: MagicMock) -> None:
        """verbose is False by default."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert agent.verbose is False

    def test_verbose_can_be_enabled(self, mock_llm: MagicMock) -> None:
        """verbose can be set to True."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer(verbose=True)
            assert agent.verbose is True

    def test_explicit_llm_bypasses_factory(self, mock_llm: MagicMock) -> None:
        """When an LLM is passed explicitly, the factory is not called."""
        custom_llm = MagicMock(spec=LLM)
        with patch(
            "src.agents.lab_developer.build_llm_for_agent"
        ) as mock_build:
            agent = create_lab_developer(llm=custom_llm)
            assert agent.llm is custom_llm
            mock_build.assert_not_called()

    def test_goal_mentions_docker_and_ci_cd(self, mock_llm: MagicMock) -> None:
        """The goal references Docker, CI/CD, and observability for Tier 3."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "Docker" in agent.goal
            assert "CI/CD" in agent.goal

    def test_backstory_mentions_polyglot(self, mock_llm: MagicMock) -> None:
        """The backstory establishes the agent as a polyglot developer."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            agent = create_lab_developer()
            assert "polyglot" in agent.backstory.lower()


# ===================================================================
# Module singletons
# ===================================================================


class TestModuleSingletons:
    """Tests for the lazy singleton accessors."""

    def test_get_architect_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_architect returns a valid Agent."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            # Reset the singleton before test
            import src.agents.curriculum_architect as ca
            ca._architect_instance = None

            agent = get_architect()
            assert isinstance(agent, Agent)
            assert "Curriculum Architect" in agent.role

    def test_get_architect_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_architect twice returns the same instance."""
        with patch(
            "src.agents.curriculum_architect.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.curriculum_architect as ca
            ca._architect_instance = None

            a1 = get_architect()
            a2 = get_architect()
            assert a1 is a2

    def test_get_lab_developer_returns_agent(self, mock_llm: MagicMock) -> None:
        """get_lab_developer returns a valid Agent."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.lab_developer as ld
            ld._lab_dev_instance = None

            agent = get_lab_developer()
            assert isinstance(agent, Agent)
            assert "Lab & Project Developer" in agent.role

    def test_get_lab_developer_is_idempotent(self, mock_llm: MagicMock) -> None:
        """Calling get_lab_developer twice returns the same instance."""
        with patch(
            "src.agents.lab_developer.build_llm_for_agent",
            return_value=mock_llm,
        ):
            import src.agents.lab_developer as ld
            ld._lab_dev_instance = None

            a1 = get_lab_developer()
            a2 = get_lab_developer()
            assert a1 is a2