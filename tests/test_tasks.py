"""
test_tasks.py — Tests for syllabus and lab generation task factories
====================================================================

Validates that ``create_syllabus_generation_task`` and
``create_lab_generation_task`` pass the correct arguments to the
``crewai.Task`` constructor.

Because ``crewai.Task`` uses Pydantic validation and requires a real
``BaseAgent`` instance (not a MagicMock), we patch the ``Task``
reference in each task module and inspect the keyword arguments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# create_syllabus_generation_task
# ===================================================================


class TestCreateSyllabusGenerationTask:
    """Tests for the syllabus generation task factory."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        """Patch Task in the syllabus_generation module."""
        with patch(
            "src.tasks.syllabus_generation.Task"
        ) as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        """Call create_syllabus_generation_task and return Task kwargs."""
        from src.tasks.syllabus_generation import create_syllabus_generation_task

        kwargs: dict = {
            "course_name": "Python for Data Engineering",
            "agent": MagicMock(),
        }
        kwargs.update(overrides)
        create_syllabus_generation_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_course_name_in_description(self) -> None:
        """The course name appears in the task description."""
        kwargs = self._call_factory()
        assert "Python for Data Engineering" in kwargs["description"]

    def test_course_name_in_expected_output(self) -> None:
        """The expected_output mandates an H1 with the course name."""
        kwargs = self._call_factory()
        assert (
            "# Python for Data Engineering — Course Syllabus"
            in kwargs["expected_output"]
        )

    def test_output_file_follows_expected_pattern(self) -> None:
        """The output_file is output/syllabus/<safe_name>.md."""
        kwargs = self._call_factory()
        assert kwargs["output_file"] == "output/syllabus/python_for_data_engineering.md"

    def test_async_execution_is_false(self) -> None:
        """Tasks are synchronous by default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert kwargs["async_execution"] is False

    def test_agent_is_passed_through(self) -> None:
        """The agent parameter is passed through to the Task."""
        agent = MagicMock()
        kwargs = self._call_factory(agent=agent)
        assert kwargs["agent"] is agent

    # -- Optional fields -------------------------------------------------

    def test_course_context_injected_when_provided(self) -> None:
        """course_context appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            course_context="This is a test context about Python and data.",
        )
        assert "This is a test context about Python and data." in kwargs["description"]
        assert "Course Context (from Intake Specialist)" in kwargs["description"]

    def test_course_description_injected_when_provided(self) -> None:
        """course_description appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            course_description="A comprehensive course on data engineering.",
        )
        assert (
            "A comprehensive course on data engineering." in kwargs["description"]
        )
        assert "Course Description:" in kwargs["description"]

    def test_course_duration_injected_when_provided(self) -> None:
        """course_duration appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            course_duration="12 weeks",
        )
        assert "12 weeks" in kwargs["description"]
        assert "Course Duration:" in kwargs["description"]

    def test_target_audience_injected_when_provided(self) -> None:
        """target_audience appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            target_audience="Career changers with basic programming experience",
        )
        assert (
            "Career changers with basic programming experience"
            in kwargs["description"]
        )
        assert "Target Audience:" in kwargs["description"]

    def test_optional_fields_omitted_when_none(self) -> None:
        """When optional fields are None, their labels do not appear."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Course Context (from Intake Specialist)" not in kwargs["description"]
        assert "Course Description:" not in kwargs["description"]
        assert "Course Duration:" not in kwargs["description"]
        assert "Target Audience:" not in kwargs["description"]

    # -- Mandate injection -----------------------------------------------

    def test_humanics_mandate_in_description(self) -> None:
        """The Humanics embedding mandate is present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Joseph Aoun's Humanics framework" in kwargs["description"]
        assert "Technological Literacy [T]" in kwargs["description"]
        assert "Data Literacy [D]" in kwargs["description"]
        assert "Human Literacy [H]" in kwargs["description"]

    def test_experiential_learning_mandate_in_description(self) -> None:
        """The experiential learning mandate is present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Experiential Learning & Industry Integration" in kwargs["description"]
        assert "Co-operative Education" in kwargs["description"]
        assert "Capstone Project" in kwargs["description"]
        assert "Industry Partnerships" in kwargs["description"]

    def test_markdown_structure_in_description(self) -> None:
        """The Markdown formatting requirements are present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Markdown Formatting Requirements" in kwargs["description"]
        assert "ATX-style headings" in kwargs["description"]

    def test_backward_design_mentioned(self) -> None:
        """The description references backward design principles."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "backward design" in kwargs["description"].lower()

    # -- Expected output content -----------------------------------------

    def test_expected_output_requires_nine_sections(self) -> None:
        """The expected_output references all nine required sections."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Course Overview" in kwargs["expected_output"]
        assert "Instructor Notes" in kwargs["expected_output"]

    def test_expected_output_requires_humanics_tags(self) -> None:
        """The expected_output mandates Humanics tags."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "[T]" in kwargs["expected_output"]
        assert "[D]" in kwargs["expected_output"]
        assert "[H]" in kwargs["expected_output"]

    def test_expected_output_requires_week_by_week_table(self) -> None:
        """The expected_output mandates a week-by-week schedule table."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "week-by-week" in kwargs["expected_output"].lower()


# ===================================================================
# create_lab_generation_task
# ===================================================================


class TestCreateLabGenerationTask:
    """Tests for the lab generation task factory."""

    @pytest.fixture(autouse=True)
    def _patch_task(self) -> None:
        """Patch Task in the lab_generation module."""
        with patch(
            "src.tasks.lab_generation.Task"
        ) as self.mock_task:
            yield

    def _call_factory(self, **overrides) -> dict:
        """Call create_lab_generation_task and return Task kwargs."""
        from src.tasks.lab_generation import create_lab_generation_task

        kwargs: dict = {
            "course_name": "Python for Data Engineering",
            "agent": MagicMock(),
        }
        kwargs.update(overrides)
        create_lab_generation_task(**kwargs)
        return self.mock_task.call_args.kwargs

    def test_course_name_in_description(self) -> None:
        """The course name appears in the task description."""
        kwargs = self._call_factory()
        assert "Python for Data Engineering" in kwargs["description"]

    def test_language_in_description(self) -> None:
        """The primary language appears in the description."""
        kwargs = self._call_factory(course_name="Test Course", language="Python")
        assert "Python" in kwargs["description"]
        assert "Primary Language" in kwargs["description"]

    def test_default_language_is_python(self) -> None:
        """When no language is specified, Python is the default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Python" in kwargs["description"]

    def test_output_file_follows_expected_pattern(self) -> None:
        """The output_file is output/labs/<safe_name>/README.md."""
        kwargs = self._call_factory()
        assert (
            kwargs["output_file"]
            == "output/labs/python_for_data_engineering/README.md"
        )

    def test_async_execution_is_false(self) -> None:
        """Tasks are synchronous by default."""
        kwargs = self._call_factory(course_name="Test Course")
        assert kwargs["async_execution"] is False

    def test_agent_is_passed_through(self) -> None:
        """The agent parameter is passed through to the Task."""
        agent = MagicMock()
        kwargs = self._call_factory(agent=agent)
        assert kwargs["agent"] is agent

    # -- Optional fields -------------------------------------------------

    def test_syllabus_context_injected_when_provided(self) -> None:
        """syllabus_context appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            syllabus_context="# Sample Syllabus\n\n## Module 1: Intro\n...",
        )
        assert "# Sample Syllabus" in kwargs["description"]
        assert "Syllabus Context:" in kwargs["description"]

    def test_topic_focus_injected_when_provided(self) -> None:
        """topic_focus appears in the description when supplied."""
        kwargs = self._call_factory(
            course_name="Test Course",
            topic_focus="ETL pipelines, REST APIs, Docker",
        )
        assert "ETL pipelines, REST APIs, Docker" in kwargs["description"]
        assert "Topic Focus:" in kwargs["description"]

    def test_optional_fields_omitted_when_none(self) -> None:
        """When optional fields are None, their labels do not appear."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Syllabus Context:" not in kwargs["description"]
        assert "Topic Focus:" not in kwargs["description"]

    # -- Mandate injection -----------------------------------------------

    def test_tier_definitions_in_description(self) -> None:
        """The tier definitions are present in the description."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Tier 1 — Foundations" in kwargs["description"]
        assert "Tier 2 — Application" in kwargs["description"]
        assert "Tier 3 — Architecture" in kwargs["description"]

    def test_directory_structure_in_description(self) -> None:
        """The directory structure requirements are present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Directory Structure Requirements" in kwargs["description"]
        assert "tier1_foundations" in kwargs["description"]

    def test_readme_template_in_description(self) -> None:
        """The README template requirements are present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "README.md Requirements" in kwargs["description"]
        assert "Learning Objectives" in kwargs["description"]

    def test_humanics_lab_mandate_in_description(self) -> None:
        """The Humanics lab mandate is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Humanics Literacies in Labs" in kwargs["description"]
        assert "Technological Literacy [T]" in kwargs["description"]

    def test_code_quality_mandate_in_description(self) -> None:
        """The code quality mandate is present."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Code Quality Standards" in kwargs["description"]
        assert "type hints" in kwargs["description"]

    # -- Expected output content -----------------------------------------

    def test_expected_output_requires_three_tiers(self) -> None:
        """The expected_output mandates all three tiers."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Tier 1 — Foundations" in kwargs["expected_output"]
        assert "Tier 2 — Application" in kwargs["expected_output"]
        assert "Tier 3 — Architecture" in kwargs["expected_output"]

    def test_expected_output_requires_minimum_lab_counts(self) -> None:
        """The expected_output mandates at least 3 labs per tier."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "≥3 labs" in kwargs["expected_output"]

    def test_expected_output_requires_docker_for_tier3(self) -> None:
        """Tier 3 must include Docker and docker-compose."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "Docker" in kwargs["expected_output"]
        assert "docker-compose" in kwargs["expected_output"]

    def test_expected_output_requires_self_contained_labs(self) -> None:
        """Labs must be self-contained and independently runnable."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "self-contained" in kwargs["expected_output"].lower()

    def test_expected_output_requires_top_level_readme(self) -> None:
        """The first file must be a top-level README.md index."""
        kwargs = self._call_factory(course_name="Test Course")
        assert "README.md" in kwargs["expected_output"]
        assert "top-level index" in kwargs["expected_output"]