"""
test_main.py — Tests for the main CLI module and CourseSpecification model
==========================================================================

Validates that ``CourseSpecification`` (the Pydantic structured-output
model used by the Intake Specialist) enforces its schema correctly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.main import CourseSpecification


class TestCourseSpecification:
    """Tests for the CourseSpecification Pydantic model."""

    def test_valid_model_creation(self) -> None:
        """A CourseSpecification can be created with both fields."""
        spec = CourseSpecification(
            course_context="A rich context string about Python.",
            primary_language="Python",
        )
        assert spec.course_context == "A rich context string about Python."
        assert spec.primary_language == "Python"

    def test_javascript_language(self) -> None:
        """primary_language accepts JavaScript."""
        spec = CourseSpecification(
            course_context="Context for JS course.",
            primary_language="JavaScript",
        )
        assert spec.primary_language == "JavaScript"

    def test_typescript_language(self) -> None:
        """primary_language accepts TypeScript."""
        spec = CourseSpecification(
            course_context="Context for TS course.",
            primary_language="TypeScript",
        )
        assert spec.primary_language == "TypeScript"

    def test_course_context_is_required(self) -> None:
        """course_context is a required field."""
        with pytest.raises(ValidationError):
            CourseSpecification(primary_language="Python")

    def test_primary_language_is_required(self) -> None:
        """primary_language is a required field."""
        with pytest.raises(ValidationError):
            CourseSpecification(course_context="Some context")

    def test_both_fields_are_required(self) -> None:
        """Both fields must be provided."""
        with pytest.raises(ValidationError):
            CourseSpecification()

    def test_model_can_be_serialized(self) -> None:
        """The model can be serialized to a dict."""
        spec = CourseSpecification(
            course_context="Test context",
            primary_language="Go",
        )
        data = spec.model_dump()
        assert data == {
            "course_context": "Test context",
            "primary_language": "Go",
            "grading_scale": None,
            "student_pathway": None,
            "year_level": None,
            "hardware_constraints": None,
        }

    def test_model_can_be_deserialized(self) -> None:
        """The model can be created from a dict."""
        data = {
            "course_context": "Deserialized context",
            "primary_language": "Rust",
        }
        spec = CourseSpecification.model_validate(data)
        assert spec.course_context == "Deserialized context"
        assert spec.primary_language == "Rust"

    def test_empty_strings_are_allowed(self) -> None:
        """Empty strings pass Pydantic validation (no min_length constraint)."""
        spec = CourseSpecification(course_context="", primary_language="")
        assert spec.course_context == ""
        assert spec.primary_language == ""

    def test_multiline_course_context(self) -> None:
        """course_context can contain multi-line text."""
        context = (
            "Course Name: Advanced JavaScript\n"
            "Tech Stack: Node.js, Express, React\n"
            "Student Profile: BOL, Year 2\n"
        )
        spec = CourseSpecification(
            course_context=context,
            primary_language="JavaScript",
        )
        assert "Node.js" in spec.course_context
        assert "BOL" in spec.course_context