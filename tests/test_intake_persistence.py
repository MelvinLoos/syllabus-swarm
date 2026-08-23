"""Tests for IntakeSession persistence (Issue #9).

Covers:
  * IntakeSession model validation (required fields, invalid JSON)
  * Round-trip: create -> serialize -> deserialize -> verify
  * --load-session CLI flag: skips interactive intake, preserves
    CourseSpecification exactly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.main import (
    CourseSpecification,
    IntakeSession,
    _generate_run_id,
    _get_load_session_path,
    _sanitize_filename,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_spec() -> CourseSpecification:
    """A realistic CourseSpecification for testing."""
    return CourseSpecification(
        course_context=(
            "Course Name: Full-Stack Web Development\\n"
            "Tech Stack: React, Node.js, PostgreSQL\\n"
            "Kerntaken: P2-K1 (designing), P3-K1 (building)\\n"
            "Student Profile: BOL, Year 2, BPV-ready"
        ),
        primary_language="JavaScript",
    )


@pytest.fixture
def sample_session(sample_spec: CourseSpecification) -> IntakeSession:
    """A fully-populated IntakeSession for round-trip tests."""
    return IntakeSession(
        course_name="Full-Stack Web Development",
        questions="1. Which frameworks? React, Node, or something else?",
        answers="React for frontend, Node.js for backend, PostgreSQL for database.",
        course_specification=sample_spec,
        timestamp="2025-08-23T12:00:00+00:00",
        run_id="2025-08-23_120000_Full_Stack_Web_Development",
    )


# ---------------------------------------------------------------------------
# IntakeSession model validation
# ---------------------------------------------------------------------------


class TestIntakeSessionValidation:
    """IntakeSession enforces all required fields."""

    def test_valid_model_creation(self, sample_session: IntakeSession) -> None:
        assert sample_session.course_name == "Full-Stack Web Development"
        assert "React" in sample_session.questions
        assert "Node" in sample_session.answers
        assert sample_session.course_specification.primary_language == "JavaScript"

    def test_course_name_is_required(self, sample_spec: CourseSpecification) -> None:
        with pytest.raises(ValidationError):
            IntakeSession(
                questions="Q1",
                answers="A1",
                course_specification=sample_spec,
                timestamp="2025-01-01T00:00:00",
                run_id="test",
            )

    def test_questions_is_required(self, sample_spec: CourseSpecification) -> None:
        with pytest.raises(ValidationError):
            IntakeSession(
                course_name="Test",
                answers="A1",
                course_specification=sample_spec,
                timestamp="2025-01-01T00:00:00",
                run_id="test",
            )

    def test_answers_is_required(self, sample_spec: CourseSpecification) -> None:
        with pytest.raises(ValidationError):
            IntakeSession(
                course_name="Test",
                questions="Q1",
                course_specification=sample_spec,
                timestamp="2025-01-01T00:00:00",
                run_id="test",
            )

    def test_course_specification_is_required(self) -> None:
        with pytest.raises(ValidationError):
            IntakeSession(
                course_name="Test",
                questions="Q1",
                answers="A1",
                timestamp="2025-01-01T00:00:00",
                run_id="test",
            )

    def test_timestamp_is_required(self, sample_spec: CourseSpecification) -> None:
        with pytest.raises(ValidationError):
            IntakeSession(
                course_name="Test",
                questions="Q1",
                answers="A1",
                course_specification=sample_spec,
                run_id="test",
            )

    def test_run_id_is_required(self, sample_spec: CourseSpecification) -> None:
        with pytest.raises(ValidationError):
            IntakeSession(
                course_name="Test",
                questions="Q1",
                answers="A1",
                course_specification=sample_spec,
                timestamp="2025-01-01T00:00:00",
            )

    def test_all_fields_are_required(self) -> None:
        with pytest.raises(ValidationError):
            IntakeSession()


# ---------------------------------------------------------------------------
# Serialization / Deserialization (round-trip)
# ---------------------------------------------------------------------------


class TestIntakeSessionRoundTrip:
    """IntakeSession survives a full serialize / deserialize cycle."""

    def test_serialize_to_json(self, sample_session: IntakeSession) -> None:
        raw = sample_session.model_dump_json(indent=2)
        data = json.loads(raw)
        assert data["course_name"] == "Full-Stack Web Development"
        assert "React" in data["questions"]
        assert "Node" in data["answers"]
        assert data["course_specification"]["primary_language"] == "JavaScript"
        assert data["timestamp"] == "2025-08-23T12:00:00+00:00"
        assert "Full_Stack_Web_Development" in data["run_id"]

    def test_deserialize_from_json(self, sample_session: IntakeSession) -> None:
        raw = sample_session.model_dump_json()
        restored = IntakeSession.model_validate_json(raw)
        assert restored.course_name == sample_session.course_name
        assert restored.questions == sample_session.questions
        assert restored.answers == sample_session.answers
        assert restored.timestamp == sample_session.timestamp
        assert restored.run_id == sample_session.run_id

    def test_course_specification_matches_after_round_trip(
        self, sample_session: IntakeSession
    ) -> None:
        raw = sample_session.model_dump_json()
        restored = IntakeSession.model_validate_json(raw)
        orig = sample_session.course_specification
        restored_spec = restored.course_specification
        assert restored_spec.course_context == orig.course_context
        assert restored_spec.primary_language == orig.primary_language
        assert isinstance(restored_spec, CourseSpecification)

    def test_file_round_trip(
        self, sample_session: IntakeSession, tmp_path: Path
    ) -> None:
        # Save
        session_file = tmp_path / "intake_session.json"
        session_file.write_text(
            sample_session.model_dump_json(indent=2), encoding="utf-8"
        )
        assert session_file.exists()
        # Load
        raw = session_file.read_text(encoding="utf-8")
        restored = IntakeSession.model_validate_json(raw)
        # Verify CourseSpecification matches
        assert (
            restored.course_specification.course_context
            == sample_session.course_specification.course_context
        )
        assert (
            restored.course_specification.primary_language
            == sample_session.course_specification.primary_language
        )
        assert restored.course_name == sample_session.course_name
        assert restored.questions == sample_session.questions
        assert restored.answers == sample_session.answers

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad_session.json"
        bad_file.write_text("this is not json", encoding="utf-8")
        with pytest.raises(Exception):
            IntakeSession.model_validate_json(
                bad_file.read_text(encoding="utf-8")
            )

    def test_valid_json_wrong_schema_raises(self, tmp_path: Path) -> None:
        wrong_file = tmp_path / "wrong_schema.json"
        wrong_file.write_text(
            json.dumps({"some": "random", "fields": 42}), encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            IntakeSession.model_validate_json(
                wrong_file.read_text(encoding="utf-8")
            )


# ---------------------------------------------------------------------------
# --load-session CLI flag tests
# ---------------------------------------------------------------------------


class TestLoadSessionCLI:
    """The --load-session flag extracts the path and bypasses intake."""

    def test_get_load_session_path_extracts_value(self, monkeypatch) -> None:
        test_path = "/tmp/output/intake_session.json"
        monkeypatch.setattr(
            sys, "argv", ["src/main.py", "--load-session", test_path]
        )
        result = _get_load_session_path()
        assert result == test_path

    def test_get_load_session_path_none_when_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["src/main.py", "Some Course"])
        assert _get_load_session_path() is None

    def test_get_load_session_path_exits_when_no_value(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["src/main.py", "--load-session"])
        with pytest.raises(SystemExit) as exc_info:
            _get_load_session_path()
        assert exc_info.value.code == 1

    def test_load_session_bypasses_interactive_intake(
        self, sample_session: IntakeSession, tmp_path: Path, monkeypatch
    ) -> None:
        session_file = tmp_path / "intake_session.json"
        session_file.write_text(
            sample_session.model_dump_json(indent=2), encoding="utf-8"
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["src/main.py", "--load-session", str(session_file), "--skip-labs"],
        )
        mock_result = MagicMock()
        mock_result.syllabus_ok = True
        mock_result.labs_ok = True
        mock_result.all_succeeded = True
        mock_result.syllabus_path = tmp_path / "syllabus.md"
        mock_result.labs_base_path = tmp_path / "labs"
        mock_result.manifest_path = None

        with patch("src.main.run_syllabus_crew", return_value=mock_result) as mock_crew:
            with patch("builtins.input") as mock_input:
                from src.main import main
                main()
                mock_input.assert_not_called()
                mock_crew.assert_called_once()
                args, kwargs = mock_crew.call_args
                # course_context is the first positional argument
                assert (
                    args[0]
                    == sample_session.course_specification.course_context
                )
                assert kwargs["course_name"] == sample_session.course_name
                assert (
                    kwargs["primary_language"]
                    == sample_session.course_specification.primary_language
                )

    def test_load_session_nonexistent_file_exits(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["src/main.py", "--load-session", "/tmp/nonexistent/intake_session.json"],
        )
        with pytest.raises(SystemExit) as exc_info:
            from src.main import main
            main()
        assert exc_info.value.code == 1

    def test_load_session_invalid_json_exits(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["src/main.py", "--load-session", str(bad_file)]
        )
        with pytest.raises(SystemExit) as exc_info:
            from src.main import main
            main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """The run-id / filename helpers produce deterministic output."""

    def test_sanitize_filename_replaces_spaces(self) -> None:
        assert _sanitize_filename("Data Science") == "Data_Science"

    def test_sanitize_filename_strips_special_chars(self) -> None:
        result = _sanitize_filename('A/B:C*D?E"F<G>H|I')
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert '"' not in result

    def test_generate_run_id_contains_timestamp(self) -> None:
        rid = _generate_run_id("Test_Course")
        # Format: YYYY-MM-DD_HHMMSS_course_name
        date_part = rid.split("_")[0]
        assert len(date_part) == 10  # "YYYY-MM-DD"
        assert date_part[4] == "-" and date_part[7] == "-"

    def test_generate_run_id_includes_safe_name(self) -> None:
        rid = _generate_run_id("My_Course")
        assert rid.endswith("_My_Course")
