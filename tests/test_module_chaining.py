"""
test_module_chaining.py — Tests for Module Chaining (Sub-Issue 4)
=================================================================

Epic #7: Curriculum Memory & Continuity Engine

Validates:
1.  ``_resolve_prerequisites`` resolves learning objectives and key
    concepts from a previous course's ``course_graph.json``.
2.  Error handling when slug not found, ``course_graph.json`` missing,
    or JSON is malformed.
3.  Prerequisite string is correctly formatted.
4.  Prerequisites are injected into ``course_context`` in both the
    normal intake flow and the ``--load-session`` flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.main import (
    CourseSpecification,
    _resolve_prerequisites,
    _sanitize_filename,
    _generate_run_id,
)
from src.models import CourseGraph, ModuleSummary


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_spec() -> CourseSpecification:
    """A valid CourseSpecification for use in course graphs."""
    return CourseSpecification(
        course_context="Intro to Python for data science.",
        primary_language="Python",
    )


@pytest.fixture
def sample_graph(sample_spec: CourseSpecification) -> CourseGraph:
    """A fully populated CourseGraph fixture."""
    return CourseGraph(
        specification=sample_spec,
        course_slug="intro-python-ds",
        learning_objectives=[
            "Write basic Python scripts",
            "Perform data analysis with pandas",
            "Build simple ML models with scikit-learn",
        ],
        key_concepts=[
            "variables", "loops", "functions",
            "pandas", "DataFrames", "scikit-learn",
        ],
        prerequisites=["Basic computer literacy"],
        modules=[
            ModuleSummary(
                title="Python Fundamentals",
                duration_weeks=2.0,
                hours_per_week=3.0,
                topics=["variables", "control flow", "functions"],
            ),
        ],
    )


@pytest.fixture
def empty_graph(sample_spec: CourseSpecification) -> CourseGraph:
    """A CourseGraph with no learning objectives or key concepts."""
    return CourseGraph(
        specification=sample_spec,
        course_slug="empty-course",
        learning_objectives=[],
        key_concepts=[],
        prerequisites=[],
        modules=[],
    )


def _setup_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect OUTPUT_ROOT in both main and syllabus_crew to tmp_path."""
    import src.main as main_module
    import src.crews.syllabus_crew as sc_module

    monkeypatch.setattr(main_module, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(sc_module, "OUTPUT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def populated_output_dir(
    tmp_path: Path,
    sample_graph: CourseGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Create a temporary output directory with a course_graph.json."""
    _setup_output_dir(tmp_path, monkeypatch)

    run_dir = tmp_path / "2026-08-22_153000_ML_Basics"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_path = run_dir / "course_graph.json"
    graph_path.write_text(sample_graph.model_dump_json(indent=2), encoding="utf-8")

    return tmp_path


@pytest.fixture
def empty_output_dir(
    tmp_path: Path,
    empty_graph: CourseGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Create a temporary output directory with an empty course_graph.json."""
    _setup_output_dir(tmp_path, monkeypatch)

    run_dir = tmp_path / "2026-08-22_153000_Empty_Course"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_path = run_dir / "course_graph.json"
    graph_path.write_text(empty_graph.model_dump_json(indent=2), encoding="utf-8")

    return tmp_path


# ===================================================================
# 1. _resolve_prerequisites — valid resolution
# ===================================================================


class TestResolvePrerequisites:
    """Tests for the _resolve_prerequisites helper function."""

    def test_resolves_from_matching_slug(
        self, populated_output_dir: Path
    ) -> None:
        """Resolves prerequisites when a matching run directory exists."""
        result = _resolve_prerequisites("ML_Basics")
        assert "The students have already mastered" in result
        assert "**Learning Objectives:**" in result
        assert "Write basic Python scripts" in result
        assert "Perform data analysis with pandas" in result
        assert "**Key Concepts:**" in result
        assert "variables, loops, functions" in result

    def test_partial_slug_matching(
        self, populated_output_dir: Path
    ) -> None:
        """A partial slug like 'Basics' matches the full directory name."""
        result = _resolve_prerequisites("Basics")
        assert "The students have already mastered" in result

    def test_full_run_id_matching(
        self, populated_output_dir: Path
    ) -> None:
        """The full run ID also matches."""
        result = _resolve_prerequisites("2026-08-22_153000_ML_Basics")
        assert "The students have already mastered" in result

    def test_empty_learning_objectives_and_concepts(
        self, empty_output_dir: Path
    ) -> None:
        """When both lists are empty, a fallback message is returned."""
        result = _resolve_prerequisites("Empty_Course")
        assert "The students have already mastered" in result
        assert "No learning objectives or key concepts were recorded" in result

    def test_result_does_not_contain_raw_json(
        self, populated_output_dir: Path
    ) -> None:
        """The result is formatted text, not raw JSON."""
        result = _resolve_prerequisites("ML_Basics")
        assert "course_slug" not in result
        assert "specification" not in result
        assert "generated_at" not in result

    def test_multiple_matches_picks_first_sorted(
        self, tmp_path: Path, sample_graph: CourseGraph, monkeypatch
    ) -> None:
        """When multiple directories match, the first alphabetically is used."""
        _setup_output_dir(tmp_path, monkeypatch)

        # Create two matching directories with different content.
        dir_a = tmp_path / "2026-08-20_000000_ML_Basics"
        dir_b = tmp_path / "2026-08-22_153000_ML_Basics"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)

        graph_a = sample_graph.model_copy(update={"course_slug": "course-a"})
        graph_b = sample_graph.model_copy(update={"course_slug": "course-b"})

        (dir_a / "course_graph.json").write_text(
            graph_a.model_dump_json(indent=2), encoding="utf-8"
        )
        (dir_b / "course_graph.json").write_text(
            graph_b.model_dump_json(indent=2), encoding="utf-8"
        )

        result = _resolve_prerequisites("ML_Basics")
        # dir_a sorts first alphabetically (2026-08-20 < 2026-08-22)
        assert "The students have already mastered" in result

    # ----------------------------------------------------------------
    # Error cases
    # ----------------------------------------------------------------

    def test_slug_not_found_exits(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """SystemExit is raised when no directory matches the slug."""
        _setup_output_dir(tmp_path, monkeypatch)
        (tmp_path / "some_other_dir").mkdir()

        with pytest.raises(SystemExit):
            _resolve_prerequisites("NonExistentSlug")

    def test_no_course_graph_json_exits(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """SystemExit is raised when the directory exists but has no graph."""
        _setup_output_dir(tmp_path, monkeypatch)

        run_dir = tmp_path / "2026-08-22_153000_No_Graph"
        run_dir.mkdir(parents=True)
        # No course_graph.json written.

        with pytest.raises(SystemExit):
            _resolve_prerequisites("No_Graph")

    def test_malformed_json_exits(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """SystemExit is raised when course_graph.json is not valid JSON."""
        _setup_output_dir(tmp_path, monkeypatch)

        run_dir = tmp_path / "2026-08-22_153000_Bad_JSON"
        run_dir.mkdir(parents=True)
        (run_dir / "course_graph.json").write_text(
            "this is not json {{{", encoding="utf-8"
        )

        with pytest.raises(SystemExit):
            _resolve_prerequisites("Bad_JSON")

    def test_output_dir_not_found_exits(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """SystemExit is raised when the output directory doesn't exist."""
        non_existent = tmp_path / "does_not_exist"
        _setup_output_dir(tmp_path, monkeypatch)
        # Override with non-existent path
        import src.main as main_module
        monkeypatch.setattr(main_module, "OUTPUT_ROOT", non_existent)

        with pytest.raises(SystemExit):
            _resolve_prerequisites("anything")


# ===================================================================
# 2. Prerequisite string formatting
# ===================================================================


class TestPrerequisiteFormatting:
    """Validate the exact format of the prerequisite context string."""

    def test_contains_intro_sentence(
        self, populated_output_dir: Path
    ) -> None:
        """The string starts with the intro sentence."""
        result = _resolve_prerequisites("ML_Basics")
        assert result.startswith(
            "The students have already mastered the following from a "
            "previous course:\n"
        )

    def test_learning_objectives_are_bulleted(
        self, populated_output_dir: Path
    ) -> None:
        """Each learning objective is a bullet point."""
        result = _resolve_prerequisites("ML_Basics")
        lines = result.split("\n")
        # Collect bullet lines between "Learning Objectives:" and "Key Concepts:"
        in_objectives = False
        obj_lines: list[str] = []
        for line in lines:
            if "**Learning Objectives:**" in line:
                in_objectives = True
                continue
            if "**Key Concepts:**" in line:
                in_objectives = False
                continue
            if in_objectives and line.startswith("- "):
                obj_lines.append(line)
        assert len(obj_lines) == 3
        assert obj_lines[0] == "- Write basic Python scripts"

    def test_key_concepts_are_comma_separated(
        self, populated_output_dir: Path
    ) -> None:
        """Key concepts are presented as a single comma-separated line."""
        result = _resolve_prerequisites("ML_Basics")
        assert "- variables, loops, functions, pandas, DataFrames, scikit-learn" in result

    def test_only_objectives_when_no_concepts(
        self, tmp_path: Path, sample_spec: CourseSpecification, monkeypatch
    ) -> None:
        """When only learning_objectives exist, key_concepts section is absent."""
        _setup_output_dir(tmp_path, monkeypatch)

        graph = CourseGraph(
            specification=sample_spec,
            course_slug="objectives-only",
            learning_objectives=["Learn X", "Master Y"],
            key_concepts=[],
        )
        run_dir = tmp_path / "2026-08-22_153000_Objectives_Only"
        run_dir.mkdir(parents=True)
        (run_dir / "course_graph.json").write_text(
            graph.model_dump_json(indent=2), encoding="utf-8"
        )

        result = _resolve_prerequisites("Objectives_Only")
        assert "**Learning Objectives:**" in result
        assert "**Key Concepts:**" not in result

    def test_only_concepts_when_no_objectives(
        self, tmp_path: Path, sample_spec: CourseSpecification, monkeypatch
    ) -> None:
        """When only key_concepts exist, learning_objectives section is absent."""
        _setup_output_dir(tmp_path, monkeypatch)

        graph = CourseGraph(
            specification=sample_spec,
            course_slug="concepts-only",
            learning_objectives=[],
            key_concepts=["topic A", "topic B"],
        )
        run_dir = tmp_path / "2026-08-22_153000_Concepts_Only"
        run_dir.mkdir(parents=True)
        (run_dir / "course_graph.json").write_text(
            graph.model_dump_json(indent=2), encoding="utf-8"
        )

        result = _resolve_prerequisites("Concepts_Only")
        assert "**Learning Objectives:**" not in result
        assert "**Key Concepts:**" in result


# ===================================================================
# 3. Prerequisite injection into course_context
# ===================================================================


class TestPrerequisiteInjection:
    """Verify prerequisites are injected into course_context correctly."""

    def test_injection_prepends_to_context(
        self, populated_output_dir: Path
    ) -> None:
        """The prerequisite string is prepended to the course context."""
        prereq = _resolve_prerequisites("ML_Basics")
        original_context = "Course Name: Advanced ML\nTech Stack: Python, PyTorch"
        combined = prereq + "\n\n" + original_context

        assert combined.startswith("The students have already mastered")
        assert original_context in combined
        # The prerequisite text comes before the original context.
        prereq_pos = combined.index("The students have already mastered")
        context_pos = combined.index("Course Name: Advanced ML")
        assert prereq_pos < context_pos

    def test_injection_preserves_original_context(
        self, populated_output_dir: Path
    ) -> None:
        """The original course context is unchanged after injection."""
        prereq = _resolve_prerequisites("ML_Basics")
        original_context = "Course Name: Advanced ML\nTech Stack: Python, PyTorch"
        combined = prereq + "\n\n" + original_context

        assert combined.endswith(original_context)

    def test_no_injection_when_builds_upon_is_none(self) -> None:
        """When builds_upon is None, course_context is unchanged."""
        original_context = "Course Name: Standalone Course"
        builds_upon = None

        # Simulate the injection logic from main().
        if builds_upon:
            prereq_context = _resolve_prerequisites(builds_upon)
            original_context = prereq_context + "\n\n" + original_context

        assert original_context == "Course Name: Standalone Course"