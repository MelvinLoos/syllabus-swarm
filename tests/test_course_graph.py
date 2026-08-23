"""
test_course_graph.py — Tests for CourseGraph & ModuleSummary (Issue #10)
========================================================================

Parent Epic: #7 Curriculum Memory & Continuity Engine, Phase 3 (Export Side)

Validates:
1.  JSON schema matches the Pydantic model definition.
2.  Round-trip: construct → serialize → deserialize → verify.
3.  CourseGraph composes CourseSpecification (no field duplication).
4.  export-course-graph command via OutputExportTool.
5.  course_graph.json written alongside README.md by manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.exporters.file_writer import (
    OutputPathConfig,
)
from src.exporters.manifest import update_output_manifest
from src.exporters.tool import OutputExportTool
from src.main import CourseSpecification
from src.models import CourseGraph, ModuleSummary

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_spec() -> CourseSpecification:
    """A valid CourseSpecification for use across tests."""
    return CourseSpecification(
        course_context="Intro to Python for data science and ML basics.",
        primary_language="Python",
    )


@pytest.fixture
def sample_module() -> ModuleSummary:
    """A single typical module."""
    return ModuleSummary(
        title="Python Fundamentals",
        duration_weeks=2.0,
        hours_per_week=3.0,
        topics=["variables", "control flow", "functions"],
    )


@pytest.fixture
def sample_graph(sample_spec: CourseSpecification) -> CourseGraph:
    """A fully populated CourseGraph."""
    return CourseGraph(
        specification=sample_spec,
        course_slug="intro-python-ds",
        learning_objectives=[
            "Write basic Python scripts",
            "Perform data analysis with pandas",
            "Build simple ML models",
        ],
        key_concepts=["variables", "loops", "pandas", "scikit-learn"],
        prerequisites=["Basic computer literacy"],
        modules=[
            ModuleSummary(
                title="Python Fundamentals",
                duration_weeks=2.0,
                hours_per_week=3.0,
                topics=["variables", "control flow", "functions"],
            ),
            ModuleSummary(
                title="Data Analysis with Pandas",
                duration_weeks=3.0,
                hours_per_week=3.0,
                topics=["DataFrames", "cleaning", "visualization"],
            ),
        ],
    )
# ===================================================================
# 1. ModuleSummary schema tests
# ===================================================================


class TestModuleSummary:
    """Validate the ModuleSummary lightweight Pydantic model."""

    def test_valid_creation(self) -> None:
        """ModuleSummary can be created with all fields."""
        m = ModuleSummary(
            title="Intro to Python",
            duration_weeks=1.5,
            hours_per_week=3.0,
            topics=["hello world", "syntax"],
        )
        assert m.title == "Intro to Python"
        assert m.duration_weeks == 1.5
        assert m.topics == ["hello world", "syntax"]

    def test_default_topics(self) -> None:
        """topics defaults to an empty list."""
        m = ModuleSummary(title="Solo Module", duration_weeks=1.0, hours_per_week=3.0)
        assert m.topics == []

    def test_title_is_required(self) -> None:
        """title must be provided."""
        with pytest.raises(ValidationError):
            ModuleSummary(duration_weeks=1.0, hours_per_week=3.0, topics=["a"])

    def test_duration_weeks_is_required(self) -> None:
        """duration_weeks must be provided."""
        with pytest.raises(ValidationError):
            ModuleSummary(title="Test", hours_per_week=3.0, topics=["a"])

    def test_empty_title_is_invalid(self) -> None:
        """title must have at least 1 character."""
        with pytest.raises(ValidationError):
            ModuleSummary(title="", duration_weeks=1.0, hours_per_week=3.0)

    def test_negative_duration_is_invalid(self) -> None:
        """duration_weeks must be >= 0."""
        with pytest.raises(ValidationError):
            ModuleSummary(title="Test", duration_weeks=-0.5, hours_per_week=3.0)

    def test_zero_duration_is_valid(self) -> None:
        """duration_weeks can be zero."""
        m = ModuleSummary(title="Placeholder", duration_weeks=0.0, hours_per_week=3.0)
        assert m.duration_weeks == 0.0

    def test_serialize_roundtrip(self) -> None:
        """Serialize to JSON and deserialize back."""
        m = ModuleSummary(
            title="Data Viz",
            duration_weeks=2.0,
            hours_per_week=3.0,
            topics=["matplotlib", "seaborn"],
        )
        data = m.model_dump()
        restored = ModuleSummary.model_validate(data)
        assert restored.title == m.title
        assert restored.duration_weeks == m.duration_weeks
        assert restored.topics == m.topics

    def test_json_schema(self) -> None:
        """Generated JSON schema matches expected top-level keys."""
        schema = ModuleSummary.model_json_schema()
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"title", "duration_weeks", "hours_per_week"}
        props = schema["properties"]
        assert props["title"]["type"] == "string"
        assert "minLength" in props["title"]
        assert props["hours_per_week"]["type"] == "number"
        assert props["hours_per_week"]["minimum"] == 0.0
        assert props["duration_weeks"]["type"] == "number"
        assert props["duration_weeks"]["minimum"] == 0.0
        assert props["topics"]["type"] == "array"

# ===================================================================
# 2. CourseGraph schema tests
# ===================================================================


class TestCourseGraph:
    """Validate the CourseGraph Pydantic model."""

    def test_valid_creation_minimal(self, sample_spec: CourseSpecification) -> None:
        """CourseGraph can be created with only required fields."""
        graph = CourseGraph(
            specification=sample_spec,
            course_slug="test-course",
        )
        assert graph.specification == sample_spec
        assert graph.course_slug == "test-course"
        assert graph.learning_objectives == []
        assert graph.key_concepts == []
        assert graph.prerequisites == []
        assert graph.modules == []
        assert graph.generated_at is not None

    def test_valid_creation_full(self, sample_graph: CourseGraph) -> None:
        """All fields populate correctly in a full graph."""
        assert sample_graph.course_slug == "intro-python-ds"
        assert len(sample_graph.learning_objectives) == 3
        assert len(sample_graph.key_concepts) == 4
        assert len(sample_graph.prerequisites) == 1
        assert len(sample_graph.modules) == 2

    def test_specification_is_required(self) -> None:
        """specification is a required field."""
        with pytest.raises(ValidationError):
            CourseGraph(course_slug="test")

    def test_course_slug_is_required(self, sample_spec: CourseSpecification) -> None:
        """course_slug is required."""
        with pytest.raises(ValidationError):
            CourseGraph(specification=sample_spec)

    def test_empty_course_slug_is_invalid(
        self, sample_spec: CourseSpecification
    ) -> None:
        """course_slug must be non-empty."""
        with pytest.raises(ValidationError):
            CourseGraph(specification=sample_spec, course_slug="")

    def test_generated_at_is_iso8601(self, sample_spec: CourseSpecification) -> None:
        """generated_at is automatically populated as ISO 8601."""
        graph = CourseGraph(specification=sample_spec, course_slug="test")
        assert "T" in graph.generated_at
        assert graph.generated_at.endswith("Z")
        import datetime
        dt = datetime.datetime.strptime(
            graph.generated_at, "%Y-%m-%dT%H:%M:%SZ"
        )
        assert isinstance(dt, datetime.datetime)

    def test_generated_at_can_be_overridden(
        self, sample_spec: CourseSpecification
    ) -> None:
        """generated_at can be set explicitly."""
        graph = CourseGraph(
            specification=sample_spec,
            course_slug="test",
            generated_at="2026-01-01T00:00:00Z",
        )
        assert graph.generated_at == "2026-01-01T00:00:00Z"


# ===================================================================
# 3. Round-trip: construct → serialize → deserialize → verify
# ===================================================================


class TestCourseGraphRoundTrip:
    """Validate that CourseGraph survives a full serialization cycle."""

    def test_round_trip_via_model_dump(self, sample_graph: CourseGraph) -> None:
        """model_dump() → model_validate() produces an equivalent graph."""
        data = sample_graph.model_dump()
        restored = CourseGraph.model_validate(data)

        assert restored.course_slug == sample_graph.course_slug
        assert restored.learning_objectives == sample_graph.learning_objectives
        assert restored.key_concepts == sample_graph.key_concepts
        assert restored.prerequisites == sample_graph.prerequisites
        assert restored.generated_at == sample_graph.generated_at
        assert len(restored.modules) == len(sample_graph.modules)

        assert restored.specification.course_context == (
            sample_graph.specification.course_context
        )
        assert restored.specification.primary_language == (
            sample_graph.specification.primary_language
        )

        for orig, rest in zip(sample_graph.modules, restored.modules, strict=False):
            assert rest.title == orig.title
            assert rest.duration_weeks == orig.duration_weeks
            assert rest.topics == orig.topics

    def test_round_trip_via_json(self, sample_graph: CourseGraph) -> None:
        """model_dump_json() → model_validate_json() produces equivalent."""
        json_str = sample_graph.model_dump_json(indent=2)
        assert isinstance(json_str, str)
        restored = CourseGraph.model_validate_json(json_str)
        assert restored.course_slug == sample_graph.course_slug
        assert (
            restored.specification.course_context
            == sample_graph.specification.course_context
        )

    def test_json_structure_matches_schema(self, sample_graph: CourseGraph) -> None:
        """The JSON output contains all expected keys at the top level."""
        data = json.loads(sample_graph.model_dump_json())
        expected_keys = {
            "specification", "course_slug", "learning_objectives",
            "key_concepts", "prerequisites", "modules", "generated_at",
        }
        assert set(data.keys()) == expected_keys
        spec = data["specification"]
        assert "course_context" in spec
        assert "primary_language" in spec

    def test_module_json_round_trip(self, sample_module: ModuleSummary) -> None:
        """ModuleSummary round-trips correctly."""
        json_str = sample_module.model_dump_json()
        restored = ModuleSummary.model_validate_json(json_str)
        assert restored.title == sample_module.title
        assert restored.duration_weeks == sample_module.duration_weeks
        assert restored.topics == sample_module.topics


# ===================================================================
# 4. CourseGraph composes CourseSpecification (no duplication)
# ===================================================================


class TestCourseGraphComposition:
    """Verify CourseGraph composes (does not duplicate) CourseSpecification."""

    def test_no_course_context_field_on_course_graph(self) -> None:
        """CourseGraph must NOT have its own course_context field."""
        assert "course_context" not in CourseGraph.model_fields

    def test_no_primary_language_field_on_course_graph(self) -> None:
        """CourseGraph must NOT have its own primary_language field."""
        assert "primary_language" not in CourseGraph.model_fields

    def test_specification_field_is_present(self) -> None:
        """The specification field exists as a composition field."""
        assert "specification" in CourseGraph.model_fields

    def test_specification_is_not_duplicated_in_json(
        self, sample_graph: CourseGraph
    ) -> None:
        """JSON output nests spec fields under specification, not top level."""
        data = json.loads(sample_graph.model_dump_json())
        for key in ("course_context", "primary_language"):
            assert key not in data, f"'{key}' must not appear at top level"


# ===================================================================
# 5. export-course-graph command via OutputExportTool
# ===================================================================


class TestExportCourseGraphCommand:
    """Test the export-course-graph command end-to-end."""

    @pytest.fixture(autouse=True)
    def setup_tmp_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redirect _PROJECT_ROOT in both tool.py and file_writer.py."""
        import src.exporters.file_writer as fw_module
        import src.exporters.tool as tool_module

        monkeypatch.setattr(tool_module, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(fw_module, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "output").mkdir(exist_ok=True)
        self.tmp_root = tmp_path

    def test_export_course_graph_writes_json(self) -> None:
        """The command writes a valid course_graph.json file."""
        tool = OutputExportTool(force=True)
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test Course",
            "course_slug": "test-course",
            "specification": {
                "course_context": "A test course context.",
                "primary_language": "Rust",
            },
            "learning_objectives": ["Learn Rust", "Build CLI tools"],
            "key_concepts": ["ownership", "borrowing"],
            "prerequisites": ["Basic programming"],
            "modules": [{
                "title": "Getting Started",
                "duration_weeks": 1.0,
                "hours_per_week": 3.0,
                "topics": ["cargo", "hello world"],
            }],
            "run_id": "2026-08-23_120000_test-course",
        })
        result = json.loads(result_str)
        assert result["status"] == "ok"
        graph_path = (
            self.tmp_root / "output" / "2026-08-23_120000_test-course"
            / "course_graph.json"
        )
        assert graph_path.exists()
        data = json.loads(graph_path.read_text())
        assert data["course_slug"] == "test-course"
        assert data["specification"]["primary_language"] == "Rust"
        assert len(data["modules"]) == 1
        assert data["modules"][0]["title"] == "Getting Started"

    def test_export_without_run_id_writes_to_output_root(self) -> None:
        """Without run_id, writes to output/course_graph.json directly."""
        tool = OutputExportTool(force=True)
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test",
            "course_slug": "test",
            "specification": {
                "course_context": "Context.",
                "primary_language": "Go",
            },
        })
        result = json.loads(result_str)
        assert result["status"] == "ok"
        graph_path = self.tmp_root / "output" / "course_graph.json"
        assert graph_path.exists()

    def test_export_missing_course_name_returns_error(self) -> None:
        """Missing course_name returns error."""
        tool = OutputExportTool()
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_slug": "test",
            "specification": {"course_context": "x", "primary_language": "Py"},
        })
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "course_name" in result["message"]

    def test_export_missing_course_slug_returns_error(self) -> None:
        """Missing course_slug returns error."""
        tool = OutputExportTool()
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test",
            "specification": {"course_context": "x", "primary_language": "Py"},
        })
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "course_slug" in result["message"]

    def test_export_missing_specification_returns_error(self) -> None:
        """Missing specification returns error."""
        tool = OutputExportTool()
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test",
            "course_slug": "test",
        })
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "specification" in result["message"]

    def test_export_invalid_specification_returns_error(self) -> None:
        """Invalid specification dict returns error."""
        tool = OutputExportTool()
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test",
            "course_slug": "test",
            "specification": {"wrong_field": True},
        })
        result = json.loads(result_str)
        assert result["status"] == "error"

    def test_export_invalid_module_returns_error(self) -> None:
        """An invalid module entry returns error."""
        tool = OutputExportTool()
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test",
            "course_slug": "test",
            "specification": {"course_context": "x", "primary_language": "Py"},
            "modules": [{"title": "Missing duration"}],
        })
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "Invalid module" in result["message"]

    def test_export_empty_lists_are_accepted(self) -> None:
        """Empty lists for optional fields work fine."""
        tool = OutputExportTool(force=True)
        result_str = tool._handle_export_course_graph({
            "command": "export-course-graph",
            "course_name": "Test",
            "course_slug": "test",
            "specification": {"course_context": "x", "primary_language": "Py"},
            "learning_objectives": [],
            "key_concepts": [],
            "prerequisites": [],
            "modules": [],
        })
        result = json.loads(result_str)
        assert result["status"] == "ok"


# ===================================================================
# 6. course_graph.json written alongside README.md
# ===================================================================


class TestManifestCourseGraphIntegration:
    """Verify manifest writes course_graph.json when CourseGraph provided."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redirect output to tmp_path for isolated testing."""
        import src.exporters.file_writer as fw_module
        import src.exporters.manifest as manifest_module
        monkeypatch.setattr(fw_module, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            fw_module, "OUTPUT_PATHS", OutputPathConfig(root=tmp_path)
        )
        monkeypatch.setattr(manifest_module, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "output").mkdir(exist_ok=True)
        self.tmp_root = tmp_path

    def test_course_graph_written_alongside_readme(
        self, sample_graph: CourseGraph
    ) -> None:
        """When course_graph is provided, it is written alongside README."""
        readme_path = update_output_manifest(
            course_name="Test Course", course_graph=sample_graph
        )
        assert readme_path.exists()
        assert readme_path.name == "README.md"
        graph_path = readme_path.parent / "course_graph.json"
        assert graph_path.exists()
        data = json.loads(graph_path.read_text())
        assert data["course_slug"] == sample_graph.course_slug
        assert data["specification"]["primary_language"] == (
            sample_graph.specification.primary_language
        )

    def test_no_course_graph_when_none_passed(self) -> None:
        """When course_graph is None, no JSON file is written."""
        graph_path = self.tmp_root / "output" / "course_graph.json"
        if graph_path.exists():
            graph_path.unlink()
        update_output_manifest(course_name="Test Course", course_graph=None)
        assert not graph_path.exists()

    def test_readme_still_written_without_course_graph(self) -> None:
        """README.md is always written, even without course_graph."""
        readme_path = update_output_manifest(course_name="Test Course")
        assert readme_path.exists()
        assert "Syllabus Swarm" in readme_path.read_text()


# ===================================================================
# 7. Models can be imported by other modules (integration)
# ===================================================================


class TestModelImports:
    """Verify models are importable from expected entry points."""

    def test_models_importable_from_src_models(self) -> None:
        """CourseGraph and ModuleSummary importable from src.models."""
        from src.models import CourseGraph, ModuleSummary
        assert CourseGraph is not None
        assert ModuleSummary is not None

    def test_course_specification_importable_from_main(self) -> None:
        """CourseSpecification remains importable from src.main."""
        spec = CourseSpecification(
            course_context="test", primary_language="Python"
        )
        assert spec.course_context == "test"

    def test_course_graph_composes_course_spec(self) -> None:
        """CourseGraph constructed with CourseSpecification from main."""
        spec = CourseSpecification(
            course_context="test context", primary_language="Go"
        )
        graph = CourseGraph(specification=spec, course_slug="test-go")
        assert graph.specification == spec


# ===================================================================
# 8. hours_per_week field tests
# ===================================================================


class TestHoursPerWeek:
    """Validate the hours_per_week field on ModuleSummary."""

    def test_hours_per_week_is_required(self) -> None:
        """hours_per_week must be provided."""
        with pytest.raises(ValidationError):
            ModuleSummary(
                title="Test", duration_weeks=1.0, topics=["a"]
            )

    def test_negative_hours_is_invalid(self) -> None:
        """hours_per_week must be >= 0."""
        with pytest.raises(ValidationError):
            ModuleSummary(
                title="Test", duration_weeks=1.0,
                hours_per_week=-1.0, topics=["a"]
            )

    def test_zero_hours_is_valid(self) -> None:
        """Zero hours (e.g. self-study module) is allowed."""
        m = ModuleSummary(
            title="Self-Study", duration_weeks=1.0, hours_per_week=0.0
        )
        assert m.hours_per_week == 0.0

    def test_high_hours_allowed(self) -> None:
        """High contact-hour bootcamps (40h/week) are allowed."""
        m = ModuleSummary(
            title="Bootcamp Sprint",
            duration_weeks=1.0,
            hours_per_week=40.0,
            topics=["intensive"],
        )
        assert m.hours_per_week == 40.0

    def test_total_effort_calculable(self) -> None:
        """Total effort = duration_weeks * hours_per_week."""
        m = ModuleSummary(
            title="Full Module", duration_weeks=3.0, hours_per_week=4.0,
            topics=["a", "b", "c"],
        )
        total = m.duration_weeks * m.hours_per_week
        assert total == 12.0

    def test_roundtrip_preserves_hours_per_week(
        self, sample_module: ModuleSummary
    ) -> None:
        """hours_per_week survives serialization/deserialization."""
        data = sample_module.model_dump()
        restored = ModuleSummary.model_validate(data)
        assert restored.hours_per_week == sample_module.hours_per_week
