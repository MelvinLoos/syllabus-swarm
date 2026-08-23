"""
test_file_writer.py — Comprehensive tests for src/exporters/file_writer.py
===========================================================================

Tests cover every public function and the internal sanitiser, using
pytest's ``tmp_path`` fixture so no real files are ever touched.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from src.exporters.file_writer import (
    FileWriteError,
    OutputPathConfig,
    _sanitize_filename,
    write_directory_tree,
    write_file,
    write_lab_file,
    write_rubric,
    write_syllabus,
)

# ===================================================================
# _sanitize_filename
# ===================================================================


class TestSanitizeFilename:
    """Tests for the internal filename sanitiser."""

    # -- Basic transformations ------------------------------------------

    def test_replaces_illegal_characters_with_dash(self) -> None:
        """Characters like < > : " / \\ | ? * become dashes."""
        result = _sanitize_filename('file<name>with|illegal*chars?"')
        # Trailing dash is stripped by edge-trimming logic
        assert result == "file-name-with-illegal-chars"

    def test_replaces_spaces_with_underscores_by_default(self) -> None:
        """Spaces become underscores when replace_spaces=True (default)."""
        result = _sanitize_filename("Data Science with Python")
        assert result == "Data_Science_with_Python"

    def test_preserves_spaces_when_requested(self) -> None:
        """Spaces are kept when replace_spaces=False."""
        result = _sanitize_filename("Data Science with Python", replace_spaces=False)
        assert result == "Data Science with Python"

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace is removed."""
        result = _sanitize_filename("  spaced  course  ")
        assert result == "spaced__course"

    # -- Control characters ---------------------------------------------

    def test_strips_control_characters(self) -> None:
        """Control characters (\\x00-\\x1f) are replaced with dashes."""
        result = _sanitize_filename("\x00\x1fhello\x0bworld")
        assert result == "hello-world"

    # -- Empty / whitespace-only ----------------------------------------

    def test_empty_string_returns_untitled(self) -> None:
        """An empty string yields 'untitled'."""
        assert _sanitize_filename("") == "untitled"

    def test_whitespace_only_returns_untitled(self) -> None:
        """A whitespace-only string yields 'untitled'."""
        assert _sanitize_filename("   ") == "untitled"

    def test_only_illegal_chars_returns_untitled(self) -> None:
        """A string composed entirely of illegal characters yields 'untitled'."""
        result = _sanitize_filename("<>:\"/\\|?*")
        assert result == "untitled"

    # -- Separator collapsing -------------------------------------------

    def test_collapses_runs_of_underscores(self) -> None:
        """Runs of more than 2 underscores are collapsed to 2."""
        result = _sanitize_filename("hello_____world")
        assert result == "hello__world"

    def test_collapses_runs_of_dashes(self) -> None:
        """Runs of more than 2 dashes are collapsed to 2."""
        result = _sanitize_filename("hello-----world")
        assert result == "hello--world"

    def test_collapses_mixed_separator_runs(self) -> None:
        """Mixed runs of underscores and dashes are collapsed per character."""
        result = _sanitize_filename("a___---___b")
        assert result == "a__--__b"

    # -- Edge trimming --------------------------------------------------

    def test_strips_leading_separators(self) -> None:
        """Leading dashes/underscores introduced by sanitisation are removed."""
        result = _sanitize_filename("???hello")
        assert result == "hello"

    def test_strips_trailing_separators(self) -> None:
        """Trailing dashes/underscores introduced by sanitisation are removed."""
        result = _sanitize_filename("hello???")
        assert result == "hello"

    def test_strips_both_edge_separators(self) -> None:
        """Both leading and trailing separators are stripped."""
        result = _sanitize_filename("???hello???")
        assert result == "hello"

    # -- Real-world examples --------------------------------------------

    def test_handles_colon_in_course_name(self) -> None:
        """Colons are replaced with dashes; ampersands are preserved."""
        result = _sanitize_filename("Data Science: ML & AI 101")
        # & is not in the illegal character set, so it stays
        assert result == "Data_Science-_ML_&_AI_101"

    def test_handles_forward_slash(self) -> None:
        """Forward slashes become dashes."""
        result = _sanitize_filename("a/b/c")
        assert result == "a-b-c"

    def test_handles_backslash(self) -> None:
        """Backslashes become dashes."""
        result = _sanitize_filename("a\\b\\c")
        assert result == "a-b-c"

    def test_handles_ampersand(self) -> None:
        """Ampersands are preserved (not in the illegal character set)."""
        result = _sanitize_filename("R&D")
        assert result == "R&D"


# ===================================================================
# write_file
# ===================================================================


class TestWriteFile:
    """Tests for the single-file writer.

    Patches ``_PROJECT_ROOT`` to *tmp_path* so that all writes are
    contained within the temporary directory and the "outside project
    root" guard does not interfere.
    """

    @pytest.fixture(autouse=True)
    def _patch_root(self, tmp_path: Path) -> None:
        """Redirect _PROJECT_ROOT to the test's temporary directory."""
        import src.exporters.file_writer as fw

        self._original_root = fw._PROJECT_ROOT
        fw._PROJECT_ROOT = tmp_path.resolve()
        self._tmp = tmp_path
        yield
        fw._PROJECT_ROOT = self._original_root

    def test_writes_content_to_file(self) -> None:
        """Content is written to the specified path."""
        dest = self._tmp / "test.md"
        result = write_file(dest, "# Hello\n")
        assert result == dest.resolve()
        assert dest.read_text() == "# Hello\n"

    def test_creates_parent_directories(self) -> None:
        """Missing parent directories are created automatically."""
        dest = self._tmp / "deep" / "nested" / "dir" / "file.txt"
        write_file(dest, "content")
        assert dest.exists()
        assert dest.read_text() == "content"

    def test_returns_resolved_path(self) -> None:
        """The returned Path is absolute and resolved."""
        dest = self._tmp / "sub" / "file.txt"
        result = write_file(dest, "data")
        assert result.is_absolute()
        assert result == dest.resolve()

    def test_coerces_non_string_content(self) -> None:
        """Non-string content is converted via str()."""
        dest = self._tmp / "num.txt"
        write_file(dest, 42)
        assert dest.read_text() == "42"

    # -- Overwrite protection -------------------------------------------

    def test_raises_file_write_error_when_file_exists(self) -> None:
        """FileWriteError is raised when the file exists and force=False."""
        dest = self._tmp / "existing.txt"
        dest.write_text("original")
        with pytest.raises(FileWriteError, match="already exists"):
            write_file(dest, "new")

    def test_force_overwrites_existing_file(self) -> None:
        """force=True allows overwriting an existing file."""
        dest = self._tmp / "existing.txt"
        dest.write_text("original")
        write_file(dest, "overwritten", force=True)
        assert dest.read_text() == "overwritten"

    # -- Path validation ------------------------------------------------

    def test_raises_file_write_error_for_path_outside_root(self) -> None:
        """A path outside the project root raises FileWriteError."""
        with pytest.raises(FileWriteError, match="outside the project root"):
            write_file("/tmp/escape.txt", "content")

    # -- Empty content warning ------------------------------------------

    def test_empty_content_triggers_warning(self) -> None:
        """Writing empty/whitespace-only content issues a UserWarning."""
        dest = self._tmp / "empty.md"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_file(dest, "", force=True)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "empty" in str(w[0].message).lower()

    def test_whitespace_only_content_triggers_warning(self) -> None:
        """Whitespace-only content also triggers the warning."""
        dest = self._tmp / "blank.md"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_file(dest, "   \n  ", force=True)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)


# ===================================================================
# write_directory_tree
# ===================================================================


class TestWriteDirectoryTree:
    """Tests for the batch directory-tree writer."""

    @pytest.fixture(autouse=True)
    def _patch_root(self, tmp_path: Path) -> None:
        """Redirect _PROJECT_ROOT to the test's temporary directory."""
        import src.exporters.file_writer as fw

        self._original_root = fw._PROJECT_ROOT
        fw._PROJECT_ROOT = tmp_path.resolve()
        self._tmp = tmp_path
        yield
        fw._PROJECT_ROOT = self._original_root

    def test_writes_all_files_from_dict(self) -> None:
        """Every entry in the dict is written to disk."""
        base = self._tmp / "labs" / "test_course"
        files = {
            "tier1_foundations/starter/lab1.py": "# TODO: implement\n",
            "tier1_foundations/solution/lab1.py": "# Solution\n",
            "tier1_foundations/starter/README.md": "# Lab 1\n",
        }
        written = write_directory_tree(base, files)
        assert len(written) == 3
        for p in written:
            assert p.exists()

    def test_returns_paths_in_insertion_order(self) -> None:
        """The returned list preserves dict insertion order."""
        base = self._tmp / "ordered"
        files = {
            "a.txt": "a",
            "b.txt": "b",
            "c.txt": "c",
        }
        written = write_directory_tree(base, files)
        assert [p.name for p in written] == ["a.txt", "b.txt", "c.txt"]

    def test_empty_dict_triggers_warning(self) -> None:
        """An empty files_dict issues a UserWarning and returns []."""
        base = self._tmp / "empty_tree"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = write_directory_tree(base, {})
            assert result == []
            assert len(w) == 1
            assert "empty" in str(w[0].message).lower()

    def test_overwrite_protection_propagates(self) -> None:
        """If a file already exists, FileWriteError is raised."""
        base = self._tmp / "protected"
        (base / "existing.txt").parent.mkdir(parents=True)
        (base / "existing.txt").write_text("old")
        files = {"existing.txt": "new"}
        with pytest.raises(FileWriteError):
            write_directory_tree(base, files)

    def test_force_allows_overwrite_in_tree(self) -> None:
        """force=True allows overwriting files in the tree."""
        base = self._tmp / "force_tree"
        (base / "file.txt").parent.mkdir(parents=True)
        (base / "file.txt").write_text("old")
        files = {"file.txt": "new"}
        write_directory_tree(base, files, force=True)
        assert (base / "file.txt").read_text() == "new"

    def test_normalises_windows_path_separators(self) -> None:
        """Backslashes in relative paths are normalised to forward slashes."""
        base = self._tmp / "normalised"
        files = {"sub\\dir\\file.txt": "content"}
        written = write_directory_tree(base, files)
        assert len(written) == 1
        assert written[0].exists()


# ===================================================================
# Convenience helpers
# ===================================================================


class TestConvenienceHelpers:
    """Tests for write_syllabus, write_rubric, and write_lab_file."""

    @pytest.fixture(autouse=True)
    def _patch_root_and_output_paths(self, tmp_path: Path) -> None:
        """Redirect _PROJECT_ROOT and OUTPUT_PATHS to a temp directory."""
        import src.exporters.file_writer as fw

        self._original_root = fw._PROJECT_ROOT
        self._original_output_paths = fw.OUTPUT_PATHS
        fw._PROJECT_ROOT = tmp_path.resolve()
        fw.OUTPUT_PATHS = OutputPathConfig(root=tmp_path)
        self._tmp = tmp_path
        yield
        fw._PROJECT_ROOT = self._original_root
        fw.OUTPUT_PATHS = self._original_output_paths

    def test_write_syllabus_creates_correct_path(self) -> None:
        """Syllabus is written to output/syllabus/<sanitized>.md."""
        path = write_syllabus("Data Science with Python", "# Syllabus Content")
        assert path.exists()
        assert path.parent.name == "syllabus"
        assert path.name == "Data_Science_with_Python.md"
        assert "# Syllabus Content" in path.read_text()

    def test_write_rubric_creates_correct_path(self) -> None:
        """Rubric is written to output/rubrics/<sanitized>-rubric.md."""
        path = write_rubric("Data Science with Python", "# Rubric Content")
        assert path.exists()
        assert path.parent.name == "rubrics"
        assert path.name == "Data_Science_with_Python-rubric.md"
        assert "# Rubric Content" in path.read_text()

    def test_write_lab_file_creates_correct_tiered_path(self) -> None:
        """Lab file is written to the correct tiered directory structure."""
        path = write_lab_file(
            "Data Science with Python",
            "tier1_foundations",
            "starter",
            "lab1.py",
            "# TODO: implement\n",
        )
        assert path.exists()
        parts = path.relative_to(self._tmp)
        expected = Path(
            "output/labs/Data_Science_with_Python/"
            "tier1_foundations/starter/lab1.py"
        )
        assert parts == expected

    def test_write_lab_file_rejects_invalid_subfolder(self) -> None:
        """A subfolder other than 'starter' or 'solution' raises ValueError."""
        with pytest.raises(ValueError, match="subfolder must be"):
            write_lab_file("Course", "tier1", "invalid", "file.py", "content")

    def test_write_lab_file_accepts_starter_subfolder(self) -> None:
        """'starter' subfolder is accepted."""
        path = write_lab_file("Course", "tier1", "starter", "file.py", "content")
        assert "starter" in str(path)

    def test_write_lab_file_accepts_solution_subfolder(self) -> None:
        """'solution' subfolder is accepted."""
        path = write_lab_file("Course", "tier1", "solution", "file.py", "content")
        assert "solution" in str(path)

    def test_write_lab_file_sanitizes_filename(self) -> None:
        """The filename component is sanitised."""
        path = write_lab_file(
            "Course", "tier1", "starter", "bad?file:name.py", "content"
        )
        assert path.name == "bad-file-name.py"

    def test_write_syllabus_overwrite_protection(self) -> None:
        """Overwrite protection works for write_syllabus."""
        write_syllabus("Test Course", "first", force=True)
        with pytest.raises(FileWriteError):
            write_syllabus("Test Course", "second")

    def test_write_rubric_overwrite_protection(self) -> None:
        """Overwrite protection works for write_rubric."""
        write_rubric("Test Course", "first", force=True)
        with pytest.raises(FileWriteError):
            write_rubric("Test Course", "second")


# ===================================================================
# OutputPathConfig
# ===================================================================


class TestOutputPathConfig:
    """Tests for the OutputPathConfig dataclass."""

    def test_creates_correct_subdirectories(self, tmp_path: Path) -> None:
        """The config computes the three mandated subdirectories."""
        cfg = OutputPathConfig(root=tmp_path)
        assert cfg.root == tmp_path
        assert cfg.syllabus_dir == tmp_path / "output" / "syllabus"
        assert cfg.labs_dir == tmp_path / "output" / "labs"
        assert cfg.rubrics_dir == tmp_path / "output" / "rubrics"

    def test_is_immutable(self, tmp_path: Path) -> None:
        """The dataclass is frozen and cannot be mutated."""
        cfg = OutputPathConfig(root=tmp_path)
        with pytest.raises(Exception):
            cfg.syllabus_dir = tmp_path / "other"  # type: ignore[misc]

    def test_default_root_is_project_root(self) -> None:
        """The default root points to the project root (contains src/)."""
        cfg = OutputPathConfig()
        assert (cfg.root / "src").is_dir()


# ===================================================================
# FileWriteError
# ===================================================================


class TestFileWriteError:
    """Tests for the FileWriteError exception class."""

    def test_is_exception_subclass(self) -> None:
        """FileWriteError inherits from Exception."""
        assert issubclass(FileWriteError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """FileWriteError can be raised and caught normally."""
        with pytest.raises(FileWriteError, match="test message"):
            raise FileWriteError("test message")

    def test_message_is_preserved(self) -> None:
        """The error message is accessible via str()."""
        try:
            raise FileWriteError("custom error text")
        except FileWriteError as e:
            assert str(e) == "custom error text"
