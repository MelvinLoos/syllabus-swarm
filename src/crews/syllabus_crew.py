"""
syllabus_crew.py — Full Syllabus + Labs Orchestration Crew
==========================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)
Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Wires together two agents into a sequential CrewAI Crew:
  1. **Curriculum Architect** — generates a Humanics-aligned syllabus in
     Markdown and saves it to ``output/syllabus/<course_name>.md``.
  2. **Lab & Project Developer** — receives the syllabus as context and
     generates tiered coding labs saved to ``output/labs/<course_name>/``.

The crew runs sequentially so the Lab Developer can use the syllabus
output as grounding context for designing relevant, scaffolded exercises.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from crewai import Agent, Crew, Process

from src.agents.curriculum_architect import get_architect
from src.agents.lab_developer import get_lab_developer
from src.tasks.syllabus_generation import create_syllabus_generation_task
from src.tasks.lab_generation import create_lab_generation_task
from src.exporters import (
    write_syllabus,
    write_file,
    write_lab_file,
    write_directory_tree,
    update_output_manifest,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT: Path = _PROJECT_ROOT / "output"

# Tier names used for the lab directory scaffolding.
_TIERS: list[Tuple[str, str]] = [
    ("tier1_foundations", "Tier 1 — Foundations"),
    ("tier2_application", "Tier 2 — Application"),
    ("tier3_architecture", "Tier 3 — Architecture"),
]


def _sanitize_filename(course_name: str) -> str:
    """Convert a course name into a safe filesystem name."""
    return (
        course_name.strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )


def _create_lab_scaffolding(labs_base_path: Path) -> Path:
    """Create the tiered lab directory scaffolding under *labs_base_path*."""
    base = labs_base_path
    base.mkdir(parents=True, exist_ok=True)

    for tier_dir_name, tier_label in _TIERS:
        tier_path = base / tier_dir_name
        starter_path = tier_path / "starter"
        solution_path = tier_path / "solution"
        starter_path.mkdir(parents=True, exist_ok=True)
        solution_path.mkdir(parents=True, exist_ok=True)

        (starter_path / ".gitkeep").touch(exist_ok=True)
        (solution_path / ".gitkeep").touch(exist_ok=True)

        tier_readme = tier_path / "README.md"
        if not tier_readme.exists():
            tier_readme.write_text(
                f"# {tier_label}\n\n"
                f"Labs for this tier will be generated here.\n\n"
                f"- **starter/** — Scaffolded exercises with TODO markers.\n"
                f"- **solution/** — Fully-commented reference implementations.\n",
                encoding="utf-8",
            )

    return base


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CrewResult:
    """Holds the result of a full syllabus + labs crew run."""

    def __init__(
        self,
        syllabus_path: Path,
        labs_base_path: Path,
        syllabus_ok: bool = True,
        labs_ok: bool = True,
        syllabus_error: str | None = None,
        labs_error: str | None = None,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.syllabus_path = syllabus_path
        self.labs_base_path = labs_base_path
        self.syllabus_ok = syllabus_ok
        self.labs_ok = labs_ok
        self.syllabus_error = syllabus_error
        self.labs_error = labs_error
        self.manifest_path = manifest_path

    @property
    def all_succeeded(self) -> bool:
        return self.syllabus_ok and self.labs_ok


# ---------------------------------------------------------------------------
# Crew runner
# ---------------------------------------------------------------------------


def _generate_run_id(course_safe_name: str) -> str:
    """Generate a unique, human-readable run identifier.

    Format: ``YYYY-MM-DD_HHMMSS_<course_safe_name>``
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{course_safe_name}"


def _find_syllabus_in_dir(resume_dir: Path) -> Path:
    """Locate the syllabus markdown file inside a resume directory.

    Looks for a ``.md`` file inside ``<resume_dir>/syllabus/``.
    Returns the first match found.

    Raises
    ------
    FileNotFoundError
        If the resume directory or syllabus subdirectory doesn't exist,
        or if no ``.md`` file is found.
    """
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    syllabus_subdir = resume_dir / "syllabus"
    if not syllabus_subdir.exists():
        raise FileNotFoundError(
            f"No 'syllabus/' subdirectory found in resume directory: {resume_dir}"
        )

    md_files = sorted(syllabus_subdir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No .md syllabus file found in: {syllabus_subdir}"
        )

    return md_files[0]


def run_syllabus_crew(
    course_name: str,
    *,
    verbose: bool = False,
    architect_agent: Optional[Agent] = None,
    lab_dev_agent: Optional[Agent] = None,
    skip_labs: bool = False,
    resume_dir: str | Path | None = None,
) -> CrewResult:
    """Run both agents sequentially and return a full result summary.

    Execution order:
      1. Curriculum Architect generates a syllabus (or loads from disk if
         *resume_dir* is provided).
      2. Lab & Project Developer processes the syllabus to generate tiered labs.

    All output is scoped under ``output/<run_id>/`` where *run_id* is a
    timestamp + course-slug combination, ensuring every pipeline run
    produces a unique, non-overlapping directory.

    Parameters
    ----------
    course_name : str
        The course title / topic to generate content for.
    verbose : bool
        Enable detailed agent and task logging.
    architect_agent : Agent or None
        Pre-built Curriculum Architect agent; lazily created when None.
    lab_dev_agent : Agent or None
        Pre-built Lab Developer agent; lazily created when None.
    skip_labs : bool
        If True, only run the Curriculum Architect (backward-compatible).
    resume_dir : str, Path, or None
        Path to a previous run directory (e.g.
        ``output/2026-08-22_153000_Course_Name``). When provided, the
        Curriculum Architect is **skipped** and the existing syllabus is
        loaded from disk instead. This saves API costs and enables a
        human-in-the-loop workflow where the syllabus can be manually
        edited before generating labs.

    Returns
    -------
    CrewResult
        Container with paths, status flags, and any error messages.
    """
    safe_name = _sanitize_filename(course_name)

    # ── 0. Handle resume mode ──────────────────────────────────────────
    syllabus_ok = False
    syllabus_error: str | None = None
    syllabus_raw: str = ""
    syllabus_path: Path

    if resume_dir is not None:
        resume_path = Path(resume_dir)
        try:
            syllabus_path = _find_syllabus_in_dir(resume_path)
            syllabus_raw = syllabus_path.read_text(encoding="utf-8").strip()

            if not syllabus_raw:
                raise RuntimeError(
                    f"Syllabus file is empty: {syllabus_path}"
                )

            syllabus_ok = True
            if verbose:
                print(f"  📄  Loaded syllabus from: {syllabus_path}")
                print(f"      ({len(syllabus_raw):,} characters)")

        except Exception as exc:
            syllabus_error = str(exc)
            # Create a fallback path so the rest of the function has
            # something to work with.
            syllabus_path = OUTPUT_ROOT / "resume_failed" / f"{safe_name}.md"
            if verbose:
                print(f"  ❌  Failed to load syllabus from resume dir: {exc}",
                      file=sys.stderr)

        # When resuming, we still create a fresh run directory for the
        # labs output (so each resume produces its own timestamped output).
        run_id = _generate_run_id(safe_name)
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        labs_dir = run_dir / "labs"
        labs_dir.mkdir(parents=True, exist_ok=True)
        labs_base_path = labs_dir / safe_name

    else:
        # ── Fresh run: create new run directory ────────────────────────
        run_id = _generate_run_id(safe_name)
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        syllabus_dir = run_dir / "syllabus"
        syllabus_dir.mkdir(parents=True, exist_ok=True)
        syllabus_path = syllabus_dir / f"{safe_name}.md"

        labs_dir = run_dir / "labs"
        labs_dir.mkdir(parents=True, exist_ok=True)
        labs_base_path = labs_dir / safe_name

        # ── 1. Curriculum Architect ────────────────────────────────────
        if architect_agent is None:
            architect = get_architect(verbose=verbose)
        else:
            architect = architect_agent

        syllabus_task = create_syllabus_generation_task(
            agent=architect, course_name=course_name
        )

        try:
            architect_crew = Crew(
                agents=[architect],
                tasks=[syllabus_task],
                process=Process.sequential,
                verbose=verbose,
            )
            architect_result = architect_crew.kickoff()
            syllabus_raw = (
                architect_result.raw
                if hasattr(architect_result, "raw")
                else str(architect_result)
            ).strip()

            if not syllabus_raw:
                raise RuntimeError(
                    "Curriculum Architect produced no output. "
                    "Check your API key, model availability, and network."
                )

            write_file(syllabus_path, syllabus_raw, force=True)
            syllabus_ok = True

        except Exception as exc:
            syllabus_error = str(exc)
            write_file(
                syllabus_path,
                f"# {course_name} — Syllabus Generation Failed\n\n"
                f"**Error:** {syllabus_error}\n",
                force=True,
            )

    # ── 2. Lab & Project Developer ─────────────────────────────────────
    labs_ok = False
    labs_error: str | None = None

    if skip_labs:
        _create_lab_scaffolding(labs_base_path)
        labs_ok = True
    else:
        _create_lab_scaffolding(labs_base_path)
        lab_dev = None

        try:
            if lab_dev_agent is not None:
                lab_dev = lab_dev_agent
            else:
                lab_dev = get_lab_developer(verbose=verbose)
        except RuntimeError as exc:
            labs_error = str(exc)

        if lab_dev is not None and syllabus_raw:
            lab_task = create_lab_generation_task(
                agent=lab_dev,
                course_name=course_name,
                syllabus_context=syllabus_raw,
                verbose=verbose,
            )

            try:
                lab_crew = Crew(
                    agents=[lab_dev],
                    tasks=[lab_task],
                    process=Process.sequential,
                    verbose=verbose,
                )
                lab_result = lab_crew.kickoff()
                lab_raw = (
                    lab_result.raw
                    if hasattr(lab_result, "raw")
                    else str(lab_result)
                ).strip()

                if not lab_raw:
                    labs_error = "Lab Developer produced no output."
                else:
                    lab_readme = labs_base_path / "README.md"
                    write_file(lab_readme, lab_raw, force=True)
                    labs_ok = True

            except Exception as exc:
                labs_error = str(exc)
        elif not syllabus_raw:
            labs_error = (
                "Skipped — Curriculum Architect produced no syllabus "
                "to use as context."
            )

        if labs_error and not (labs_base_path / "README.md").exists():
            write_file(
                labs_base_path / "README.md",
                f"# {course_name} — Lab Generation Failed\n\n"
                f"**Error:** {labs_error}\n",
                force=True,
            )

    # ── 3. Generate output manifest ────────────────────────────────────
    try:
        manifest_path = update_output_manifest(
            course_name,
            syllabus_path=syllabus_path,
            labs_base_path=labs_base_path,
        )
    except Exception as exc:
        if verbose:
            print(f"  [Warning] Manifest generation failed: {exc}", file=sys.stderr)
        manifest_path = None

    # ── 4. Return combined result ──────────────────────────────────────
    return CrewResult(
        syllabus_path=syllabus_path,
        labs_base_path=labs_base_path,
        syllabus_ok=syllabus_ok,
        labs_ok=labs_ok,
        syllabus_error=syllabus_error,
        labs_error=labs_error,
        manifest_path=manifest_path,
    )