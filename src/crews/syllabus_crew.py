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

from pathlib import Path
from typing import Optional, Tuple

from crewai import Agent, Crew, Process

from src.agents.curriculum_architect import get_architect
from src.agents.lab_developer import get_lab_developer
from src.tasks.syllabus_generation import create_syllabus_task
from src.tasks.lab_generation import create_lab_task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SYLLABUS_OUTPUT_DIR: Path = _PROJECT_ROOT / "output" / "syllabus"
LABS_OUTPUT_DIR: Path = _PROJECT_ROOT / "output" / "labs"

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


def _create_lab_scaffolding(course_safe_name: str) -> Path:
    """Create the tiered lab directory scaffolding under output/labs/."""
    base = LABS_OUTPUT_DIR / course_safe_name
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
    ) -> None:
        self.syllabus_path = syllabus_path
        self.labs_base_path = labs_base_path
        self.syllabus_ok = syllabus_ok
        self.labs_ok = labs_ok
        self.syllabus_error = syllabus_error
        self.labs_error = labs_error

    @property
    def all_succeeded(self) -> bool:
        return self.syllabus_ok and self.labs_ok
# ---------------------------------------------------------------------------
# Crew runner
# ---------------------------------------------------------------------------


def run_syllabus_crew(
    course_name: str,
    *,
    verbose: bool = False,
    architect_agent: Optional[Agent] = None,
    lab_dev_agent: Optional[Agent] = None,
    skip_labs: bool = False,
) -> CrewResult:
    """Run both agents sequentially and return a full result summary.

    Execution order:
      1. Curriculum Architect generates a syllabus.
      2. Lab & Project Developer processes the syllabus to generate tiered labs.

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

    Returns
    -------
    CrewResult
        Container with paths, status flags, and any error messages.
    """
    safe_name = _sanitize_filename(course_name)

    syllabus_dir = SYLLABUS_OUTPUT_DIR
    syllabus_dir.mkdir(parents=True, exist_ok=True)
    syllabus_path = syllabus_dir / f"{safe_name}.md"

    labs_base_path = LABS_OUTPUT_DIR / safe_name

    # ── 1. Curriculum Architect ────────────────────────────────────────
    if architect_agent is None:
        architect = get_architect(verbose=verbose)
    else:
        architect = architect_agent

    syllabus_task = create_syllabus_task(agent=architect, course_name=course_name)

    syllabus_ok = False
    syllabus_error: str | None = None
    syllabus_raw: str = ""

    try:
        architect_crew = Crew(
            agents=[architect],
            tasks=[syllabus_task],
            process=Process.sequential,
            verbose=verbose,
        )
        architect_result = architect_crew.kickoff()
        syllabus_raw = str(architect_result).strip()

        if not syllabus_raw:
            raise RuntimeError(
                "Curriculum Architect produced no output. "
                "Check your API key, model availability, and network."
            )

        syllabus_path.write_text(syllabus_raw, encoding="utf-8")
        syllabus_ok = True

    except Exception as exc:
        syllabus_error = str(exc)
        syllabus_path.write_text(
            f"# {course_name} — Syllabus Generation Failed\n\n"
            f"**Error:** {syllabus_error}\n",
            encoding="utf-8",
        )

    # ── 2. Lab & Project Developer ─────────────────────────────────────
    labs_ok = False
    labs_error: str | None = None

    if skip_labs:
        labs_base_path = _create_lab_scaffolding(safe_name)
        labs_ok = True
    else:
        labs_base_path = _create_lab_scaffolding(safe_name)
        lab_dev = None

        try:
            if lab_dev_agent is not None:
                lab_dev = lab_dev_agent
            else:
                lab_dev = get_lab_developer(verbose=verbose)
        except RuntimeError as exc:
            labs_error = str(exc)

        if lab_dev is not None and syllabus_raw:
            lab_task = create_lab_task(
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
                lab_raw = str(lab_result).strip()

                if not lab_raw:
                    labs_error = "Lab Developer produced no output."
                else:
                    lab_readme = labs_base_path / "README.md"
                    lab_readme.write_text(lab_raw, encoding="utf-8")
                    labs_ok = True

            except Exception as exc:
                labs_error = str(exc)
        elif not syllabus_raw:
            labs_error = (
                "Skipped — Curriculum Architect produced no syllabus "
                "to use as context."
            )

        if labs_error and not (labs_base_path / "README.md").exists():
            (labs_base_path / "README.md").write_text(
                f"# {course_name} — Lab Generation Failed\n\n"
                f"**Error:** {labs_error}\n",
                encoding="utf-8",
            )

    # ── 3. Return combined result ──────────────────────────────────────
    return CrewResult(
        syllabus_path=syllabus_path,
        labs_base_path=labs_base_path,
        syllabus_ok=syllabus_ok,
        labs_ok=labs_ok,
        syllabus_error=syllabus_error,
        labs_error=labs_error,
    )