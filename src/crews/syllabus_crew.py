"""
syllabus_crew.py — Syllabus Generation Crew
============================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)

Wires together the Curriculum Architect agent and the syllabus-generation
task into a CrewAI Crew, handles execution, and saves the result to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from crewai import Agent, Crew, Process

from src.agents.curriculum_architect import get_architect
from src.tasks.syllabus_generation import create_syllabus_task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent.parent / "output" / "syllabus"


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


# ---------------------------------------------------------------------------
# Crew runner
# ---------------------------------------------------------------------------


def run_syllabus_crew(
    course_name: str,
    *,
    verbose: bool = False,
    agent: Optional[Agent] = None,
) -> Path:
    """Run the Curriculum Architect crew and save the generated syllabus.

    Parameters
    ----------
    course_name : str
        The course title / topic to generate a syllabus for.
    verbose : bool
        Enable detailed agent and task logging.
    agent : Agent or None
        Pre-built Curriculum Architect agent; lazily created when None.

    Returns
    -------
    Path
        Path to the saved Markdown output file.

    Raises
    ------
    RuntimeError
        If the API key is missing or the crew fails to produce output.
    """
    # --- 1. Build or retrieve the agent ---
    if agent is None:
        architect = get_architect(verbose=verbose)
    else:
        architect = agent

    # --- 2. Build the task ---
    task = create_syllabus_task(agent=architect, course_name=course_name)

    # --- 3. Assemble the crew ---
    crew = Crew(
        agents=[architect],
        tasks=[task],
        process=Process.sequential,
        verbose=verbose,
    )

    # --- 4. Run ---
    result = crew.kickoff()

    # --- 5. Validate result ---
    raw_output = str(result).strip()

    if not raw_output:
        raise RuntimeError(
            "Crew completed but produced no output. "
            "Check your API key, model availability, and network connectivity."
        )

    # --- 6. Save output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(course_name)
    output_path = OUTPUT_DIR / f"{safe_name}.md"

    output_path.write_text(raw_output, encoding="utf-8")

    return output_path