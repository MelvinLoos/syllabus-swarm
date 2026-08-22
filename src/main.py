#!/usr/bin/env python3
"""
main.py — Syllabus Swarm CLI Entry Point
=========================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)
Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Orchestrates the full syllabus-and-labs generation pipeline:

  0. Runs the **Intake Specialist** agent to interview the user and gather
     rich course context mapped to Dutch SBB Kwalificatiedossiers.
  1. Runs the **Curriculum Architect** agent to generate a Humanics-aligned
     Markdown syllabus saved to ``output/syllabus/<course_name>.md``.
  2. Runs the **Lab & Project Developer** agent using the syllabus as
     context to generate tiered coding labs saved under
     ``output/labs/<course_name>/``.
  3. Prints a clear success/failure summary for all agents.

Usage
-----
  python src/main.py "Data Science with Python"
  python -m src.main "Full-Stack Web Development"
  python src/main.py             # interactive prompt
  python src/main.py "ML Basics" --skip-labs  # syllabus only
  python src/main.py "ML Basics" --resume-from output/2026-08-22_153000_ML_Basics

Environment
-----------
  Requires OPENROUTER_API_KEY in .env (copy from .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `from src.…` imports work
# regardless of whether the script is invoked as `python src/main.py` or
# `python -m src.main`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables *before* any internal imports that read them.
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

from crewai import Crew, Process, Task
from pydantic import BaseModel, Field

from src.agents.intake_specialist import get_intake_specialist
from src.crews.syllabus_crew import CrewResult, run_syllabus_crew


# ---------------------------------------------------------------------------
# Pydantic structured-output model for the Intake Specialist
# ---------------------------------------------------------------------------


class CourseSpecification(BaseModel):
    """Structured output from the Intake Specialist synthesis step.

    This model ensures the LLM returns both the rich pedagogical context
    AND the exact programming language, eliminating the need for brittle
    regex-based language detection downstream.
    """

    course_context: str = Field(
        description="The rich pedagogical context and requirements "
        "synthesised from the user's answers."
    )
    primary_language: str = Field(
        description="The exact programming language to be used for labs "
        "(e.g., 'JavaScript', 'Python', 'TypeScript', 'Java', 'Go', 'Rust')."
    )

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _gather_course_name() -> str:
    """Return the course name from CLI args or interactive input."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    try:
        return input("📘 Enter course name / topic: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  No input provided. Exiting.", file=sys.stderr)
        sys.exit(1)


def _validate_name(name: str) -> str:
    """Ensure the course name is non-empty and reasonable."""
    if not name:
        print("❌ Course name cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(name) > 200:
        print("❌ Course name is too long (max 200 characters).", file=sys.stderr)
        sys.exit(1)
    return name


def _should_skip_labs() -> bool:
    """Check CLI arguments for --skip-labs flag."""
    return "--skip-labs" in sys.argv


def _get_resume_dir() -> str | None:
    """Extract the --resume-from value from CLI arguments, or None."""
    try:
        idx = sys.argv.index("--resume-from")
    except ValueError:
        return None
    if idx + 1 >= len(sys.argv):
        print("❌ --resume-from requires a directory path.", file=sys.stderr)
        sys.exit(1)
    return sys.argv[idx + 1]


def _clean_cli_flags() -> None:
    """Remove known CLI flags from sys.argv so they don't pollute the course name."""
    for flag in ("--skip-labs", "--resume-from"):
        while flag in sys.argv:
            idx = sys.argv.index(flag)
            # Remove the flag and its value (if it has one)
            if flag == "--resume-from" and idx + 1 < len(sys.argv):
                sys.argv.pop(idx)  # value
            sys.argv.pop(idx)  # flag


# ---------------------------------------------------------------------------
# Intake Specialist — interactive interview loop
# ---------------------------------------------------------------------------


def _run_intake(course_name: str, *, verbose: bool = False) -> tuple[str, str]:
    """Run the Intake Specialist to gather rich course context.

    Steps:
      1. Send the course name to the Intake Specialist, who replies with
         2-3 targeted clarifying questions.
      2. Display the questions and capture the user's multi-line response.
      3. Send the course name + user answers back to the Intake Specialist
         for synthesis into a ``CourseSpecification`` structured output.

    Parameters
    ----------
    course_name : str
        The initial course name / topic from the user.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    tuple[str, str]
        A ``(course_context, primary_language)`` pair extracted from the
        Intake Specialist's structured output.
    """
    intake_agent = get_intake_specialist(verbose=verbose)

    # ── Step 1: Ask clarifying questions ────────────────────────────────
    question_task = Task(
        description=(
            f"The user wants a syllabus for the following course:\n\n"
            f"**Course Name:** {course_name}\n\n"
            f"Your task: Ask 2-3 concise, targeted clarifying questions "
            f"about:\n"
            f"1. The tech stack and tooling (languages, frameworks, "
            f"platforms, version control, deployment tools)\n"
            f"2. Which kerntaken to emphasise: planning (P1-K1), designing "
            f"(P2-K1), building (P3-K1), and/or testing (P4-K1) software\n"
            f"3. The student profile: BOL or BBL pathway, year level "
            f"(1, 2, or 3), and BPV (internship) readiness\n\n"
            f"Output ONLY the questions — no preamble, no commentary, "
            f"no markdown formatting.  Number them 1, 2, 3.  Keep each "
            f"question to one or two sentences."
        ),
        expected_output=(
            "2-3 numbered clarifying questions about tech stack, "
            "kerntaken focus, and student profile.  No preamble or "
            "commentary — just the questions."
        ),
        agent=intake_agent,
        async_execution=False,
    )

    print("\n🐝  Consulting the Intake Specialist...\n")

    try:
        question_crew = Crew(
            agents=[intake_agent],
            tasks=[question_task],
            process=Process.sequential,
            verbose=verbose,
        )
        question_result = question_crew.kickoff()
        questions = (
            question_result.raw
            if hasattr(question_result, "raw")
            else str(question_result)
        ).strip()
    except Exception as exc:
        print(f"\n⚠️  Intake Specialist failed to generate questions: {exc}")
        print("   Proceeding with bare course name as context.\n")
        return f"Course Name: {course_name}", "Python"

    # ── Step 2: Display questions and capture answers ────────────────────
    print(f"{'─' * 60}")
    print(questions)
    print(f"{'─' * 60}")
    print()
    print("📝  Please type your answers below.")
    print("    Press Enter twice (blank line) when finished.\n")

    user_answers_lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                if user_answers_lines and user_answers_lines[-1] == "":
                    # Two blank lines in a row → done
                    user_answers_lines.pop()  # remove the trailing blank
                    break
            user_answers_lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  Input interrupted.", file=sys.stderr)

    user_answers = "\n".join(user_answers_lines).strip()

    if not user_answers:
        print("\n⚠️  No answers provided. Proceeding with bare course name.\n")
        return f"Course Name: {course_name}", "Python"

    # ── Step 3: Synthesise course context ────────────────────────────────
    synthesis_task = Task(
        description=(
            f"You interviewed the user about their course and received "
            f"the following information.\n\n"
            f"**Course Name:** {course_name}\n\n"
            f"**Your Questions:**\n{questions}\n\n"
            f"**User's Answers:**\n{user_answers}\n\n"
            f"Your task: Synthesise the course name, your questions, and "
            f"the user's answers into a structured ``CourseSpecification`` "
            f"object with two fields:\n\n"
            f"1. ``course_context`` — A rich, structured text block "
            f"(plain text, NOT Markdown) containing:\n"
            f"   - The course name and a one-sentence summary\n"
            f"   - Tech stack and tooling details\n"
            f"   - Kerntaken emphasis (which of P1-K1 through P4-K1 to "
            f"focus on)\n"
            f"   - Student profile (BOL/BBL, year level, BPV readiness)\n"
            f"   - Any additional pedagogical notes or constraints\n\n"
            f"2. ``primary_language`` — The EXACT programming language "
            f"that will be used for coding labs (e.g., 'JavaScript', "
            f"'Python', 'TypeScript', 'Java', 'Go', 'Rust').  This MUST "
            f"be a single language name, not a list or description.\n\n"
            f"The course_context will be passed directly to the "
            f"Curriculum Architect agent who will design the syllabus.  "
            f"The primary_language will determine the file extensions, "
            f"linters, and tooling used in the coding labs."
        ),
        expected_output=(
            "A CourseSpecification object with two fields: "
            "course_context (rich pedagogical context as plain text) and "
            "primary_language (the exact programming language for labs, "
            "e.g. 'JavaScript', 'Python')."
        ),
        output_pydantic=CourseSpecification,
        agent=intake_agent,
        async_execution=False,
    )

    print("\n🐝  Synthesising course context...\n")

    try:
        synthesis_crew = Crew(
            agents=[intake_agent],
            tasks=[synthesis_task],
            process=Process.sequential,
            verbose=verbose,
        )
        synthesis_result = synthesis_crew.kickoff()

        # Extract the typed Pydantic model from the result.
        if hasattr(synthesis_result, "pydantic") and synthesis_result.pydantic is not None:
            spec: CourseSpecification = synthesis_result.pydantic
            course_context = spec.course_context.strip()
            primary_language = spec.primary_language.strip()
        else:
            # Fallback: parse from raw text if pydantic output unavailable.
            course_context = (
                synthesis_result.raw
                if hasattr(synthesis_result, "raw")
                else str(synthesis_result)
            ).strip()
            primary_language = "Python"

        if not course_context:
            print("⚠️  Synthesis produced no output. Using bare course name.\n")
            return f"Course Name: {course_name}", "Python"

        print(f"{'─' * 60}")
        print("📋  Course Context (sent to Curriculum Architect):")
        print(f"{'─' * 60}")
        print(course_context)
        print(f"   Primary Language: {primary_language}")
        print(f"{'─' * 60}\n")

        return course_context, primary_language

    except Exception as exc:
        print(f"\n⚠️  Synthesis failed: {exc}")
        print("   Proceeding with bare course name as context.\n")
        return f"Course Name: {course_name}", "Python"


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(result: CrewResult, course_name: str) -> None:
    """Print a clear success/failure summary for both agents."""
    print(f"\n{'=' * 60}")
    print(f"  🐝  Syllabus Swarm — Results Summary")
    print(f"  Course: {course_name}")
    print(f"{'=' * 60}")

    # ── Syllabus Agent ──────────────────────
    print()
    if result.syllabus_ok:
        print(f"  ✅  Curriculum Architect  —  SUCCESS")
        print(f"      📄  Syllabus: {result.syllabus_path}")
        try:
            size = result.syllabus_path.stat().st_size
            print(f"      📏  Size:     {size:,} bytes")
        except OSError:
            pass
    else:
        print(f"  ❌  Curriculum Architect  —  FAILED")
        if result.syllabus_error:
            print(f"      ↳ {result.syllabus_error}")

    # ── Labs Agent ──────────────────────────
    if result.labs_ok:
        print(f"  ✅  Lab & Project Developer  —  SUCCESS")
        print(f"      📁  Labs:  {result.labs_base_path}/")
        _print_lab_tree(result.labs_base_path)
    else:
        print(f"  ❌  Lab & Project Developer  —  FAILED")
        if result.labs_error:
            print(f"      ↳ {result.labs_error}")

    # ── Manifest ────────────────────────────
    print()
    if result.manifest_path and result.manifest_path.exists():
        print(f"  📋  Output Manifest: {result.manifest_path}")
    else:
        print(f"  ⚠️   Output Manifest not generated.")

    # ── Export summary ──────────────────────
    _print_export_summary(result)

    # ── Overall verdict ─────────────────────
    print()
    if result.all_succeeded:
        print(f"  🎉  All agents completed successfully!")
    elif result.syllabus_ok:
        print(f"  ⚠️   Syllabus generated but labs failed.")
    else:
        print(f"  💥  Both agents failed.")
        print(f"      Check OPENROUTER_API_KEY and network connectivity.")
    print(f"{'=' * 60}\n")


def _print_lab_tree(base: Path, indent: int = 6) -> None:
    """Print a compact tree view of the lab directory structure."""
    prefix = " " * indent
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            print(f"{prefix}▸ {entry.name}/")
            _print_lab_tree(entry, indent + 3)
        else:
            print(f"{prefix}  {entry.name}")


def _print_export_summary(result: CrewResult) -> None:
    """Print a summary of what was exported and where."""
    print()
    print(f"  ── What Was Exported ──")

    # Syllabus
    if result.syllabus_path.exists():
        try:
            sz = result.syllabus_path.stat().st_size
            print(f"  📄  Syllabus → {_fmt_size(sz):>10}  {result.syllabus_path}")
        except OSError:
            print(f"  📄  Syllabus →              {result.syllabus_path}")

    # Labs
    if result.labs_base_path.exists():
        file_count = 0
        total_size = 0
        for f in result.labs_base_path.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                file_count += 1
                try:
                    total_size += f.stat().st_size
                except OSError:
                    pass
        print(
            f"  🧪  Labs     → {_fmt_size(total_size):>10}  "
            f"{result.labs_base_path}/  ({file_count} files)"
        )

    # Manifest
    if result.manifest_path and result.manifest_path.exists():
        try:
            sz = result.manifest_path.stat().st_size
            print(f"  📋  Manifest → {_fmt_size(sz):>10}  {result.manifest_path}")
        except OSError:
            print(f"  📋  Manifest →              {result.manifest_path}")


def _fmt_size(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[idx]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — orchestrate the intake + syllabus + labs pipeline."""

    # --- 1. Gather input -------------------------------------------------
    skip_labs = _should_skip_labs()
    resume_dir = _get_resume_dir()

    # Validate: --skip-labs + --resume-from is a no-op
    if skip_labs and resume_dir:
        print(
            "❌ --skip-labs and --resume-from cannot be used together "
            "(this would be a no-op).",
            file=sys.stderr,
        )
        sys.exit(1)

    _clean_cli_flags()

    course_name = _gather_course_name()
    course_name = _validate_name(course_name)

    print(f"\n{'=' * 60}")
    print(f"  🐝  Syllabus Swarm")
    print(f"  Course:     {course_name}")
    print(f"  Model:      Per-agent via OpenRouter (see .env.example)")
    if resume_dir:
        print(f"  Resume:     {resume_dir}")
    print(f"  Labs:       {'Skip' if skip_labs else 'Generate'}")
    print(f"{'=' * 60}\n")

    # --- 2. Run Intake Specialist (skip if resuming) ---------------------
    if resume_dir:
        # When resuming, we already have a syllabus — no need for intake.
        course_context = f"Course Name: {course_name}"
        primary_language = "Python"
        print("📋  Resume mode — skipping Intake Specialist.\n")
    else:
        course_context, primary_language = _run_intake(course_name, verbose=True)

    # --- 3. Run the crew -------------------------------------------------
    try:
        result = run_syllabus_crew(
            course_context,
            primary_language=primary_language,
            verbose=True,
            skip_labs=skip_labs,
            resume_dir=resume_dir,
        )
    except RuntimeError as exc:
        print(f"\n❌  Fatal Runtime Error: {exc}", file=sys.stderr)
        print("   → Check that OPENROUTER_API_KEY is set in .env.", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\n❌  Fatal Unexpected Error: {exc}", file=sys.stderr)
        sys.exit(3)

    # --- 4. Print summary ------------------------------------------------
    _print_summary(result, course_name)


if __name__ == "__main__":
    main()