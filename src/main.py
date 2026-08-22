#!/usr/bin/env python3
"""
main.py — Syllabus Swarm CLI Entry Point
=========================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)

Orchestrates the full syllabus-generation pipeline:
  1. Accepts a course name/topic via CLI argument or interactive prompt.
  2. Instantiates the Curriculum Architect agent (DeepSeek R1 via OpenRouter).
  3. Creates the syllabus-generation task.
  4. Assembles and runs a CrewAI Crew.
  5. Saves the generated Markdown syllabus to output/syllabus/<course>.md.
  6. Prints a success/failure summary with the output path.

Usage
-----
  python src/main.py "Data Science with Python"
  python -m src.main "Full-Stack Web Development"
  python src/main.py             # interactive prompt

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

from src.crews.syllabus_crew import run_syllabus_crew

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _gather_course_name() -> str:
    """Return the course name from CLI args or interactive input."""
    if len(sys.argv) > 1:
        # Join all arguments in case the user didn't quote multi-word names.
        return " ".join(sys.argv[1:]).strip()

    # Interactive prompt
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — orchestrate the syllabus generation pipeline."""

    # --- 1. Gather input ---
    course_name = _gather_course_name()
    course_name = _validate_name(course_name)

    print(f"\n{'=' * 60}")
    print(f"  🐝  Syllabus Swarm — Curriculum Architect")
    print(f"  Course: {course_name}")
    print(f"  Model:  DeepSeek R1 via OpenRouter")
    print(f"{'=' * 60}\n")

    # --- 2. Run the crew ---
    try:
        output_path = run_syllabus_crew(course_name, verbose=True)
    except RuntimeError as exc:
        print(f"\n❌  Runtime Error: {exc}", file=sys.stderr)
        print("   → Check that OPENROUTER_API_KEY is set in your .env file.", file=sys.stderr)
        print("   → Verify the model is available at https://openrouter.ai/models", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\n❌  Unexpected Error: {exc}", file=sys.stderr)
        print("   → This may be a networking, API, or dependency issue.", file=sys.stderr)
        sys.exit(3)

    # --- 3. Print summary ---
    print(f"\n{'=' * 60}")
    print(f"  ✅  Syllabus generated successfully!")
    print(f"  📄  Output: {output_path}")
    print(f"  📏  Size:   {output_path.stat().st_size:,} bytes")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()