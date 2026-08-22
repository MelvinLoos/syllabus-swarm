#!/usr/bin/env python3
"""
main.py — Syllabus Swarm CLI Entry Point
=========================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)
Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Orchestrates the full syllabus-and-labs generation pipeline:
  1. Accepts a course name/topic via CLI argument or interactive prompt.
  2. Runs the **Curriculum Architect** agent to generate a Humanics-aligned
     Markdown syllabus saved to ``output/syllabus/<course_name>.md``.
  3. Runs the **Lab & Project Developer** agent using the syllabus as
     context to generate tiered coding labs saved under
     ``output/labs/<course_name>/``.
  4. Prints a clear success/failure summary for both agents.

Usage
-----
  python src/main.py "Data Science with Python"
  python -m src.main "Full-Stack Web Development"
  python src/main.py             # interactive prompt
  python src/main.py "ML Basics" --skip-labs  # syllabus only

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

from src.crews.syllabus_crew import CrewResult, run_syllabus_crew

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — orchestrate the syllabus + labs generation pipeline."""

    # --- 1. Gather input -------------------------------------------------
    skip_labs = _should_skip_labs()
    while "--skip-labs" in sys.argv:
        sys.argv.remove("--skip-labs")

    course_name = _gather_course_name()
    course_name = _validate_name(course_name)

    print(f"\n{'=' * 60}")
    print(f"  🐝  Syllabus Swarm")
    print(f"  Course:     {course_name}")
    print(f"  Model:      DeepSeek R1 via OpenRouter")
    print(f"  Labs:       {'Skip' if skip_labs else 'Generate'}")
    print(f"{'=' * 60}\n")

    # --- 2. Run the crew -------------------------------------------------
    try:
        result = run_syllabus_crew(
            course_name, verbose=True, skip_labs=skip_labs
        )
    except RuntimeError as exc:
        print(f"\n❌  Fatal Runtime Error: {exc}", file=sys.stderr)
        print("   → Check that OPENROUTER_API_KEY is set in .env.", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\n❌  Fatal Unexpected Error: {exc}", file=sys.stderr)
        sys.exit(3)

    # --- 3. Print summary ------------------------------------------------
    _print_summary(result, course_name)


if __name__ == "__main__":
    main()