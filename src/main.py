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
    """Entry point — orchestrate the syllabus + labs generation pipeline."""

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

    # --- 2. Run the crew -------------------------------------------------
    try:
        result = run_syllabus_crew(
            course_name,
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

    # --- 3. Print summary ------------------------------------------------
    _print_summary(result, course_name)


if __name__ == "__main__":
    main()