"""
theory_generation.py — Theory Artifact Generation Task
======================================================

Defines a CrewAI **Task** that, when executed by the Theory Instructor
agent, produces interactive theory artifacts derived from a course
syllabus.  The task specification:

  • Reads the generated syllabus and identifies the core concept for
    each of the 3 tiers.
  • Generates exactly ONE theory artifact per tier using the format
    most appropriate for the concept (HTML interactive, terminal script,
    or Mermaid.js Markdown diagram).
  • Writes artifacts into a ``theory/`` subfolder inside each tier's
    lab directory (e.g. ``output/<run_id>/labs/<course>/tier1_foundations/theory/``),
    keeping theory and starter code bundled together for the student.
  • Uses the ``output_export_tool`` with ``command="write-directory-tree"``
    to persist files to disk.
"""

from __future__ import annotations

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Format selection guidance — helps the agent pick the right format per tier.
# ---------------------------------------------------------------------------
_FORMAT_SELECTION_GUIDE: str = (
    "## 🎨  Format Selection Guide\n\n"
    "For each tier, choose the ONE format that best fits the core concept:\n\n"
    "| Tier | Typical Concepts | Recommended Format |\n"
    "|------|-----------------|--------------------|\n"
    "| Tier 1 — Foundations | Syntax, control flow, data structures, basic OOP, "
    "algorithms | **Format A (HTML/JS)** — visualise sorting, searching, "
    "recursion, or state transitions |\n"
    "| Tier 2 — Application | APIs, data pipelines, CLI tools, testing, "
    "databases | **Format B (Terminal Script)** — step-by-step walkthrough "
    "of API calls, ETL flows, or CLI workflows |\n"
    "| Tier 3 — Architecture | Microservices, Docker, CI/CD, system design, "
    "observability | **Format C (Mermaid.js Markdown)** — sequence diagrams, "
    "class diagrams, deployment topologies |\n\n"
    "**Override rule:** If a tier's concept clearly fits a different format "
    "better, use your judgment.  The table above is a guideline, not a "
    "straitjacket.  For example, if Tier 1 covers REST APIs, a terminal "
    "script may be more appropriate than HTML.\n"
)

# ---------------------------------------------------------------------------
# Artifact requirements — what each format must include.
# ---------------------------------------------------------------------------
_ARTIFACT_REQUIREMENTS: str = (
    "## 📐  Artifact Requirements Per Format\n\n"
    "### Format A — Interactive HTML/JS (``.html``)\n"
    "- Single, self-contained file that opens in any browser.\n"
    "- Inline ``<style>`` for all CSS (no external stylesheets).\n"
    "- Inline ``<script>`` for all JavaScript (no external libraries unless "
    "absolutely essential; if needed, use a CDN ``<script src=\"...\">``).\n"
    "- Clear title (``<h1>``) and learning objectives (``<ul>``) at the top.\n"
    "- Interactive controls: buttons, sliders, input fields, or draggable "
    "elements that let the student manipulate the visualisation.\n"
    "- Step-by-step mode where applicable (e.g. 'Next Step' button for "
    "algorithms).\n"
    "- Responsive layout that works on a 1366×768 screen (common laptop).\n"
    "- Comments in the JavaScript explaining key logic.\n\n"
    "### Format B — Pausing Terminal Script (``.sh``)\n"
    "- Single, self-contained shell script (bash).\n"
    "- Starts with ``#!/usr/bin/env bash`` and ``set -euo pipefail``.\n"
    "- Every section prints a clear heading (e.g. ``echo '=== Step 1: ... ==='``).\n"
    "- Pauses between steps with ``read -r -p 'Press Enter to continue...'``.\n"
    "- Explains *what* each command does and *why* before running it.\n"
    "- Handles errors gracefully (explain what went wrong if a command fails).\n"
    "- Includes a summary at the end recapping what was learned.\n"
    "- Must be runnable with ``bash theory.sh`` from the theory/ directory.\n\n"
    "### Format C — Mermaid.js Markdown (``.md``)\n"
    "- Valid Markdown with embedded Mermaid.js code blocks.\n"
    "- Starts with an H1 title and learning objectives.\n"
    "- Each diagram is in a fenced code block: `` ```mermaid ... ``` ``.\n"
    "- Explanatory prose between diagrams — not just a wall of diagrams.\n"
    "- At least 2 distinct diagrams per artifact (e.g. class diagram + "
    "sequence diagram).\n"
    "- Diagrams must use proper Mermaid syntax that renders correctly on "
    "GitHub, VS Code, and Mermaid Live.\n"
    "- Include a 'Key Takeaways' section at the bottom.\n"
)

# ---------------------------------------------------------------------------
# Tool-usage mandate — forces the agent to use output_export_tool.
# ---------------------------------------------------------------------------
_TOOL_USAGE_MANDATE: str = (
    "## 🔴 MANDATORY — Use the `output_export_tool` to Write Files\n\n"
    "You have access to the **`output_export_tool`**.  You MUST use the "
    "`write-directory-tree` command to write your theory artifacts to disk.\n\n"
    "**Workflow:**\n"
    "1. Read the ``run_id`` and ``course_name`` from the context provided "
    "at the start of this task description (look for ``**Run ID:**`` and "
    "``**Course Name:**``).  You MUST include these in EVERY tool call.\n"
    "2. Generate the theory artifact for **Tier 1** first.  Call "
    "`output_export_tool` with `command=\"write-directory-tree\"`, passing "
    "`base_path` and `files` (a dict mapping relative paths to file contents).\n"
    "3. After Tier 1 files are written, move to **Tier 2** and repeat.\n"
    "4. After Tier 2, move to **Tier 3** and repeat.\n"
    "5. Once ALL files for ALL tiers are successfully written to disk, "
    "produce your final textual response: a Markdown summary listing each "
    "tier, the format chosen, the filename, and a one-line description of "
    "the artifact.\n\n"
    "**Base path for each tier:**\n"
    "```\n"
    "output/<run_id>/labs/<course_name>/<tier>/theory/\n"
    "```\n\n"
    "**Example tool call for Tier 1 (you MUST follow this exact structure):**\n"
    "```json\n"
    "{{\n"
    '  "command": "write-directory-tree",\n'
    '  "base_path": "output/2026-08-23_120000_Javascript_OOP/labs/Javascript_OOP/tier1_foundations/theory",\n'
    '  "files": {{\n'
    '    "sorting_visualizer.html": "<!DOCTYPE html>\\n<html lang=\\"en\\">\\n..."\n'
    "  }}\n"
    "}}\n"
    "```\n\n"
    "**Do NOT** attempt to write all file contents inline in your final "
    "text response.  Use the tool for every file.  Your final text response "
    "must be a real summary of what was generated — NOT a placeholder.\n"
)

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_theory_task(
    *,
    agent: Agent,
    course_name: str,
    syllabus_context: str | None = None,
    run_id: str | None = None,
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that generates interactive theory artifacts.

    Parameters
    ----------
    agent : Agent
        The Theory Instructor agent that will execute this task.
    course_name : str
        The name of the vocational course (e.g. "Javascript OOP").
    syllabus_context : str or None
        The generated syllabus content.  The agent extracts topics, modules,
        and learning objectives from this context to design relevant theory
        artifacts.  When omitted the agent infers topics from the course name.
    run_id : str or None
        The unique run identifier (e.g. ``"2026-08-23_120000_Course_Name"``).
        When provided, it is injected into the task description so the agent
        can construct the correct ``base_path`` for ``write-directory-tree``
        tool calls.
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ---- Build the description ------------------------------------------
    description_parts: list[str] = [
        f"Generate interactive theory artifacts for the following course:\n\n"
        f"**Course Name:** {course_name}\n",
    ]

    if run_id:
        description_parts.append(
            f"\n**Run ID:** {run_id}\n"
            f"(MUST use this ``run_id`` to construct the ``base_path`` for "
            f"every `write-directory-tree` tool call)\n"
        )

    if syllabus_context:
        # Truncate syllabus to ~3000 chars to keep prompt manageable.
        ctx = syllabus_context
        if len(ctx) > 4000:
            ctx = ctx[:4000] + "\n\n[... syllabus truncated for length ...]\n"
        description_parts.append(
            f"\n**Syllabus Context:**\n\n{ctx}\n"
        )

    description_parts.append(
        f"\n\nYour task is to read the syllabus above, identify the core "
        f"concept for each of the 3 tiers, and generate exactly ONE theory "
        f"artifact per tier using the format most appropriate for the "
        f"concept.\n\n"
        f"{_FORMAT_SELECTION_GUIDE}\n\n"
        f"{_ARTIFACT_REQUIREMENTS}\n\n"
        f"{_TOOL_USAGE_MANDATE}\n"
    )

    description = "".join(description_parts)

    # ---- Build the expected_output --------------------------------------
    expected_output = (
        "## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
        "Use the `output_export_tool` with `command=\"write-directory-tree\"` "
        "to write ALL theory artifacts to disk.  Write exactly ONE artifact "
        "per tier into the `theory/` subfolder of each tier's lab directory:\n\n"
        "- `output/<run_id>/labs/<course_name>/tier1_foundations/theory/`\n"
        "- `output/<run_id>/labs/<course_name>/tier2_application/theory/`\n"
        "- `output/<run_id>/labs/<course_name>/tier3_architecture/theory/`\n\n"
        "**Once all files are written**, produce a Markdown summary listing "
        "each tier, the format chosen (A/B/C), the filename, and a one-line "
        "description of the artifact.\n"
    )

    # ---- Compute the output file path -----------------------------------
    output_file: str = "output/theory/README.md"

    # ---- Assemble and return the Task -----------------------------------
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_file=output_file,
        async_execution=False,
    )


# ---------------------------------------------------------------------------
# Convenience alias — mirrors the naming convention expected by crews.
# ---------------------------------------------------------------------------
create_theory_generation_task = create_theory_task

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.theory_instructor import create_theory_instructor

    agent = create_theory_instructor(verbose=True)
    task = create_theory_task(
        course_name="Javascript OOP",
        agent=agent,
        run_id="2026-08-23_120000_Javascript_OOP",
        verbose=True,
    )
    print("✅ Theory Generation Task created successfully.\n")
    print(f"   Output file:  {task.output_file}")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")
