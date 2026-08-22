"""
lab_generation.py — Tiered Lab Generation Task (Humanics-Aligned)
=================================================================

Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Defines a CrewAI **Task** that, when executed by the Lab Developer
agent, produces a complete set of tiered coding labs derived from a
course syllabus.  The task specification:

  • Generates labs across **three progressive tiers** — Foundations,
    Application, and Architecture — each building on the previous.
  • Produces **starter/** scaffolding with TODO markers and **solution/**
    reference implementations for every lab.
  • Includes a **README.md** per lab with learning objectives and key
    concepts.
  • Writes output to ``output/labs/<course_name>/<tier_name>/starter/``
    and ``output/labs/<course_name>/<tier_name>/solution/``.
  • Ensures every lab is **self-contained** — a student can clone and
    run it in isolation with a single command.
"""

from __future__ import annotations

from typing import Optional

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Directory-structure requirement — injected into the description so the
# agent produces output organised in the mandated layout.
# ---------------------------------------------------------------------------
_DIRECTORY_STRUCTURE_REQUIREMENT: str = (
    "## 📁  Directory Structure Requirements\n\n"
    "Your output must create the following directory layout under "
    "`output/labs/<course_name>/`.  The placeholders `<course_name>` and "
    "`<tier_name>` must be replaced with the actual values:\n\n"
    "```\n"
    "output/labs/<course_name>/\n"
    "│\n"
    "├── tier1_foundations/\n"
    "│   ├── starter/\n"
    "│   │   ├── README.md          # Learning objectives, key concepts,\n"
    "│   │   │                       # setup instructions, and pre-requisites\n"
    "│   │   └── <lab_files>.py     # Scaffolded code with TODO markers\n"
    "│   └── solution/\n"
    "│       ├── README.md          # Same as starter, plus solution notes\n"
    "│       └── <lab_files>.py     # Fully-working, commented code\n"
    "│\n"
    "├── tier2_application/\n"
    "│   ├── starter/\n"
    "│   │   ├── README.md\n"
    "│   │   └── <multi-file project scaffold>\n"
    "│   └── solution/\n"
    "│       ├── README.md\n"
    "│       └── <multi-file project with full implementation>\n"
    "│\n"
    "└── tier3_architecture/\n"
    "    ├── starter/\n"
    "    │   ├── README.md\n"
    "    │   └── <system-design / microservices scaffold>\n"
    "    └── solution/\n"
    "        ├── README.md\n"
    "        └── <system-design / microservices full implementation>\n"
    "```\n\n"
    "**Key rules:**\n"
    "- Every lab directory (starter and solution) MUST contain a README.md.\n"
    "- Starter files MUST contain `# TODO:` markers — at least 3 per file —\n"
    "  that a student can search for with `grep -r \"TODO\"`.  Each TODO must\n"
    "  be a concrete, actionable instruction (never a placeholder).\n"
    "- Solution files MUST be complete, runnable, and thoroughly commented\n"
    "  so a self-study learner can understand every line.\n"
    "- Labs MUST be self-contained: a student should be able to `cd` into\n"
    "  any starter/ or solution/ directory, follow the README, and run the\n"
    "  code with zero external dependencies beyond what is documented.\n"
    "- File names and directory names MUST use snake_case.\n"
)
# ---------------------------------------------------------------------------
# Tier definitions — injected into the description as non-negotiable scope.
# ---------------------------------------------------------------------------
_TIER_DEFINITIONS: str = (
    "## 🔺  Tier Definitions (Non-Negotiable)\n\n"
    "### Tier 1 — Foundations\n"
    "**Focus:** Syntax drills, basic patterns, and single-file exercises.\n\n"
    "- Each lab must be a single `.py` file (plus README).\n"
    "- Topics: variables, control flow, functions, data structures (lists,\n"
    "  dicts, sets), file I/O, error handling, and basic OOP.\n"
    "- Code complexity: ≤ 150 lines per solution file.\n"
    "- Every lab must include at least one exercise that reads input and\n"
    "  produces output (stdin/stdout or file-based).\n"
    "- Humanics tags: `[T]` on every file, `[D]` on data-handling exercises,\n"
    "  `[H]` in README reflections on code readability and collaboration.\n\n"
    "### Tier 2 — Application\n"
    "**Focus:** Multi-file projects, APIs, and data processing.\n\n"
    "- Each lab must span 3–5 files (plus README) organised as a coherent\n"
    "  mini-project.\n"
    "- Topics: REST API consumption, database CRUD, data pipelines (ETL),\n"
    "  CLI tools, web scraping, testing with pytest, and packaging.\n"
    "- Must include a `requirements.txt` or `pyproject.toml` listing all\n"
    "  dependencies.\n"
    "- Code complexity: ≤ 500 lines total across all solution files.\n"
    "- Humanics tags: `[T]` on tooling/integration, `[D]` on every\n"
    "  data-processing step, `[H]` in README sections on API ethics,\n"
    "  data privacy, and team-worthy code practices.\n\n"
    "### Tier 3 — Architecture\n"
    "**Focus:** System design, microservices, and deployment.\n\n"
    "- Each lab must span 5–10 files representing a service-oriented\n"
    "  architecture.\n"
    "- Topics: message queues (RabbitMQ/Redis), containerisation (Docker),\n"
    "  service orchestration (docker-compose), database sharding/replication,\n"
    "  CI/CD pipelines, cloud deployment (AWS/GCP/Azure basics), and\n"
    "  observability (logging, metrics, health checks).\n"
    "- Must include a `Dockerfile` and `docker-compose.yml` so the entire\n"
    "  system can be started with `docker-compose up`.\n"
    "- Must include a `Makefile` or shell script for common operations.\n"
    "- Code complexity: ≤ 1500 lines total across all solution files.\n"
    "- Humanics tags: `[T]` on every service, `[D]` on monitoring/metrics\n"
    "  dashboards, `[H]` in README sections on system ethics, accessibility,\n"
    "  and incident-response communication.\n"
)
# ---------------------------------------------------------------------------
# README template mandate — injected into the description.
# ---------------------------------------------------------------------------
_README_TEMPLATE_REQUIREMENT: str = (
    "## 📄  README.md Requirements (Per Lab)\n\n"
    "Every starter/ and solution/ directory MUST contain a README.md with "
    "the following exact sections:\n\n"
    "```markdown\n"
    "# {Lab Title}\n\n"
    "## 🎯 Learning Objectives\n"
    "- Bullet list of 3–5 specific, measurable learning outcomes.\n\n"
    "## 🧠 Key Concepts\n"
    "- Bullet list of concepts this lab teaches or reinforces.\n\n"
    "## 📋 Prerequisites\n"
    "- What the student must already know or have installed.\n"
    "- Include exact Python version (≥3.10) and any packages.\n\n"
    "## 🚀 Getting Started\n"
    "```bash\n"
    "# Exact commands to clone, install deps, and run.\n"
    "# Must work when copy-pasted into a fresh terminal.\n"
    "```\n\n"
    "## 🏗️ Project Structure\n"
    "```\n"
    "# Tree view of all files with brief descriptions.\n"
    "```\n\n"
    "## 📝 Lab Instructions\n"
    "# Step-by-step walkthrough of what to build.\n"
    "# In starter/ this is the main guide.\n"
    "# In solution/ add a \"Solution Walkthrough\" sub-section.\n\n"
    "## 🔍 Humanics Reflection  [H]\n"
    "# Prompts that ask the student to reflect on:\n"
    "# - How does this code impact people?\n"
    "# - What ethical considerations apply?\n"
    "# - How would you explain this to a non-technical colleague?\n\n"
    "## 📚 Additional Resources\n"
    "# Links to docs, tutorials, and further reading.\n"
    "```\n"
)

# ---------------------------------------------------------------------------
# Humanics embedding mandate — injected verbatim into the description.
# ---------------------------------------------------------------------------
_HUMANICS_LAB_MANDATE: str = (
    "## 🔴 MANDATORY — Humanics Literacies in Labs\n\n"
    "Every lab you produce must visibly integrate **three inseparable "
    "literacies** grounded in Joseph Aoun's Humanics framework:\n\n"
    "- **Technological Literacy [T]** — Every starter file must require "
    "the student to write, debug, and run code.  Every solution file must "
    "demonstrate idiomatic, well-structured code with meaningful comments.  "
    "Include tooling fluency (git commands, virtual environments, linters).\n"
    "- **Data Literacy [D]** — At least one lab per tier must involve "
    "collecting, processing, analysing, or visualising data.  Include "
    "exercises that require the student to make a data-informed decision "
    "and justify it in a comment or README section.\n"
    "- **Human Literacy [H]** — Every README.md must include a **Humanics "
    "Reflection** section (see README template above) that prompts the "
    "student to consider ethics, accessibility, team dynamics, or societal "
    "impact.  Tier 3 labs must also address incident response and "
    "responsible deployment.\n\n"
    "These are **not** separate modules — they are threads woven into "
    "every lab.  Tag relevant sections with `[T]`, `[D]`, or `[H]` so "
    "instructors can verify coverage at a glance.\n"
)

# ---------------------------------------------------------------------------
# Code-quality mandate — injected into the description.
# ---------------------------------------------------------------------------
_CODE_QUALITY_MANDATE: str = (
    "## 🔴 MANDATORY — Code Quality Standards\n\n"
    "- All Python code must use **type hints** (Python 3.10+ syntax, e.g. "
    "`list[str]`, `dict[str, int]`, `str | None`).\n"
    "- Every `.py` file must start with a module-level docstring explaining "
    "its purpose.\n"
    "- Functions must have docstrings (at least a one-liner for simple "
    "functions, full Google-style for complex ones).\n"
    "- Solution code must pass `ruff check` and `mypy --strict` with zero "
    "errors (unless the lab *teaches* fixing those errors).\n"
    "- TODO markers must follow the format: `# TODO(<tier>): <actionable "
    "instruction>` — e.g. `# TODO(t1): Implement the binary_search "
    "function to return -1 when the target is not found.`\n"
    "- Never leave a TODO that says \"implement this\" without specifying "
    "exactly *what* to implement and *how* to verify correctness.\n"
)
# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_lab_generation_task(
    *,
    course_name: str,
    agent: Agent,
    syllabus_context: str | None = None,
    topic_focus: str | None = None,
    language: str = "Python",
    verbose: bool = False,
) -> Task:
    """Create a CrewAI Task that generates tiered coding labs from a syllabus.

    Parameters
    ----------
    course_name : str
        The name of the vocational course (e.g. "Full-Stack Web Development").
    agent : Agent
        The Lab Developer agent that will execute this task.
    syllabus_context : str or None
        Optional syllabus content to ground lab generation.  When provided the
        agent extracts topics, modules, and learning objectives from this
        context to design relevant labs.  When omitted the agent infers lab
        topics from the course name.
    topic_focus : str or None
        Optional comma-separated list of specific topics to emphasise
        (e.g. "recursion, graph algorithms, REST APIs").
    language : str
        The primary programming language for the labs.  Defaults to
        ``"Python"``.
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ---- Build the description ------------------------------------------
    description_parts: list[str] = [
        f"Generate a comprehensive set of tiered coding labs for the "
        f"following course:\n\n"
        f"**Course Name:** {course_name}\n"
        f"**Primary Language:** {language}\n",
    ]

    if syllabus_context:
        description_parts.append(
            f"\n**Syllabus Context:**\n\n{syllabus_context}\n"
        )

    if topic_focus:
        description_parts.append(
            f"\n**Topic Focus:** {topic_focus}\n"
        )

    description_parts.append(
        "\n\nYour task is to design a complete set of self-contained coding "
        "labs organised into three progressive tiers.  Each tier must have "
        "**at least 3 labs**, and every lab must include both a **starter/** "
        "scaffold and a **solution/** reference implementation.\n\n"
        "_TIER_DEFINITIONS_\n\n"
        "_DIRECTORY_STRUCTURE_REQUIREMENT_\n\n"
        "_README_TEMPLATE_REQUIREMENT_\n\n"
        "_HUMANICS_LAB_MANDATE_\n\n"
        "_CODE_QUALITY_MANDATE_\n"
    )

    # Inject the full mandate text (not references).
    description = (
        "".join(description_parts)
        .replace("_TIER_DEFINITIONS_", _TIER_DEFINITIONS)
        .replace(
            "_DIRECTORY_STRUCTURE_REQUIREMENT_",
            _DIRECTORY_STRUCTURE_REQUIREMENT,
        )
        .replace(
            "_README_TEMPLATE_REQUIREMENT_",
            _README_TEMPLATE_REQUIREMENT,
        )
        .replace("_HUMANICS_LAB_MANDATE_", _HUMANICS_LAB_MANDATE)
        .replace("_CODE_QUALITY_MANDATE_", _CODE_QUALITY_MANDATE)
    )

    # ---- Build the expected_output --------------------------------------
    expected_output = (
        f"A complete, self-contained set of tiered {language} coding labs "
        f"for the course **{course_name}**.  The output must:\n\n"
        "1. Organise all labs under `output/labs/<course_name>/` using the "
        "exact directory structure specified above (tier1_foundations, "
        "tier2_application, tier3_architecture).\n\n"
        "2. **Tier 1 — Foundations** (≥3 labs): Single-file exercises "
        "covering syntax drills, basic patterns, and algorithmic thinking.  "
        "Each lab: one `.py` starter file with ≥5 TODO markers, one `.py` "
        "solution file with full comments, and a README.md in both "
        "directories.  Topics should ladder from simple (variables, loops) "
        "to intermediate (functions, file I/O, error handling).\n\n"
        "3. **Tier 2 — Application** (≥3 labs): Multi-file mini-projects "
        "spanning 3–5 files each.  Must include at least one REST API "
        "consumer, one data-processing pipeline (ETL), and one CLI tool.  "
        "Each lab must ship with a `requirements.txt` and a `README.md` in "
        "both starter/ and solution/.  Starter files must have TODO markers "
        "in every file; solutions must be fully runnable.\n\n"
        "4. **Tier 3 — Architecture** (≥3 labs): Service-oriented systems "
        "with 5–10 files each.  Must include at least one microservices "
        "system with Docker and docker-compose, one CI/CD pipeline "
        "definition, and one observability/monitoring setup.  Each lab "
        "must ship with `Dockerfile`, `docker-compose.yml`, `Makefile`, "
        "and `README.md` in both starter/ and solution/.\n\n"
        "5. **README quality:** Every lab's README must follow the mandated "
        "template (Learning Objectives, Key Concepts, Prerequisites, "
        "Getting Started, Project Structure, Lab Instructions, Humanics "
        "Reflection, Additional Resources).  The Humanics Reflection "
        "section must contain at least 3 thought-provoking questions.\n\n"
        "6. **Code quality:** All solution code must use Python 3.10+ type "
        "hints, module-level docstrings, function docstrings, and pass "
        "`ruff check`.  TODO markers must use the `# TODO(<tier>):` format "
        "with actionable instructions.\n\n"
        "7. **Self-contained:** Every lab must be independently cloneable "
        "and runnable.  A student should need only Python 3.10+, the "
        "documented dependencies, and `docker` (for Tier 3) to complete "
        "every exercise.\n\n"
        f"The first file in the output must be:\n\n"
        f"`output/labs/{course_name.lower().replace(' ', '_')}/README.md`\n"
        f"— a top-level index listing all labs with one-line descriptions.\n"
    )

    # ---- Compute the output file path -----------------------------------
    safe_name: str = (
        course_name.strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace(":", "-")
        .replace("&", "and")
        .lower()
    )
    output_file: str = f"output/labs/{safe_name}/README.md"

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
create_lab_task = create_lab_generation_task

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agents.curriculum_architect import create_curriculum_architect

    # Use the existing architect agent for self-test; in production a
    # dedicated Lab Developer agent would be used.
    agent = create_curriculum_architect(verbose=True)
    task = create_lab_generation_task(
        course_name="Python for Data Engineering",
        agent=agent,
        topic_focus="ETL pipelines, REST APIs, Docker, PostgreSQL",
        language="Python",
        verbose=True,
    )
    print("✅ Lab Generation Task created successfully.\n")
    print(f"   Output file:  {task.output_file}")
    print(f"   Agent role:   {agent.role.split(chr(10))[0]}")
    print(f"   Async:        {task.async_execution}")
    print(f"   Description length: {len(task.description)} chars")
    print(f"   Expected output length: {len(task.expected_output)} chars")