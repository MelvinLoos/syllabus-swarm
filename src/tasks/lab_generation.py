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
  • Dynamically adapts file extensions, linters, and tooling references
    based on the ``language`` parameter (e.g. ``.py`` / ``ruff`` for
    Python, ``.js`` / ``eslint`` for JavaScript).
"""

from __future__ import annotations

from typing import Optional

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Language configuration — maps a language name to its tooling defaults.
# ---------------------------------------------------------------------------

_LANGUAGE_CONFIG: dict[str, dict[str, str]] = {
    "Python": {
        "ext": ".py",
        "linter": "ruff check",
        "type_checker": "mypy --strict",
        "dep_file": "requirements.txt",
        "build_file": "pyproject.toml",
        "lang_version": "Python ≥3.10",
        "run_cmd": "python",
    },
    "JavaScript": {
        "ext": ".js",
        "linter": "eslint",
        "type_checker": "tsc --noEmit  # if using JSDoc types",
        "dep_file": "package.json",
        "build_file": "package.json",
        "lang_version": "Node.js ≥18 LTS",
        "run_cmd": "node",
    },
    "TypeScript": {
        "ext": ".ts",
        "linter": "eslint",
        "type_checker": "tsc --noEmit",
        "dep_file": "package.json",
        "build_file": "tsconfig.json",
        "lang_version": "Node.js ≥18 LTS + TypeScript ≥5.0",
        "run_cmd": "npx ts-node",
    },
    "Java": {
        "ext": ".java",
        "linter": "checkstyle",
        "type_checker": "javac -Xlint:all",
        "dep_file": "pom.xml  # or build.gradle",
        "build_file": "pom.xml",
        "lang_version": "Java ≥17 LTS",
        "run_cmd": "java",
    },
    "Go": {
        "ext": ".go",
        "linter": "golangci-lint run",
        "type_checker": "go vet",
        "dep_file": "go.mod",
        "build_file": "go.mod",
        "lang_version": "Go ≥1.21",
        "run_cmd": "go run",
    },
    "Rust": {
        "ext": ".rs",
        "linter": "clippy",
        "type_checker": "cargo check",
        "dep_file": "Cargo.toml",
        "build_file": "Cargo.toml",
        "lang_version": "Rust ≥1.75 (stable)",
        "run_cmd": "cargo run",
    },
    "C#": {
        "ext": ".cs",
        "linter": "dotnet format",
        "type_checker": "dotnet build /warnaserror",
        "dep_file": "*.csproj",
        "build_file": "*.csproj",
        "lang_version": ".NET ≥8.0",
        "run_cmd": "dotnet run",
    },
    "PHP": {
        "ext": ".php",
        "linter": "phpcs",
        "type_checker": "phpstan analyse",
        "dep_file": "composer.json",
        "build_file": "composer.json",
        "lang_version": "PHP ≥8.2",
        "run_cmd": "php",
    },
}


def _get_lang_config(language: str) -> dict[str, str]:
    """Return the tooling config for *language*, falling back to Python."""
    # Normalise: strip whitespace, title-case for matching.
    key = language.strip()
    if key in _LANGUAGE_CONFIG:
        return _LANGUAGE_CONFIG[key]
    # Try case-insensitive match.
    for name, cfg in _LANGUAGE_CONFIG.items():
        if name.lower() == key.lower():
            return cfg
    # Fallback to Python with a warning-like default.
    return _LANGUAGE_CONFIG["Python"]


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
    "│   │   └── <lab_files>{ext}     # Scaffolded code with TODO markers\n"
    "│   └── solution/\n"
    "│       ├── README.md          # Same as starter, plus solution notes\n"
    "│       └── <lab_files>{ext}     # Fully-working, commented code\n"
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
    "- All source files MUST use the `{ext}` extension for {language}.\n"
)
# ---------------------------------------------------------------------------
# Tier definitions — injected into the description as non-negotiable scope.
# ---------------------------------------------------------------------------
_TIER_DEFINITIONS: str = (
    "## 🔺  Tier Definitions (Non-Negotiable)\n\n"
    "### Tier 1 — Foundations\n"
    "**Focus:** Syntax drills, basic patterns, and single-file exercises.\n\n"
    "- Each lab must be a single `{ext}` file (plus README).\n"
    "- Topics: variables, control flow, functions, data structures (lists,\n"
    "  dicts/objects, sets), file I/O, error handling, and basic OOP.\n"
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
    "  CLI tools, web scraping, testing, and packaging.\n"
    "- Must include a `{dep_file}` listing all dependencies.\n"
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
    "# {{Lab Title}}\n\n"
    "## 🎯 Learning Objectives\n"
    "- Bullet list of 3–5 specific, measurable learning outcomes.\n\n"
    "## 🧠 Key Concepts\n"
    "- Bullet list of concepts this lab teaches or reinforces.\n\n"
    "## 📋 Prerequisites\n"
    "- What the student must already know or have installed.\n"
    "- Include exact language version ({lang_version}) and any packages.\n\n"
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
    "- All {language} code must use **type hints** where the language "
    "supports them ({lang_version} syntax).\n"
    "- Every `{ext}` file must start with a module-level docstring/comment "
    "explaining its purpose.\n"
    "- Functions must have docstrings/JSDoc (at least a one-liner for simple "
    "functions, full documentation for complex ones).\n"
    "- Solution code must pass `{linter}` and `{type_checker}` with zero "
    "errors (unless the lab *teaches* fixing those errors).\n"
    "- TODO markers must follow the format: `# TODO(<tier>): <actionable "
    "instruction>` — e.g. `# TODO(t1): Implement the binary_search "
    "function to return -1 when the target is not found.`\n"
    "- Never leave a TODO that says \"implement this\" without specifying "
    "exactly *what* to implement and *how* to verify correctness.\n"
)

# ---------------------------------------------------------------------------
# Tool-usage mandate — forces the agent to use output_export_tool iteratively.
# ---------------------------------------------------------------------------
_TOOL_USAGE_MANDATE: str = (
    "## 🔴 MANDATORY — Use the `output_export_tool` to Write Files\n\n"
    "You have access to the **`output_export_tool`** (specifically the "
    "`write-labs` command).  You MUST use this tool to generate and save "
    "the actual code files for Tier 1, Tier 2, and Tier 3 to disk.\n\n"
    "**Workflow:**\n"
    "1. Generate the labs for **Tier 1** first.  Call `output_export_tool` "
    "with `command=\"write-labs\"`, passing the `course_name`, `tier` "
    "(e.g. `\"tier1_foundations\"`), and a `files` dict mapping relative "
    "paths to file contents.\n"
    "2. After Tier 1 files are written, move to **Tier 2** and repeat.\n"
    "3. After Tier 2, move to **Tier 3** and repeat.\n"
    "4. Once ALL files for ALL tiers are successfully written to disk, "
    "your FINAL textual response to this task should ONLY be the top-level "
    "index Markdown (`README.md`) summarising all the labs with one-line "
    "descriptions.\n\n"
    "**Example tool call:**\n"
    "```json\n"
    "{{\n"
    '  "command": "write-labs",\n'
    '  "course_name": "Javascript OOP Basics",\n'
    '  "tier": "tier1_foundations",\n'
    '  "files": {{\n'
    '    "starter/README.md": "# Lab 1: ...",\n'
    '    "starter/lab1_variables{ext}": "// TODO(t1): ...",\n'
    '    "solution/README.md": "# Lab 1 Solution: ...",\n'
    '    "solution/lab1_variables{ext}": "// Complete solution ..."\n'
    "  }}\n"
    "}}\n"
    "```\n\n"
    "**Do NOT** attempt to write all file contents inline in your final "
    "text response.  Use the tool for every file.  Your text response "
    "should be ONLY the top-level index after all files are saved.\n"
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
        ``"Python"``.  Determines file extensions, linters, type checkers,
        and dependency-file names used throughout the task instructions.
    verbose : bool
        Enable detailed task execution logging.

    Returns
    -------
    Task
        Fully-configured CrewAI Task ready to be assigned to a Crew.
    """
    # ── Resolve language-specific tooling ──────────────────────────────
    cfg = _get_lang_config(language)
    ext = cfg["ext"]
    linter = cfg["linter"]
    type_checker = cfg["type_checker"]
    dep_file = cfg["dep_file"]
    build_file = cfg["build_file"]
    lang_version = cfg["lang_version"]
    run_cmd = cfg["run_cmd"]

    # ── Format the static mandates with language-specific values ────────
    fmt = dict(
        language=language,
        ext=ext,
        linter=linter,
        type_checker=type_checker,
        dep_file=dep_file,
        build_file=build_file,
        lang_version=lang_version,
        run_cmd=run_cmd,
    )

    directory_structure = _DIRECTORY_STRUCTURE_REQUIREMENT.format(**fmt)
    tier_definitions = _TIER_DEFINITIONS.format(**fmt)
    readme_template = _README_TEMPLATE_REQUIREMENT.format(**fmt)
    code_quality = _CODE_QUALITY_MANDATE.format(**fmt)
    tool_usage = _TOOL_USAGE_MANDATE.format(**fmt)

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
        "_TOOL_USAGE_MANDATE_\n\n"
        "_TIER_DEFINITIONS_\n\n"
        "_DIRECTORY_STRUCTURE_REQUIREMENT_\n\n"
        "_README_TEMPLATE_REQUIREMENT_\n\n"
        "_HUMANICS_LAB_MANDATE_\n\n"
        "_CODE_QUALITY_MANDATE_\n"
    )

    # Inject the full mandate text (not references).
    description = (
        "".join(description_parts)
        .replace("_TOOL_USAGE_MANDATE_", tool_usage)
        .replace("_TIER_DEFINITIONS_", tier_definitions)
        .replace(
            "_DIRECTORY_STRUCTURE_REQUIREMENT_",
            directory_structure,
        )
        .replace(
            "_README_TEMPLATE_REQUIREMENT_",
            readme_template,
        )
        .replace("_HUMANICS_LAB_MANDATE_", _HUMANICS_LAB_MANDATE)
        .replace("_CODE_QUALITY_MANDATE_", code_quality)
    )

    # ---- Build the expected_output --------------------------------------
    expected_output = (
        f"## 🔴 CRITICAL: You MUST use the `output_export_tool`\n\n"
        f"You have access to the `output_export_tool` with the `write-labs` "
        f"command.  You MUST use this tool to generate and save the actual "
        f"code files for Tier 1, Tier 2, and Tier 3 to disk.  Do this "
        f"step-by-step — generate one tier at a time, write its files via "
        f"the tool, then move to the next tier.\n\n"
        f"**Once all files are successfully written**, your FINAL textual "
        f"response to this task should ONLY be the top-level index Markdown "
        f"summarizing the labs.\n\n"
        f"---\n\n"
        f"A complete, self-contained set of tiered **{language}** coding "
        f"labs for the course **{course_name}**.  The output must:\n\n"
        f"1. Organise all labs under `output/labs/<course_name>/` using the "
        f"exact directory structure specified above (tier1_foundations, "
        f"tier2_application, tier3_architecture).\n\n"
        f"2. **Tier 1 — Foundations** (≥3 labs): Single-file exercises "
        f"covering syntax drills, basic patterns, and algorithmic thinking.  "
        f"Each lab: one `{ext}` starter file with ≥5 TODO markers, one "
        f"`{ext}` solution file with full comments, and a README.md in both "
        f"directories.  Topics should ladder from simple (variables, loops) "
        f"to intermediate (functions, file I/O, error handling).\n\n"
        f"3. **Tier 2 — Application** (≥3 labs): Multi-file mini-projects "
        f"spanning 3–5 files each.  Must include at least one REST API "
        f"consumer, one data-processing pipeline (ETL), and one CLI tool.  "
        f"Each lab must ship with a `{dep_file}` and a `README.md` in "
        f"both starter/ and solution/.  Starter files must have TODO markers "
        f"in every file; solutions must be fully runnable.\n\n"
        f"4. **Tier 3 — Architecture** (≥3 labs): Service-oriented systems "
        f"with 5–10 files each.  Must include at least one microservices "
        f"system with Docker and docker-compose, one CI/CD pipeline "
        f"definition, and one observability/monitoring setup.  Each lab "
        f"must ship with `Dockerfile`, `docker-compose.yml`, `Makefile`, "
        f"and `README.md` in both starter/ and solution/.\n\n"
        f"5. **README quality:** Every lab's README must follow the mandated "
        f"template (Learning Objectives, Key Concepts, Prerequisites, "
        f"Getting Started, Project Structure, Lab Instructions, Humanics "
        f"Reflection, Additional Resources).  The Humanics Reflection "
        f"section must contain at least 3 thought-provoking questions.\n\n"
        f"6. **Code quality:** All solution code must use {lang_version} "
        f"idioms, module-level documentation, function documentation, and "
        f"pass `{linter}`.  TODO markers must use the "
        f"`# TODO(<tier>):` format with actionable instructions.\n\n"
        f"7. **Self-contained:** Every lab must be independently cloneable "
        f"and runnable.  A student should need only {lang_version}, the "
        f"documented dependencies, and `docker` (for Tier 3) to complete "
        f"every exercise.\n\n"
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