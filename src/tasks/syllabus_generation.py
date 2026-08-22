"""
syllabus_generation.py — Syllabus Generation Task
==================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)

Defines the CrewAI Task that instructs the Curriculum Architect agent
to produce a comprehensive, Humanics-aligned vocational syllabus in
structured Markdown format.
"""

from __future__ import annotations

from crewai import Agent, Task

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------


def create_syllabus_task(
    agent: Agent,
    course_name: str,
) -> Task:
    """Create a syllabus-generation Task bound to the given agent.

    Parameters
    ----------
    agent : Agent
        A pre-built Curriculum Architect agent (or any CrewAI Agent).
    course_name : str
        The course title / topic for which to generate a syllabus.

    Returns
    -------
    Task
        A CrewAI Task configured for syllabus generation.
    """
    description = (
        f'Generate a comprehensive vocational syllabus for the course: '
        f'**"{course_name}"**.\n\n'
        f'Your syllabus must be a complete, structured Markdown document '
        f'that integrates Joseph Aoun\'s **Humanics** framework by weaving '
        f'together three inseparable literacies throughout every module:\n\n'
        f'1. **Technological Literacy** — Hands-on coding exercises, '
        f'tooling labs (version control, CI/CD, containerisation), '
        f'systems-design challenges, and exposure to modern development '
        f'environments.  Learners must *build* and *debug*, not just read.\n\n'
        f'2. **Data Literacy** — Modules that teach learners to collect, '
        f'clean, analyse, and visualise data.  Emphasise evidence-based '
        f'reasoning, interpretation of analytics, and data-informed '
        f'decision-making that complements the technical stack.\n\n'
        f'3. **Human Literacy** — Embedded reflections on professional '
        f'ethics, responsible AI usage, cross-cultural communication, '
        f'team collaboration practices, and the societal impact of '
        f'technology.  Every lab should prompt learners to ask *why* '
        f'and *for whom* they are building.\n\n'
        f'**Required sections** (use H2 `##` headings for each):\n\n'
        f'- **## Course Overview** — title, catalogue description, '
        f'target audience, prerequisites, and 5–7 measurable learning '
        f'objectives.\n\n'
        f'- **## Technological Literacy Modules** — concrete coding labs, '
        f'tooling workshops, and systems-design exercises.\n\n'
        f'- **## Data Literacy Modules** — data collection, cleaning, '
        f'analysis, visualisation, and quantitative reasoning exercises.\n\n'
        f'- **## Human Literacy Modules** — ethics discussions, '
        f'collaboration workshops, communication exercises, cultural '
        f'awareness prompts, and societal-impact case studies.\n\n'
        f'- **## Assessment Strategy** — formative and summative '
        f'assessments, grading rubrics, and a capstone project description.\n\n'
        f'- **## Weekly Schedule / Timeline** — a week-by-week or '
        f'module-by-module breakdown with estimated contact hours, '
        f'labs, and self-study time.\n\n'
        f'- **## Resources & References** — recommended textbooks, '
        f'tools, APIs, datasets, and further reading.\n\n'
        f'**Formatting rules:**\n'
        f'- Use proper Markdown throughout (headings, bullet lists, '
        f'numbered lists, tables where appropriate, bold/italic for emphasis).\n'
        f'- Each Humanics literacy MUST appear as a labelled thread '
        f'in every module (e.g., `🏷️ Tech Literacy`, '
        f'`🏷️ Data Literacy`, `🏷️ Human Literacy`).\n'
        f'- Be concrete — name real tools, languages, frameworks, '
        f'datasets, and APIs; avoid vague abstractions.\n'
        f'- Be actionable — every module must include at least one '
        f'hands-on exercise.\n'
        f'- Target a 12-week vocational programme (~300 total hours).'
    )

    expected_output = (
        f'A complete, well-structured Markdown syllabus document for '
        f'"{course_name}" suitable for direct import into an LMS or '
        f'handout to instructors and learners.  The document must contain '
        f'all six required sections, use consistent `##` H2 headings, '
        f'and include Humanics literacy labels in every module.'
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )