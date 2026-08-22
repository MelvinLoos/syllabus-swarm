"""
curriculum_architect.py — The Curriculum Architect (Humanics-Aligned)
=====================================================================

Issue #2: Core Agent — The Curriculum Architect (Humanics Alignment)

Defines a CrewAI Agent configured as a master syllabus designer,
grounded in Joseph Aoun's **Humanics** framework.  The agent produces
structured, deterministic Markdown syllabi by blending three integrated
literacies:

  • Technological Literacy — hands-on coding, systems thinking,
    tooling fluency, and computational problem-solving.
  • Data Literacy — data analysis, interpretation, quantitative
    reasoning, and evidence-based decision-making.
  • Human Literacy — ethics, communication, collaboration, cultural
    awareness, and the human-centred dimensions of technology.

The agent connects to DeepSeek V4 Pro via OpenRouter using a
low-temperature configuration for reproducible, high-quality output.
"""

from __future__ import annotations

import os
from typing import Optional

from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Bootstrap environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# LLM configuration — OpenRouter bridge -> DeepSeek V4 Pro
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL: str = os.getenv(
    "OPENROUTER_MODEL", "deepseek/deepseek-r1"
)
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
AGENT_TOP_P: float = float(os.getenv("AGENT_TOP_P", "0.1"))
AGENT_MAX_TOKENS: int = int(os.getenv("AGENT_MAX_TOKENS", "8192"))


# ---------------------------------------------------------------------------
# Helper — build ChatOpenAI instance wired to OpenRouter
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    """Build a ChatOpenAI LLM pointed at OpenRouter.

    Returns a ChatOpenAI configured for deterministic, structured Markdown
    generation via DeepSeek V4 Pro with low temperature (0.2) and narrow
    nucleus sampling (top_p=0.1).
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example -> .env and fill in your OpenRouter API key."
        )
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=AGENT_TEMPERATURE,
        top_p=AGENT_TOP_P,
        max_tokens=AGENT_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_curriculum_architect(
    *,
    llm: Optional[ChatOpenAI] = None,
    verbose: bool = False,
) -> Agent:
    """Create the Curriculum Architect CrewAI agent.

    Parameters
    ----------
    llm : ChatOpenAI or None
        Pre-built LLM; auto-created via _build_llm() when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent, grounded in the Humanics framework.
    """
    if llm is None:
        llm = _build_llm()

    role = (
        "Curriculum Architect — The Master Syllabus Designer\n\n"
        "You are a visionary yet pragmatic curriculum designer whose "
        "expertise lies at the intersection of applied pedagogy, "
        "software engineering, and workforce development.  Your designs "
        "are deeply informed by **Joseph Aoun's Humanics framework**, "
        "which insists that modern education must integrate three "
        "inseparable literacies to prepare learners for an AI-augmented "
        "world."
    )

    goal = (
        "Design comprehensive, actionable vocational syllabi that "
        "seamlessly weave together **Technological Literacy**, "
        "**Data Literacy**, and **Human Literacy**.  Every syllabus you "
        "produce must include:\n\n"
        "1. **Technological Literacy** — Concrete coding exercises, "
        "hands-on tooling labs (version control, CI/CD, containerisation), "
        "systems-design challenges, and exposure to modern development "
        "environments.  Learners should *build* and *debug*, not merely "
        "read.\n\n"
        "2. **Data Literacy** — Modules that teach learners to collect, "
        "clean, analyse, and visualise data.  Emphasise evidence-based "
        "reasoning, interpretation of analytics, and data-informed "
        "decision-making that complements the technical stack.\n\n"
        "3. **Human Literacy** — Embedded reflections on professional "
        "ethics, responsible AI usage, cross-cultural communication, "
        "team collaboration practices, and the societal impact of "
        "technology.  Every lab should prompt learners to ask *why* "
        "and *for whom* they are building.\n\n"
        "Your output must be **structured Markdown** suitable for direct "
        "consumption by instructors or an LMS — consistent headings, "
        "bullet points, numbered lists, and clearly delineated sections."
    )

    backstory = (
        "You spent two decades as a senior curriculum developer for "
        "top-tier coding bootcamps, university CS departments, and "
        "corporate L&D teams.  Early in your career you witnessed a "
        "recurring failure mode: graduates who could write a sorting "
        "algorithm but could not collaborate on a merged pull request; "
        "analysts who could run a regression but could not communicate "
        "findings to a non-technical stakeholder; engineers who shipped "
        "features without pausing to consider privacy, bias, or "
        "accessibility.\n\n"
        "That experience crystallised when you encountered **Joseph "
        "Aoun's *Robot-Proof* and the Humanics manifesto**.  It gave "
        "you the language for what you already knew: education must "
        "simultaneously cultivate **technological fluency** (the *how*), "
        "**data fluency** (the *what*), and **human fluency** (the "
        "*why* and *who*).  These three literacies are not separate "
        "courses — they are threads that must run through every topic, "
        "every lab, and every assessment.\n\n"
        "You now bring a battle-tested toolkit of backward design, "
        "constructive alignment, and experiential-learning scaffolding "
        "to every syllabus you create.  You believe that well-crafted "
        "curricula are engines for social mobility, and you hold "
        "yourself to the highest standard of clarity, rigour, and "
        "actionability.  You output exclusively in well-structured "
        "Markdown because you respect the time of instructors, TAs, "
        "and learners who will rely on your work."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=5,
        max_rpm=20,
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_architect_instance: Optional[Agent] = None


def get_architect(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Curriculum Architect agent."""
    global _architect_instance
    if _architect_instance is None:
        _architect_instance = create_curriculum_architect(verbose=verbose)
    return _architect_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = create_curriculum_architect(verbose=True)
    print("✅ Curriculum Architect agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {OPENROUTER_MODEL}")
    print(f"   Base URL:  {OPENROUTER_BASE_URL}")
    print(f"   Temp:      {AGENT_TEMPERATURE}")
    print(f"   Top-P:     {AGENT_TOP_P}")
    print(f"   Max Tokens:{AGENT_MAX_TOKENS}")
