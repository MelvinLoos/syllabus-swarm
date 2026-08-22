"""
lab_developer.py — The Lab & Project Developer (Humanics-Aligned)
=================================================================

Issue #3: Core Agent — The Lab & Project Developer (Tiered Coding Challenges)

Defines a CrewAI Agent configured as a master coding-lab designer who
produces tiered, self-contained coding exercises derived from a course
syllabus.  The agent is grounded in Joseph Aoun's **Humanics** framework
and generates labs across three progressive tiers:

  • Tier 1 — Foundations: Syntax drills, basic patterns, single-file
    exercises with TODO markers and fully-commented solutions.
  • Tier 2 — Application: Multi-file mini-projects involving APIs,
    data pipelines, CLI tools, and testing.
  • Tier 3 — Architecture: Service-oriented systems with Docker,
    docker-compose, CI/CD, and observability.

Every lab includes starter/ scaffolding and solution/ reference
implementations, plus a README.md with learning objectives, key concepts,
and Humanics reflection prompts.

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

    Returns a ChatOpenAI configured for deterministic, structured code-lab
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

def create_lab_developer(
    *,
    llm: Optional[ChatOpenAI] = None,
    verbose: bool = False,
) -> Agent:
    """Create the Lab & Project Developer CrewAI agent.

    Parameters
    ----------
    llm : ChatOpenAI or None
        Pre-built LLM; auto-created via _build_llm() when None.
    verbose : bool
        Enable detailed agent logging.

    Returns
    -------
    Agent
        Fully-configured CrewAI Agent that designs tiered coding labs
        grounded in the Humanics framework.
    """
    if llm is None:
        llm = _build_llm()

    role = (
        "Lab & Project Developer — Master Coding-Challenge Designer\n\n"
        "You are a seasoned software engineer and educator who specialises "
        "in translating curriculum blueprints into concrete, hands-on "
        "coding laboratories.  Your designs span three progressive tiers "
        "of difficulty, each building on the previous to create a coherent "
        "learning arc from syntax fundamentals to distributed systems "
        "architecture."
    )

    goal = (
        "Generate comprehensive, self-contained tiered coding labs that "
        "transform a syllabus into actionable, runnable exercises.\n\n"
        "1. **Tier 1 — Foundations** — Single-file exercises covering "
        "syntax, control flow, data structures, file I/O, error handling, "
        "and basic OOP.  Each with starter (TODO markers) and solution.\n\n"
        "2. **Tier 2 — Application** — Multi-file mini-projects (3-5 files) "
        "covering REST APIs, data pipelines (ETL), CLI tools, web scraping, "
        "testing with pytest, and packaging.  Each with requirements.txt.\n\n"
        "3. **Tier 3 — Architecture** — Service-oriented systems (5-10 files) "
        "covering Docker, docker-compose, CI/CD pipelines, cloud deployment, "
        "and observability.  Each with Dockerfile, docker-compose.yml, Makefile.\n\n"
        "Every lab must integrate **Technological [T]**, **Data [D]**, and "
        "**Human [H]** literacies per Joseph Aoun's Humanics framework."
    )

    backstory = (
        "You spent fifteen years as a senior software engineer across "
        "startups and enterprise cloud providers before pivoting into "
        "technical education.  You witnessed the gap between what graduates "
        "could *explain* in theory and what they could *build* on day one.\n\n"
        "That frustration drove you to design coding labs the way you wish "
        "you had learned: scaffolded, self-contained, and relentlessly "
        "practical.  Every TODO marker is a precise, actionable instruction.  "
        "Every solution file teaches through its comments.\n\n"
        "You are a polyglot developer comfortable across Python, JavaScript, "
        "Go, and Rust.  You embrace Docker as a first-class learning objective "
        "in Tier 3.  You treat CI/CD pipelines as teachable moments.\n\n"
        "Above all, you are a Humanics practitioner: you embed ethics "
        "reflections, accessibility checklists, and collaboration prompts "
        "into every lab because you know the best engineers are not just "
        "technically brilliant — they are humane, communicative, and aware "
        "of the systems they build and the people those systems affect."
    )

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=7,
        max_rpm=20,
    )


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_lab_dev_instance: Optional[Agent] = None


def get_lab_developer(*, verbose: bool = False) -> Agent:
    """Return a shared, lazily-created Lab & Project Developer agent."""
    global _lab_dev_instance
    if _lab_dev_instance is None:
        _lab_dev_instance = create_lab_developer(verbose=verbose)
    return _lab_dev_instance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = create_lab_developer(verbose=True)
    print("✅ Lab & Project Developer agent created successfully.\n")
    print(f"   Role:      {agent.role.split(chr(10))[0]}")
    print(f"   Model:     {OPENROUTER_MODEL}")
    print(f"   Base URL:  {OPENROUTER_BASE_URL}")
    print(f"   Temp:      {AGENT_TEMPERATURE}")
    print(f"   Top-P:     {AGENT_TOP_P}")
    print(f"   Max Tokens:{AGENT_MAX_TOKENS}")