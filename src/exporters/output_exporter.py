"""
output_exporter.py — Output Exporter LLM Wiring
================================================

Issue #4: Automated Output Exporter & Markdown Packager
Issue #5: Per-Agent Model Configuration — Specialized LLMs for Each Agent

The **Output Exporter** agent compiles and packages all generated materials
(syllabi, tiered labs, and rubrics) into clean, portable directory
structures and a consolidated Markdown manifest.

This module centralises the exporter's LLM needs.  Any formatting,
summarisation, or packaging step that requires a language model must obtain
its ``LLM`` instance from here — which delegates to the shared
per-agent factory ``build_llm_for_agent(OUTPUT_EXPORTER)`` — rather than
constructing an LLM by hand.

Public API
----------
* ``build_output_exporter_llm(api_key=None)`` — build the exporter's LLM.
* ``get_output_exporter_llm(api_key=None)`` — alias for the same builder.
"""

from __future__ import annotations

from typing import Optional

from crewai import LLM

from src.llm_factory import OUTPUT_EXPORTER, build_llm_for_agent


def build_output_exporter_llm(
    *,
    api_key: Optional[str] = None,
) -> LLM:
    """Build the Output Exporter's ``crewai.LLM`` via the shared factory.

    All LLM construction for the Output Exporter is delegated to
    :func:`src.llm_factory.build_llm_for_agent`, so model selection,
    temperature, top_p, and max_tokens follow the project-wide 4-tier
    per-agent fallback chain (``AGENT_OUTPUT_EXPORTER_*`` →
    ``AGENT_DEFAULT_*`` → legacy globals → hardcoded defaults).

    Parameters
    ----------
    api_key : str or None
        OpenRouter API key.  When omitted the key is read from the
        ``OPENROUTER_API_KEY`` environment variable.

    Returns
    -------
    LLM
        A fully-configured ``crewai.LLM`` instance wired to OpenRouter for the
        ``OUTPUT_EXPORTER`` role.
    """
    return build_llm_for_agent(OUTPUT_EXPORTER, api_key=api_key)


# Convenience alias — mirrors the naming used by the other agent modules.
get_output_exporter_llm = build_output_exporter_llm


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.llm_factory import get_effective_config

    llm = build_output_exporter_llm()
    config = get_effective_config(OUTPUT_EXPORTER)
    print("✅ Output Exporter LLM built successfully.\n")
    print(f"   Role:      {OUTPUT_EXPORTER}")
    print(f"   Model:     {config['model']}")
    print(f"   Base URL:  {config['base_url']}")
    print(f"   Temp:      {config['temperature']}")
    print(f"   Top-P:     {config['top_p']}")
    print(f"   Max Tokens:{config['max_tokens']}")
    print(f"   llm.model: {llm.model}")