# syllabus-swarm 🐝

> A local multi-agent workspace for generating vocational-level software development curriculum.

**syllabus-swarm** uses AI agents (powered by [CrewAI](https://www.crewai.com/) / [LangChain](https://www.langchain.com/)) connected to specialized language models via **OpenRouter** to collaboratively design, develop, and package complete course materials — syllabi, tiered coding labs, and evaluation rubrics.

---

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:MelvinLoos/syllabus-swarm.git
cd syllabus-swarm

# 2. Set up the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API keys
cp .env.example .env
# Edit .env and paste your OpenRouter API key

# 5. Verify the connection
python test_openrouter.py
```

---

## Architecture

syllabus-swarm is built on three specialized AI agents, each assigned a model optimized for its specific role:

| Agent | Role | Default Model | Rationale |
|---|---|---|---|
| **Curriculum Architect** | Designs syllabi using the Humanics framework (data literacy, technological literacy, human literacy) + experiential learning | `deepseek/deepseek-v4-pro` | State-of-the-art multi-step reasoning for crafting logically coherent, pedagogically sound syllabi that span weeks of content across three integrated literacies. |
| **Lab & Project Developer** | Generates tiered hands-on coding exercises with starter code and fully-commented solution keys | `qwen/qwen3-coder` | Purpose-built for programming tasks — produces cleaner starter code, more idiomatic solutions, and fewer hallucinated API calls than general-purpose models. |
| **Output Exporter** | Compiles and packages all materials into clean directory structures and a consolidated manifest | `deepseek/deepseek-v4-flash-latest` | Low-latency, low-cost completions ideal for manifest generation, file assembly, and Markdown packaging — reliability without burning reasoning-token budgets. |

All models are served via **OpenRouter** (`https://openrouter.ai/api/v1`).

### Output Structure

| Agent | Output |
|---|---|
| Curriculum Architect | `output/syllabus/` |
| Lab & Project Developer | `output/labs/` |
| Output Exporter | `output/README.md` (manifest) |

---

## Model Configuration

Every agent obtains its LLM through a shared factory (`src/llm_factory.py`) that resolves model, temperature, top_p, and max_tokens through a **4-tier fallback chain**.

### Environment Variable Convention

```
AGENT_{ROLE}_{PROPERTY}
```

- **`{ROLE}`** — uppercase snake_case agent identifier: `CURRICULUM_ARCHITECT`, `LAB_DEVELOPER`, `OUTPUT_EXPORTER`
- **`{PROPERTY}`** — `MODEL`, `TEMPERATURE`, `MAX_TOKENS`, or `TOP_P`

Supported properties and their defaults:

| Property | Default | Description |
|---|---|---|
| `MODEL` | `deepseek/deepseek-v4-pro` (fallback) | OpenRouter model identifier |
| `TEMPERATURE` | `0.2` | Generation randomness (0.0 = deterministic, 1.0 = creative) |
| `MAX_TOKENS` | `8192` | Maximum completion tokens per agent call |
| `TOP_P` | `0.1` | Nucleus sampling threshold |

### Fallback Chain (highest to lowest priority)

```
1. AGENT_{ROLE}_{PROPERTY}    ← Per-agent override (most specific)
          │
2. AGENT_DEFAULT_{PROPERTY}   ← Catch-all default for all agents
          │
3. Legacy globals             ← OPENROUTER_MODEL / AGENT_TEMPERATURE /
          │                     AGENT_MAX_TOKENS (deprecated, kept for
          │                     backward compatibility)
          │
4. Hardcoded defaults         ← Values in src/llm_factory.py
```

### Backward-Compatible Fallback

The hardcoded catch-all model is `deepseek/deepseek-v4-pro`. Any agent that lacks both a per-agent override and an `AGENT_DEFAULT_MODEL` will use this model. This ensures existing `.env` files without per-agent configuration continue to work.

### Per-Agent Model Defaults (v2)

These defaults are set in `.env.example` and take effect when you copy it to `.env`:

```bash
# Curriculum Architect — deep reasoning for syllabus design
AGENT_CURRICULUM_ARCHITECT_MODEL=deepseek/deepseek-v4-pro

# Lab & Project Developer — code generation specialist
AGENT_LAB_DEVELOPER_MODEL=qwen/qwen3-coder

# Output Exporter — fast, deterministic packaging
AGENT_OUTPUT_EXPORTER_MODEL=deepseek/deepseek-v4-flash-latest
```

To override any agent's model, set the corresponding environment variable before running:

```bash
AGENT_CURRICULUM_ARCHITECT_MODEL=anthropic/claude-sonnet-4 python src/main.py "Python Basics"
```

> See [OpenRouter Models](https://openrouter.ai/models) for a complete list of available model identifiers.

### Verifying Configuration

To see exactly which model each agent is using (with full environment variable resolution):

```bash
python -m src.llm_factory
```

This prints the resolved model, temperature, top_p, and max_tokens for every known agent.

---

## Usage

```bash
# Full pipeline: syllabus + labs + manifest
python src/main.py "Data Science with Python"

# Syllabus only (skip lab generation)
python src/main.py "Full-Stack Web Development" --skip-labs

# Interactive prompt
python src/main.py
```

---

## License

MIT