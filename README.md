# syllabus-swarm 🐝

> A local multi-agent workspace for generating vocational-level software development curriculum.

**syllabus-swarm** uses AI agents (powered by [CrewAI](https://www.crewai.com/) / [LangChain](https://www.langchain.com/)) connected to **DeepSeek V4 Pro** via **OpenRouter** to collaboratively design, develop, and package complete course materials — syllabi, tiered coding labs, and evaluation rubrics.

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

| Agent | Role | Output |
|-------|------|--------|
| **Curriculum Architect** | Designs syllabi using the Humanics framework (data literacy, technological literacy, human literacy) + experiential learning | `output/syllabus/` |
| **Lab & Project Developer** | Generates tiered hands-on coding exercises with starter code and fully-commented solution keys | `output/labs/` |
| **Output Exporter** | Compiles and packages all materials into clean directory structures | `output/` |

---

## License

MIT