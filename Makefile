# ============================================================
# syllabus-swarm — Development Makefile
# ============================================================
# Canonical entrypoints for setting up and running the project.
# The project requires Python 3.12+; run `make setup` to bootstrap
# a fresh environment safely.

.PHONY: help check-python setup install verify clean

## help: list available targets
help:
	@echo "Available targets:"
	@echo "  check-python   Verify Python 3.12+ is available"
	@echo "  setup          Create .venv and install dependencies"
	@echo "  install        Install/upgrade dependencies in the active environment"
	@echo "  verify         Smoke-test the LLM factory connection"
	@echo "  clean          Remove the virtual environment and caches"

## check-python: fail fast if Python 3.12+ is not available
check-python:
	@python3 -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info < (4,) else 1)' \
		|| (echo "ERROR: Python 3.12+ is required (found: $$(python3 --version))."; \
		    echo "       Install it via:  brew install python@3.12"; \
		    echo "       or:             brew install pyenv && pyenv install 3.12"; \
		    exit 1)
	@echo "Python OK: $$(python3 --version)"

## setup: create a fresh virtual environment and install dependencies
setup: check-python
	@echo "Creating virtual environment with Python 3.12+..."
	python3 -m venv .venv
	@echo "Installing dependencies..."
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate the environment with:"
	@echo "  source .venv/bin/activate"
	@echo "Then configure your API key:"
	@echo "  cp .env.example .env"

## install: install/upgrade dependencies in the current environment
install:
	pip install --upgrade -r requirements.txt

## verify: smoke-test the LLM factory connection
verify:
	python -m src.llm_factory

## clean: remove virtual environment and caches
clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
