# ==============================================================================
# Munasib Khareed - Project Automation Makefile
# ==============================================================================
.PHONY: help setup lint format test run-api run-ui train precompute clean

PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn
STREAMLIT ?= streamlit
PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy

help:
	@echo "Munasib Khareed - Command Targets:"
	@echo "  setup       - Install dependencies in editable mode and download assets"
	@echo "  lint        - Run ruff linter and mypy type checks"
	@echo "  format      - Format code using ruff"
	@echo "  test        - Run automated tests with coverage report"
	@echo "  run-api     - Launch the FastAPI backend server"
	@echo "  run-ui      - Launch the Streamlit frontend dashboard"
	@echo "  train       - Train/fine-tune models and surrogate regressors"
	@echo "  precompute  - Precompute product profiles, aspects, and risk scores into cache"
	@echo "  clean       - Remove build artifacts, caches, and temporary files"

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PYTHON) scripts/download_assets.py

lint:
	$(RUFF) check .
	$(MYPY) src/ tests/

format:
	$(RUFF) format .
	$(RUFF) check --fix .

test:
	$(PYTEST) tests/unit -v --cov=src/munasib --cov-report=term-missing

run-api:
	$(UVICORN) munasib.api.main:app --host 127.0.0.1 --port 8000 --reload

run-ui:
	$(STREAMLIT) run app/Home.py --server.port 8501

train:
	$(PYTHON) scripts/train_models.py

precompute:
	$(PYTHON) scripts/precompute_pipeline.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
