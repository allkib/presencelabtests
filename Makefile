PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup run test lint format clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -U pip

setup: $(BIN)/python  ## Create venv and install package + dev deps
	$(BIN)/pip install -e ".[dev]"

run:  ## Start the web demo (http://127.0.0.1:8000)
	$(BIN)/posedemo $(ARGS)

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
