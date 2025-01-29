SHELL := /bin/zsh   # or /bin/bash if you prefer

.PHONY: setup clean test lint format check

# Create venv and install dependencies
setup:
	pip install uv
	uv venv
	source .venv/bin/activate
	uv pip install -e ".[dev]"

# Clean up generated files and caches
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Run tests with pytest
test:
	uv run pytest tests -c pyproject.toml

# Run linting with ruff
lint:
	uv run ruff check . --config pyproject.toml

# Format code with ruff
format:
	uv run ruff format . --config pyproject.toml

# Run type check
check:
	uv run mypy --config-file pyproject.toml
