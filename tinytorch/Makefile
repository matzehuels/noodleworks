.PHONY: test lint clean type-check check

# Config file
CONFIG := pyproject.toml

# Create virtual env
venv:
	python -m venv .venv

# Test commands
test:
	python -m pytest

lint:
	ruff check --config $(CONFIG) .

# Type checking
type-check:
	mypy --config-file $(CONFIG) .

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Run all checks
check: lint type-check test
