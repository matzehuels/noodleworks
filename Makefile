.PHONY: help hooks clean-hooks run-hooks

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


hooks: ## Install and update pre-commit hooks
	pip install pre-commit nbstripout
	pre-commit install
	pre-commit autoupdate

clean-hooks: ## Uninstall pre-commit hooks
	pre-commit uninstall

run-hooks: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Default target
.DEFAULT_GOAL := help 