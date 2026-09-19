.DEFAULT_GOAL := help

.PHONY: help setup lock run test check

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [ARGS=\"...\"]\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create the virtual environment and install dependencies
	uv sync

lock: ## Update the uv lockfile
	uv lock

run: ## Generate a world (ARGS="--width N --height N --tree-density F --seed N --agents N")
	uv run python main.py $(ARGS)

test: ## Run the test suite
	uv run python -m unittest discover -s tests -v

check: test ## Run tests and compile-check the Python source
	uv run python -m compileall -q jevu main.py tests
