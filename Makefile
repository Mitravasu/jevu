.DEFAULT_GOAL := help

.PHONY: help setup setup-rl lock run run-rl eval-rl view agent-actions train-rl test check

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [ARGS=\"...\"]\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create the virtual environment and install dependencies
	uv sync

setup-rl: ## Install the project with reinforcement-learning dependencies
	uv sync --extra rl

lock: ## Update the uv lockfile
	uv lock

run: ## Run a game (ARGS="--width N --height N --tree-density F --seed N --agents N --max-turns N")
	uv run python main.py $(ARGS)

run-rl: ## Run a saved RL model (ARGS="--bundle artifacts/rl/.../RUN [game options]")
	uv run --extra rl python main.py --agent-backend rl $(ARGS)

eval-rl: ## Score a saved RL model across fixed seeds and sizes up to 50
	uv run --extra rl python eval_rl.py $(ARGS)

view: ## Animate a game (ARGS="same as run, plus --turns-per-second F")
	uv run python main.py --pygame $(ARGS)

agent-actions: ## Print one agent's actions (ARGS="logs/SIMULATION.jsonl A1")
	uv run python extract_agent_actions.py $(ARGS)

train-rl: ## Train an RL model (ARGS="--config configs/rl/survival_claim.toml")
	uv run --extra rl python train_rl.py $(ARGS)

test: ## Run the test suite
	uv run --extra rl python -m unittest discover -s tests -v

check: test ## Run tests and compile-check the Python source
	uv run --extra rl python -m compileall -q jevu main.py train_rl.py eval_rl.py tests
