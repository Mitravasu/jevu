"""Train and save a JevU reinforcement-learning policy."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from jevu.rl.config import load_training_config
from jevu.rl.trainer import train


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to training TOML")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one short PPO rollout to verify the complete pipeline",
    )
    parser.add_argument(
        "--console-output",
        choices=("progress", "table", "quiet"),
        help="Override the config's console display mode",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    if args.smoke:
        config = config.smoke_config()
    if args.console_output is not None:
        config = config.with_console_output(args.console_output)
    if config.run.console_output == "table":
        print(f"Resolved training configuration: {config.to_dict()}")
    elif config.run.console_output == "progress":
        print(f"Training {config.run.name}: {config.run.total_timesteps:,} timesteps")
    run_directory = train(config)
    print(f"Saved RL model bundle: {run_directory}")


if __name__ == "__main__":
    main()
