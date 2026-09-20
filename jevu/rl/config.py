"""Typed, strict configuration for reinforcement-learning runs."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "survival-claim-v1"
    seed: int = 0
    total_timesteps: int = 500_000
    output_dir: str = "artifacts/rl"
    console_output: str = "progress"


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    width: int = 10
    height: int = 10
    fruit_tree_density: float = 0.15
    max_turns: int = 250
    agent_count: int = 1


@dataclass(frozen=True, slots=True)
class RewardConfig:
    survival_per_turn: float = 1.0
    claim_tile: float = 2.0
    death: float = -25.0
    ineffective_action: float = -0.05


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    name: str = "maskable_ppo"
    device: str = "auto"
    parallel_envs: int = 1
    learning_rate: float = 3e-4
    n_steps: int = 2_048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    every_timesteps: int = 50_000
    keep_last: int = 3


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run: RunConfig
    environment: EnvironmentConfig
    reward: RewardConfig
    algorithm: AlgorithmConfig
    checkpoint: CheckpointConfig

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.run.name):
            raise ValueError(
                "run.name may contain only letters, numbers, '.', '_', '-'"
            )
        if self.run.total_timesteps <= 0:
            raise ValueError("run.total_timesteps must be positive")
        if not self.run.output_dir.strip():
            raise ValueError("run.output_dir cannot be empty")
        if self.run.console_output not in {"progress", "table", "quiet"}:
            raise ValueError("run.console_output must be progress, table, or quiet")
        if self.environment.width <= 0 or self.environment.height <= 0:
            raise ValueError("environment width and height must be positive")
        if not 0.0 <= self.environment.fruit_tree_density <= 1.0:
            raise ValueError("environment.fruit_tree_density must be between 0 and 1")
        if self.environment.max_turns <= 0:
            raise ValueError("environment.max_turns must be positive")
        if self.environment.agent_count <= 0:
            raise ValueError("environment.agent_count must be positive")
        if self.environment.agent_count > (
            self.environment.width * self.environment.height
        ):
            raise ValueError("environment.agent_count cannot exceed the tile count")
        if self.algorithm.name != "maskable_ppo":
            raise ValueError("algorithm.name must be 'maskable_ppo'")
        if self.algorithm.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("algorithm.device must be auto, cpu, mps, or cuda")
        if self.algorithm.parallel_envs <= 0:
            raise ValueError("algorithm.parallel_envs must be positive")
        if self.algorithm.learning_rate <= 0:
            raise ValueError("algorithm.learning_rate must be positive")
        if self.algorithm.n_steps <= 0 or self.algorithm.batch_size <= 0:
            raise ValueError("algorithm n_steps and batch_size must be positive")
        rollout_size = self.algorithm.n_steps * self.algorithm.parallel_envs
        if rollout_size % self.algorithm.batch_size:
            raise ValueError(
                "algorithm.n_steps * parallel_envs must be divisible by batch_size"
            )
        if not 0.0 <= self.algorithm.gamma <= 1.0:
            raise ValueError("algorithm.gamma must be between 0 and 1")
        if not 0.0 <= self.algorithm.gae_lambda <= 1.0:
            raise ValueError("algorithm.gae_lambda must be between 0 and 1")
        if self.algorithm.ent_coef < 0:
            raise ValueError("algorithm.ent_coef cannot be negative")
        if self.checkpoint.every_timesteps <= 0:
            raise ValueError("checkpoint.every_timesteps must be positive")
        if self.checkpoint.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def smoke_config(self) -> TrainingConfig:
        """Return a fast configuration that still executes one PPO rollout."""

        n_steps = min(
            self.algorithm.n_steps,
            max(1, 64 // self.algorithm.parallel_envs),
        )
        rollout_size = n_steps * self.algorithm.parallel_envs
        divisors = [
            value
            for value in range(1, min(self.algorithm.batch_size, rollout_size) + 1)
            if rollout_size % value == 0
        ]
        return replace(
            self,
            run=replace(self.run, total_timesteps=rollout_size),
            algorithm=replace(
                self.algorithm,
                n_steps=n_steps,
                batch_size=max(divisors),
            ),
            checkpoint=replace(self.checkpoint, every_timesteps=rollout_size),
        )

    def with_console_output(self, console_output: str) -> TrainingConfig:
        """Return a validated copy with a different console display mode."""

        return replace(self, run=replace(self.run, console_output=console_output))


def _section[ConfigSection](
    raw: dict[str, Any],
    name: str,
    section_type: type[ConfigSection],
) -> ConfigSection:
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    known = {field.name for field in fields(section_type)}
    unknown = set(values) - known
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown [{name}] field(s): {names}")
    defaults = section_type()
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        default = getattr(defaults, key)
        expected_type = type(default)
        if (
            expected_type is float
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            normalized[key] = float(value)
        elif type(value) is expected_type:
            normalized[key] = value
        else:
            raise ValueError(
                f"[{name}].{key} must be {expected_type.__name__}, "
                f"not {type(value).__name__}"
            )
    try:
        return section_type(**normalized)
    except TypeError as exc:
        raise ValueError(f"Invalid [{name}] configuration: {exc}") from exc


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration from TOML."""

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    expected = {"run", "environment", "reward", "algorithm", "checkpoint"}
    unknown = set(raw) - expected
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration section(s): {names}")
    return TrainingConfig(
        run=_section(raw, "run", RunConfig),
        environment=_section(raw, "environment", EnvironmentConfig),
        reward=_section(raw, "reward", RewardConfig),
        algorithm=_section(raw, "algorithm", AlgorithmConfig),
        checkpoint=_section(raw, "checkpoint", CheckpointConfig),
    )


def config_to_toml(config: TrainingConfig) -> str:
    """Serialize a resolved configuration without an additional dependency."""

    lines: list[str] = []
    for section, values in config.to_dict().items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            rendered = (
                json.dumps(value) if isinstance(value, str) else str(value).lower()
            )
            lines.append(f"{key} = {rendered}")
        lines.append("")
    return "\n".join(lines)
