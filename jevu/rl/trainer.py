"""Configuration-driven MaskablePPO training and bundle creation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from jevu.game_state import GameState
from jevu.rl.action_codec import action_mapping
from jevu.rl.agent_aware_ppo import (
    AgentAwareMaskableDictRolloutBuffer,
    AgentAwareMaskablePPO,
)
from jevu.rl.bundle import (
    BUNDLE_VERSION,
    METADATA_FILENAME,
    MODEL_FILENAME,
    game_rules_metadata,
    write_json,
)
from jevu.rl.config import TrainingConfig, config_to_toml
from jevu.rl.device import resolve_device
from jevu.rl.environment import JevUGymEnv
from jevu.rl.observation import ObservationSpec
from jevu.rl.selector import RLActionSelector
from jevu.world import WorldConfig


class ArtifactCallback(BaseCallback):
    """Write episode metrics and bounded recovery checkpoints."""

    def __init__(
        self, run_directory: Path, every_timesteps: int, keep_last: int
    ) -> None:
        super().__init__()
        self.run_directory = run_directory
        self.every_timesteps = every_timesteps
        self.keep_last = keep_last
        self.next_checkpoint = every_timesteps
        self.metrics_path = run_directory / "training_metrics.jsonl"
        self.checkpoint_directory = run_directory / "checkpoints"

    def _on_training_start(self) -> None:
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path.touch(exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            summary = info.get("episode_summary")
            if summary is not None:
                record = {"timesteps": self.num_timesteps, **summary}
                with self.metrics_path.open("a") as metrics_file:
                    metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint = self.checkpoint_directory / f"model-{self.num_timesteps}"
            self.model.save(checkpoint)
            checkpoints = sorted(
                self.checkpoint_directory.glob("model-*.zip"),
                key=lambda path: path.stat().st_mtime,
            )
            for old_checkpoint in checkpoints[: -self.keep_last]:
                old_checkpoint.unlink()
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.every_timesteps
        return True


class LossProgressBarCallback(ProgressBarCallback):
    """Show the latest PPO training loss beside timestep progress."""

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self.pbar.set_description_str("loss=waiting", refresh=False)

    def _update_loss(self) -> None:
        loss = self.model.logger.name_to_value.get("train/loss")
        if loss is not None:
            self.pbar.set_description_str(
                f"loss={float(loss):.5g}",
                refresh=False,
            )

    def _on_step(self) -> bool:
        should_continue = super()._on_step()
        self._update_loss()
        return should_continue

    def _on_training_end(self) -> None:
        self._update_loss()
        super()._on_training_end()


def _unique_run_directory(config: TrainingConfig) -> Path:
    root = Path(config.run.output_dir) / config.run.name
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"{timestamp}-seed-{config.run.seed}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _metadata(
    config: TrainingConfig,
    status: str,
    resolved_device: str,
) -> dict[str, Any]:
    spec = ObservationSpec(
        config.environment.max_turns,
    )
    return {
        "bundle_version": BUNDLE_VERSION,
        "status": status,
        "algorithm": config.algorithm.name,
        "device": {
            "requested": config.algorithm.device,
            "resolved": resolved_device,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "seed": config.run.seed,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in (
                "gymnasium",
                "numpy",
                "sb3-contrib",
                "stable-baselines3",
                "torch",
            )
        },
        "game_rules": game_rules_metadata(),
        "observation_schema": spec.metadata(),
        "action_mapping": action_mapping(),
        "configuration": config.to_dict(),
    }


def _smoke_test_bundle(
    run_directory: Path,
    config: TrainingConfig,
    resolved_device: str,
) -> dict[str, Any]:
    spec = ObservationSpec(
        config.environment.max_turns,
    )
    reloaded_model = MaskablePPO.load(
        run_directory / MODEL_FILENAME,
        device=resolved_device,
    )
    selector = RLActionSelector(reloaded_model, spec)
    game_state = GameState.create(
        WorldConfig(
            width=config.environment.width,
            height=config.environment.height,
            fruit_tree_density=config.environment.fruit_tree_density,
            seed=config.run.seed,
        ),
        agent_count=config.environment.agent_count,
    )
    for _ in range(config.environment.max_turns):
        if not game_state.agents:
            break
        game_state = game_state.step(selector)
    return {
        "passed": True,
        "turns": game_state.turn,
        "survivors": len(game_state.agents),
        "claimed_tiles": len(game_state.tile_claims),
        "claims_by_survivor": {
            agent.id: sum(
                owner == agent.number for owner in game_state.tile_claims.values()
            )
            for agent in game_state.agents
        },
        "world_seed": config.run.seed,
    }


def _make_training_environment(config: TrainingConfig) -> VecEnv:
    """Create one local world or several process-isolated parallel worlds."""

    environment_factory = partial(
        JevUGymEnv,
        config.environment,
        config.reward,
    )
    factories = [environment_factory for _ in range(config.algorithm.parallel_envs)]
    if config.algorithm.parallel_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def train(config: TrainingConfig) -> Path:
    """Train, save, reload, and smoke-test one configured model bundle."""

    resolved_device = resolve_device(config.algorithm.device)
    if config.run.console_output != "quiet":
        print(f"RL device: {resolved_device} (requested: {config.algorithm.device})")
        print(f"Parallel worlds: {config.algorithm.parallel_envs}")
    run_directory = _unique_run_directory(config)
    (run_directory / "resolved_config.toml").write_text(config_to_toml(config))
    write_json(
        run_directory / METADATA_FILENAME,
        _metadata(config, "training", resolved_device),
    )
    environment = _make_training_environment(config)
    artifact_callback = ArtifactCallback(
        run_directory,
        config.checkpoint.every_timesteps,
        config.checkpoint.keep_last,
    )
    algorithm = config.algorithm
    show_tables = config.run.console_output == "table"
    show_progress = config.run.console_output == "progress"
    callbacks: list[BaseCallback] = [artifact_callback]
    if show_progress:
        callbacks.append(LossProgressBarCallback())
    model = AgentAwareMaskablePPO(
        "MultiInputPolicy",
        environment,
        learning_rate=algorithm.learning_rate,
        n_steps=algorithm.n_steps,
        batch_size=algorithm.batch_size,
        gamma=algorithm.gamma,
        gae_lambda=algorithm.gae_lambda,
        ent_coef=algorithm.ent_coef,
        seed=config.run.seed,
        device=resolved_device,
        tensorboard_log=str(run_directory / "tensorboard"),
        verbose=int(show_tables),
        rollout_buffer_class=AgentAwareMaskableDictRolloutBuffer,
    )
    try:
        model.learn(
            total_timesteps=config.run.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        model.save(run_directory / MODEL_FILENAME.removesuffix(".zip"))
        write_json(
            run_directory / METADATA_FILENAME,
            _metadata(config, "model_saved", resolved_device),
        )
        smoke_result = _smoke_test_bundle(run_directory, config, resolved_device)
        write_json(run_directory / "smoke_test.json", smoke_result)
        write_json(
            run_directory / METADATA_FILENAME,
            _metadata(config, "complete", resolved_device),
        )
        RLActionSelector.from_bundle(run_directory)
    finally:
        environment.close()
    return run_directory
