"""Configuration-driven MaskablePPO training and bundle creation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import zlib
from dataclasses import dataclass
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
        self,
        run_directory: Path,
        every_timesteps: int,
        keep_last: int,
        starting_timesteps: int = 0,
    ) -> None:
        super().__init__()
        self.run_directory = run_directory
        self.every_timesteps = every_timesteps
        self.keep_last = keep_last
        self.next_checkpoint = (
            (starting_timesteps // every_timesteps) + 1
        ) * every_timesteps
        self.starting_timesteps = starting_timesteps
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
                record = {
                    "timesteps": self.num_timesteps,
                    "run_timesteps": self.num_timesteps - self.starting_timesteps,
                    **summary,
                }
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


@dataclass(frozen=True, slots=True)
class ResumeSource:
    """Validated model archive and metadata used to continue training."""

    bundle_directory: Path
    model_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WarmStartSource:
    """Validated legacy bundle used only to initialize policy weights."""

    bundle_directory: Path
    model_path: Path
    metadata: dict[str, Any]


def _resolve_resume_source(
    path: str | Path,
    config: TrainingConfig,
) -> ResumeSource:
    """Resolve and validate a final bundle or recovery checkpoint."""

    requested = Path(path)
    if requested.is_dir():
        bundle_directory = requested
        model_path = requested / MODEL_FILENAME
        require_complete = True
    elif requested.is_file() and requested.suffix == ".zip":
        model_path = requested
        bundle_directory = (
            requested.parent.parent
            if requested.parent.name == "checkpoints"
            else requested.parent
        )
        require_complete = False
    else:
        raise ValueError(
            "--resume must be a completed bundle directory or checkpoint .zip"
        )

    metadata_path = bundle_directory / METADATA_FILENAME
    if not metadata_path.is_file() or not model_path.is_file():
        raise ValueError(f"Incomplete resume source: {requested}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    if metadata.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Resume source uses an unsupported bundle version")
    if require_complete and metadata.get("status") != "complete":
        raise ValueError("Only a completed bundle directory can be resumed")
    if metadata.get("algorithm") != config.algorithm.name:
        raise ValueError("Resume source uses an incompatible RL algorithm")

    expected_observation = ObservationSpec(
        max_turns=config.environment.max_turns
    ).metadata()
    if metadata.get("observation_schema") != expected_observation:
        raise ValueError(
            "Resume source uses an incompatible observation schema; "
            "only models with the current schema can be continued"
        )
    if metadata.get("action_mapping") != action_mapping():
        raise ValueError("Resume source uses an incompatible action mapping")
    if metadata.get("game_rules") != game_rules_metadata():
        raise ValueError("Resume source uses incompatible game rules")
    return ResumeSource(bundle_directory, model_path, metadata)


def _resolve_warm_start_source(
    path: str | Path,
    config: TrainingConfig,
) -> WarmStartSource:
    """Validate a completed schema-v2 bundle for schema-v3 weight migration."""

    bundle_directory = Path(path)
    model_path = bundle_directory / MODEL_FILENAME
    metadata_path = bundle_directory / METADATA_FILENAME
    if (
        not bundle_directory.is_dir()
        or not metadata_path.is_file()
        or not model_path.is_file()
    ):
        raise ValueError("--warm-start must be a completed schema-v2 bundle directory")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    if metadata.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Warm-start source uses an unsupported bundle version")
    if metadata.get("status") != "complete":
        raise ValueError("Warm-start source must be a completed bundle")
    if metadata.get("algorithm") != config.algorithm.name:
        raise ValueError("Warm-start source uses an incompatible RL algorithm")
    if metadata.get("action_mapping") != action_mapping():
        raise ValueError("Warm-start source uses an incompatible action mapping")
    if metadata.get("game_rules") != game_rules_metadata():
        raise ValueError("Warm-start source uses incompatible game rules")

    source = metadata.get("observation_schema", {})
    target = ObservationSpec(max_turns=config.environment.max_turns).metadata()
    source_channels = source.get("channels")
    target_channels = target["channels"]
    compatible_fields = ("view", "window_size", "max_turns", "max_food", "scalars")
    if (
        source.get("version") != 2
        or not isinstance(source_channels, list)
        or source_channels != target_channels[: len(source_channels)]
        or len(target_channels) - len(source_channels) != 2
        or any(source.get(field) != target[field] for field in compatible_fields)
    ):
        raise ValueError(
            "Warm-start source must use the compatible schema-v2 observation"
        )
    return WarmStartSource(bundle_directory, model_path, metadata)


def _expanded_input_weight(
    source_weight: Any,
    target_weight: Any,
    *,
    source_map_features: int,
    target_map_features: int,
    scalar_features: int,
) -> Any:
    """Expand one input layer while preserving old map and scalar behavior."""

    if source_weight.shape[1] != source_map_features + scalar_features:
        raise ValueError("Warm-start source policy has an unexpected input shape")
    if target_weight.shape[1] != target_map_features + scalar_features:
        raise ValueError("Warm-start target policy has an unexpected input shape")
    expanded = target_weight.new_zeros(target_weight.shape)
    expanded[:, :source_map_features] = source_weight[:, :source_map_features]
    expanded[:, target_map_features:] = source_weight[:, source_map_features:]
    return expanded


def _warm_start_policy(
    source_model: AgentAwareMaskablePPO,
    target_model: AgentAwareMaskablePPO,
    source_metadata: dict[str, Any],
    target_observation: dict[str, object],
) -> None:
    """Copy a schema-v2 policy into schema v3 with zeroed new-channel weights."""

    source_observation = source_metadata["observation_schema"]
    window_area = int(target_observation["window_size"]) ** 2
    source_map_features = len(source_observation["channels"]) * window_area
    target_map_features = len(target_observation["channels"]) * window_area
    scalar_features = len(target_observation["scalars"])
    expanded_keys = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    migrated_state: dict[str, Any] = {}
    for key, target_value in target_state.items():
        if key not in source_state:
            raise ValueError(f"Warm-start source policy is missing parameter {key}")
        source_value = source_state[key]
        if key in expanded_keys:
            migrated_state[key] = _expanded_input_weight(
                source_value,
                target_value,
                source_map_features=source_map_features,
                target_map_features=target_map_features,
                scalar_features=scalar_features,
            )
        elif source_value.shape != target_value.shape:
            raise ValueError(f"Warm-start policy parameter has changed shape: {key}")
        else:
            migrated_state[key] = source_value
    target_model.policy.load_state_dict(migrated_state)


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
    *,
    current_timesteps: int,
    starting_timesteps: int = 0,
    resume_source: ResumeSource | None = None,
    warm_start_source: WarmStartSource | None = None,
    effective_seed: int | None = None,
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
        "effective_seed": (
            config.run.seed if effective_seed is None else effective_seed
        ),
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
        "timesteps": {
            "requested_additional": config.run.total_timesteps,
            "starting": starting_timesteps,
            "completed_additional": max(0, current_timesteps - starting_timesteps),
            "cumulative": current_timesteps,
        },
        "lineage": (
            {
                "mode": "resume",
                "parent_bundle": str(resume_source.bundle_directory),
                "source_model": str(resume_source.model_path),
            }
            if resume_source is not None
            else (
                {
                    "mode": "warm_start",
                    "parent_bundle": str(warm_start_source.bundle_directory),
                    "source_model": str(warm_start_source.model_path),
                    "source_observation_schema": warm_start_source.metadata.get(
                        "observation_schema"
                    ),
                    "source_timesteps": warm_start_source.metadata.get("timesteps"),
                }
                if warm_start_source is not None
                else None
            )
        ),
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


def train(
    config: TrainingConfig,
    *,
    resume_from: str | Path | None = None,
    warm_start_from: str | Path | None = None,
) -> Path:
    """Train, save, reload, and smoke-test one configured model bundle."""

    if resume_from is not None and warm_start_from is not None:
        raise ValueError("--resume and --warm-start cannot be used together")
    resolved_device = resolve_device(config.algorithm.device)
    resume_source = (
        _resolve_resume_source(resume_from, config)
        if resume_from is not None
        else None
    )
    warm_start_source = (
        _resolve_warm_start_source(warm_start_from, config)
        if warm_start_from is not None
        else None
    )
    effective_seed = config.run.seed
    initialization_source = resume_source or warm_start_source
    if initialization_source is not None:
        source_hash = zlib.crc32(str(initialization_source.model_path).encode())
        effective_seed = (config.run.seed + source_hash) % (2**31 - 1)
    if config.run.console_output != "quiet":
        print(f"RL device: {resolved_device} (requested: {config.algorithm.device})")
        print(f"Parallel worlds: {config.algorithm.parallel_envs}")
        if resume_source is not None:
            print(f"Resuming model: {resume_source.model_path}")
            source_algorithm = resume_source.metadata.get("configuration", {}).get(
                "algorithm", {}
            )
            source_rollout_size = source_algorithm.get("n_steps", 0) * source_algorithm.get(
                "parallel_envs", 0
            )
            current_rollout_size = (
                config.algorithm.n_steps * config.algorithm.parallel_envs
            )
            if source_rollout_size and source_rollout_size != current_rollout_size:
                print(
                    "Resume rollout size changed: "
                    f"{source_rollout_size:,} -> {current_rollout_size:,} decisions/update "
                    f"(parallel_envs={source_algorithm['parallel_envs']} -> "
                    f"{config.algorithm.parallel_envs}, "
                    f"n_steps={source_algorithm['n_steps']} -> "
                    f"{config.algorithm.n_steps})"
                )
        elif warm_start_source is not None:
            print(f"Warm-starting policy: {warm_start_source.model_path}")
            print("Optimizer and timestep count start fresh under schema v3")
    run_directory = _unique_run_directory(config)
    (run_directory / "resolved_config.toml").write_text(config_to_toml(config))
    environment = _make_training_environment(config)
    algorithm = config.algorithm
    show_tables = config.run.console_output == "table"
    show_progress = config.run.console_output == "progress"
    try:
        model_parameters: dict[str, Any] = {
            "learning_rate": algorithm.learning_rate,
            "n_steps": algorithm.n_steps,
            "batch_size": algorithm.batch_size,
            "gamma": algorithm.gamma,
            "gae_lambda": algorithm.gae_lambda,
            "ent_coef": algorithm.ent_coef,
            "seed": effective_seed,
            "tensorboard_log": str(run_directory / "tensorboard"),
            "verbose": int(show_tables),
            "rollout_buffer_class": AgentAwareMaskableDictRolloutBuffer,
        }
        if resume_source is not None:
            model = AgentAwareMaskablePPO.load(
                resume_source.model_path,
                env=environment,
                device=resolved_device,
                **model_parameters,
            )
        else:
            model = AgentAwareMaskablePPO(
                "MultiInputPolicy",
                environment,
                device=resolved_device,
                **model_parameters,
            )
        if warm_start_source is not None:
            source_model = AgentAwareMaskablePPO.load(
                warm_start_source.model_path,
                device=resolved_device,
            )
            _warm_start_policy(
                source_model,
                model,
                warm_start_source.metadata,
                ObservationSpec(config.environment.max_turns).metadata(),
            )
        starting_timesteps = model.num_timesteps
        write_json(
            run_directory / METADATA_FILENAME,
            _metadata(
                config,
                "training",
                resolved_device,
                current_timesteps=starting_timesteps,
                starting_timesteps=starting_timesteps,
                resume_source=resume_source,
                warm_start_source=warm_start_source,
                effective_seed=effective_seed,
            ),
        )
        artifact_callback = ArtifactCallback(
            run_directory,
            config.checkpoint.every_timesteps,
            config.checkpoint.keep_last,
            starting_timesteps=starting_timesteps,
        )
        callbacks: list[BaseCallback] = [artifact_callback]
        if show_progress:
            callbacks.append(LossProgressBarCallback())
        model.learn(
            total_timesteps=config.run.total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=resume_source is None,
        )
        model.save(run_directory / MODEL_FILENAME.removesuffix(".zip"))
        write_json(
            run_directory / METADATA_FILENAME,
            _metadata(
                config,
                "model_saved",
                resolved_device,
                current_timesteps=model.num_timesteps,
                starting_timesteps=starting_timesteps,
                resume_source=resume_source,
                warm_start_source=warm_start_source,
                effective_seed=effective_seed,
            ),
        )
        smoke_result = _smoke_test_bundle(run_directory, config, resolved_device)
        write_json(run_directory / "smoke_test.json", smoke_result)
        write_json(
            run_directory / METADATA_FILENAME,
            _metadata(
                config,
                "complete",
                resolved_device,
                current_timesteps=model.num_timesteps,
                starting_timesteps=starting_timesteps,
                resume_source=resume_source,
                warm_start_source=warm_start_source,
                effective_seed=effective_seed,
            ),
        )
        RLActionSelector.from_bundle(run_directory)
    finally:
        environment.close()
    return run_directory
