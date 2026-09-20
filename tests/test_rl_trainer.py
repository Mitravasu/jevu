import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch as th

from jevu.rl.bundle import METADATA_FILENAME, MODEL_FILENAME
from jevu.rl.config import (
    AlgorithmConfig,
    CheckpointConfig,
    EnvironmentConfig,
    RewardConfig,
    RunConfig,
    TrainingConfig,
)
from jevu.rl.observation import OBSERVATION_SCHEMA_VERSION
from jevu.rl.trainer import (
    _expanded_input_weight,
    _make_training_environment,
    _metadata,
    _resolve_resume_source,
    _resolve_warm_start_source,
)


def training_config(parallel_envs: int) -> TrainingConfig:
    return TrainingConfig(
        run=RunConfig(total_timesteps=64),
        environment=EnvironmentConfig(),
        reward=RewardConfig(),
        algorithm=AlgorithmConfig(
            parallel_envs=parallel_envs,
            n_steps=64,
            batch_size=64,
        ),
        checkpoint=CheckpointConfig(),
    )


class RLTrainerTests(unittest.TestCase):
    def _resume_bundle(
        self,
        directory: str,
        *,
        observation_version: int = OBSERVATION_SCHEMA_VERSION,
        status: str = "complete",
    ) -> Path:
        bundle = Path(directory) / "bundle"
        bundle.mkdir()
        config = training_config(1)
        metadata = _metadata(
            config,
            status,
            "cpu",
            current_timesteps=64,
        )
        metadata["observation_schema"]["version"] = observation_version
        (bundle / METADATA_FILENAME).write_text(json.dumps(metadata))
        (bundle / MODEL_FILENAME).touch()
        return bundle

    def _schema_v2_bundle(self, directory: str) -> Path:
        bundle = self._resume_bundle(directory, observation_version=2)
        metadata_path = bundle / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text())
        observation = metadata["observation_schema"]
        observation["channels"] = observation["channels"][:-2]
        observation.pop("recent_position_window")
        metadata_path.write_text(json.dumps(metadata))
        return bundle

    @patch("jevu.rl.trainer.DummyVecEnv")
    def test_single_world_uses_local_vector_environment(self, vector_type) -> None:
        expected = Mock()
        vector_type.return_value = expected

        environment = _make_training_environment(training_config(1))

        self.assertIs(environment, expected)
        factories = vector_type.call_args.args[0]
        self.assertEqual(len(factories), 1)

    @patch("jevu.rl.trainer.SubprocVecEnv")
    def test_multiple_worlds_use_spawned_processes(self, vector_type) -> None:
        expected = Mock()
        vector_type.return_value = expected

        environment = _make_training_environment(training_config(4))

        self.assertIs(environment, expected)
        factories = vector_type.call_args.args[0]
        self.assertEqual(len(factories), 4)
        self.assertEqual(vector_type.call_args.kwargs, {"start_method": "spawn"})
        self.assertIsNot(factories[0](), factories[1]())

    def test_completed_bundle_is_a_valid_resume_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._resume_bundle(directory)

            source = _resolve_resume_source(bundle, training_config(1))

            self.assertEqual(source.bundle_directory, bundle)
            self.assertEqual(source.model_path, bundle / MODEL_FILENAME)

    def test_checkpoint_can_resume_an_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._resume_bundle(directory, status="training")
            checkpoint_directory = bundle / "checkpoints"
            checkpoint_directory.mkdir()
            checkpoint = checkpoint_directory / "model-64.zip"
            checkpoint.touch()

            source = _resolve_resume_source(checkpoint, training_config(1))

            self.assertEqual(source.bundle_directory, bundle)
            self.assertEqual(source.model_path, checkpoint)

    def test_outdated_schema_bundle_cannot_be_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._resume_bundle(directory, observation_version=1)

            with self.assertRaisesRegex(ValueError, "current schema"):
                _resolve_resume_source(bundle, training_config(1))

    def test_schema_v2_bundle_is_a_valid_warm_start_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._schema_v2_bundle(directory)

            source = _resolve_warm_start_source(bundle, training_config(1))

            self.assertEqual(source.bundle_directory, bundle)
            self.assertEqual(source.model_path, bundle / MODEL_FILENAME)

    def test_expanded_input_weight_preserves_old_features(self) -> None:
        source = th.arange(12, dtype=th.float32).reshape(2, 6)
        target = th.ones((2, 8), dtype=th.float32)

        expanded = _expanded_input_weight(
            source,
            target,
            source_map_features=4,
            target_map_features=6,
            scalar_features=2,
        )

        th.testing.assert_close(expanded[:, :4], source[:, :4])
        th.testing.assert_close(expanded[:, 4:6], th.zeros((2, 2)))
        th.testing.assert_close(expanded[:, 6:], source[:, 4:])


if __name__ == "__main__":
    unittest.main()
