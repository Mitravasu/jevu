import unittest
from unittest.mock import Mock, patch

from jevu.rl.config import (
    AlgorithmConfig,
    CheckpointConfig,
    EnvironmentConfig,
    RewardConfig,
    RunConfig,
    TrainingConfig,
)
from jevu.rl.trainer import _make_training_environment


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


if __name__ == "__main__":
    unittest.main()
