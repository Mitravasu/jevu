import tempfile
import unittest
from pathlib import Path

from jevu.rl.config import load_training_config

VALID_CONFIG = """
[run]
name = "test-run"
seed = 3
total_timesteps = 64
output_dir = "artifacts/test"
console_output = "progress"

[environment]
width = 4
height = 3
fruit_tree_density = 0.25
max_turns = 20
agent_count = 1

[reward]
survival_per_turn = 1.0
claim_tile = 2.0
death = -25.0
ineffective_action = -0.05

[algorithm]
name = "maskable_ppo"
device = "auto"
parallel_envs = 2
learning_rate = 0.0003
n_steps = 64
batch_size = 16
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.01

[checkpoint]
every_timesteps = 64
keep_last = 2
"""


class RLConfigTests(unittest.TestCase):
    def _load(self, contents: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(contents)
            return load_training_config(path)

    def test_loads_valid_configuration(self) -> None:
        config = self._load(VALID_CONFIG)

        self.assertEqual(config.run.name, "test-run")
        self.assertEqual(config.run.console_output, "progress")
        self.assertEqual(config.environment.width, 4)
        self.assertEqual(config.algorithm.batch_size, 16)
        self.assertEqual(config.algorithm.device, "auto")
        self.assertEqual(config.algorithm.parallel_envs, 2)

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Unknown \[run\] field"):
            self._load(VALID_CONFIG.replace('name = "test-run"', 'unknown = "x"'))

    def test_accepts_multi_agent_training(self) -> None:
        config = self._load(VALID_CONFIG.replace("agent_count = 1", "agent_count = 2"))

        self.assertEqual(config.environment.agent_count, 2)

    def test_rejects_more_agents_than_tiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "tile count"):
            self._load(VALID_CONFIG.replace("agent_count = 1", "agent_count = 13"))

    def test_rejects_wrong_field_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed must be int"):
            self._load(VALID_CONFIG.replace("seed = 3", 'seed = "three"'))

    def test_smoke_configuration_is_one_valid_rollout(self) -> None:
        config = self._load(VALID_CONFIG).smoke_config()

        rollout_size = config.algorithm.n_steps * config.algorithm.parallel_envs
        self.assertEqual(config.run.total_timesteps, rollout_size)
        self.assertEqual(rollout_size % config.algorithm.batch_size, 0)

    def test_rejects_non_positive_parallel_environment_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "parallel_envs must be positive"):
            self._load(VALID_CONFIG.replace("parallel_envs = 2", "parallel_envs = 0"))

    def test_batch_size_must_divide_combined_rollout(self) -> None:
        contents = VALID_CONFIG.replace("batch_size = 16", "batch_size = 10")
        with self.assertRaisesRegex(ValueError, r"n_steps \* parallel_envs"):
            self._load(contents)

    def test_batch_size_can_span_multiple_environments(self) -> None:
        contents = VALID_CONFIG.replace("n_steps = 64", "n_steps = 32").replace(
            "batch_size = 16", "batch_size = 64"
        )

        config = self._load(contents)

        self.assertEqual(config.algorithm.n_steps, 32)
        self.assertEqual(config.algorithm.parallel_envs, 2)
        self.assertEqual(config.algorithm.batch_size, 64)

    def test_console_output_override_is_validated(self) -> None:
        config = self._load(VALID_CONFIG).with_console_output("quiet")
        self.assertEqual(config.run.console_output, "quiet")

        with self.assertRaisesRegex(ValueError, "console_output"):
            config.with_console_output("noisy")


if __name__ == "__main__":
    unittest.main()
