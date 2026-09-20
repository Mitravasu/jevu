import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.rl.action_codec import encode_actions
from jevu.rl.config import EnvironmentConfig, RewardConfig
from jevu.rl.environment import JevUGymEnv


class RLEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment_config = EnvironmentConfig(
            width=3,
            height=3,
            fruit_tree_density=0.0,
            max_turns=3,
            agent_count=1,
        )
        self.reward_config = RewardConfig()

    def test_passes_gymnasium_environment_checker(self) -> None:
        environment = JevUGymEnv(self.environment_config, self.reward_config)
        check_env(environment, skip_render_check=True)

    def test_seeded_trajectories_are_reproducible(self) -> None:
        first = JevUGymEnv(self.environment_config, self.reward_config)
        second = JevUGymEnv(self.environment_config, self.reward_config)
        first_observation, first_info = first.reset(seed=42)
        second_observation, second_info = second.reset(seed=42)
        self.assertEqual(first_info, second_info)
        np.testing.assert_array_equal(
            first_observation["map"], second_observation["map"]
        )

        action = encode_actions(
            TurnActions(interact=InteractAction.CLAIM, explore=ExploreAction.STAY)
        )
        first_step = first.step(action)
        second_step = second.step(action)

        np.testing.assert_array_equal(first_step[0]["map"], second_step[0]["map"])
        self.assertEqual(first_step[1:], second_step[1:])

    def test_claim_reward_is_only_paid_once(self) -> None:
        environment = JevUGymEnv(self.environment_config, self.reward_config)
        environment.reset(seed=1)
        action = encode_actions(
            TurnActions(interact=InteractAction.CLAIM, explore=ExploreAction.STAY)
        )

        _, first_reward, _, _, first_info = environment.step(action)
        _, second_reward, _, _, second_info = environment.step(action)

        self.assertEqual(first_info["newly_claimed_tiles"], 1)
        self.assertEqual(first_reward, 3.0)
        self.assertEqual(second_info["newly_claimed_tiles"], 0)
        self.assertEqual(second_reward, 0.95)

    def test_shared_policy_trains_each_agent_in_one_world(self) -> None:
        config = EnvironmentConfig(
            width=2,
            height=1,
            fruit_tree_density=0.0,
            max_turns=1,
            agent_count=2,
        )
        environment = JevUGymEnv(config, self.reward_config)
        environment.reset(seed=7)
        claim = encode_actions(
            TurnActions(interact=InteractAction.CLAIM, explore=ExploreAction.STAY)
        )

        second_observation, first_reward, terminated, truncated, first_info = (
            environment.step(claim)
        )

        self.assertEqual(first_info["acting_agent_id"], "A1")
        self.assertEqual(first_reward, 3.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(environment.game_state.turn, 0)
        self.assertEqual(second_observation["map"][4].sum(), 1.0)

        _, second_reward, terminated, truncated, second_info = environment.step(claim)

        self.assertEqual(second_info["acting_agent_id"], "A2")
        self.assertEqual(second_reward, 3.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(environment.game_state.turn, 1)
        self.assertEqual(len(environment.game_state.tile_claims), 2)
        self.assertEqual(second_info["episode_summary"]["agent_decisions"], 2)
        self.assertEqual(second_info["episode_summary"]["world_turns"], 1)


if __name__ == "__main__":
    unittest.main()
