import unittest

from jevu.rl.config import RewardConfig
from jevu.rl.reward import TransitionEvents, calculate_reward


class RLRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RewardConfig(
            survival_per_turn=1.0,
            claim_tile=2.0,
            death=-25.0,
            ineffective_action=-0.05,
        )

    def test_rewards_survival_and_new_territory(self) -> None:
        reward = calculate_reward(
            TransitionEvents(
                survived=True,
                newly_claimed_tiles=1,
                ineffective_actions=0,
            ),
            self.config,
        )

        self.assertEqual(reward.survival, 1.0)
        self.assertEqual(reward.territory, 2.0)
        self.assertEqual(reward.total, 3.0)

    def test_death_replaces_survival_reward(self) -> None:
        reward = calculate_reward(
            TransitionEvents(
                survived=False,
                newly_claimed_tiles=0,
                ineffective_actions=1,
            ),
            self.config,
        )

        self.assertEqual(reward.survival, 0.0)
        self.assertEqual(reward.death, -25.0)
        self.assertEqual(reward.total, -25.05)

    def test_rejects_negative_event_counts(self) -> None:
        with self.assertRaises(ValueError):
            calculate_reward(TransitionEvents(True, -1, 0), self.config)


if __name__ == "__main__":
    unittest.main()
