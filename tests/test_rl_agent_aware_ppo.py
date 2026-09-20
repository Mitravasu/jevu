import unittest

import numpy as np
import torch
from gymnasium import spaces

from jevu.rl.agent_aware_ppo import AgentAwareMaskableDictRolloutBuffer
from jevu.rl.observation import ObservationSpec


class AgentAwareRolloutBufferTests(unittest.TestCase):
    def _buffer(self, size: int) -> AgentAwareMaskableDictRolloutBuffer:
        return AgentAwareMaskableDictRolloutBuffer(
            buffer_size=size,
            observation_space=ObservationSpec(max_turns=10).space(),
            action_space=spaces.MultiDiscrete([4, 5]),
            gamma=1.0,
            gae_lambda=1.0,
            n_envs=1,
        )

    def test_gae_links_each_agent_to_its_own_next_decision(self) -> None:
        buffer = self._buffer(4)
        buffer.rewards[:, 0] = [1.0, 2.0, 10.0, 20.0]
        buffer.values[:, 0] = 0.0
        buffer.agent_keys[:, 0] = [(1, 1), (1, 2), (1, 1), (1, 2)]
        buffer.survived[:, 0] = True

        buffer.compute_returns_and_advantage(torch.zeros(1), np.asarray([False]))

        np.testing.assert_allclose(buffer.advantages[:, 0], [11.0, 22.0, 10.0, 20.0])

    def test_death_terminates_once_and_does_not_link_to_new_episode(self) -> None:
        buffer = self._buffer(2)
        buffer.rewards[:, 0] = [-100.0, 5.0]
        buffer.values[:, 0] = 0.0
        buffer.agent_keys[:, 0] = [(1, 1), (2, 1)]
        buffer.survived[:, 0] = [False, True]

        buffer.compute_returns_and_advantage(torch.zeros(1), np.asarray([False]))

        np.testing.assert_allclose(buffer.advantages[:, 0], [-100.0, 5.0])


if __name__ == "__main__":
    unittest.main()
