import unittest

import numpy as np

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import Agent
from jevu.rl.action_codec import (
    ACTION_SHAPE,
    action_masks,
    decode_actions,
    encode_actions,
)
from jevu.world import Position, TileType, World, WorldConfig


class RLActionCodecTests(unittest.TestCase):
    def test_every_action_pair_round_trips(self) -> None:
        for interact in InteractAction:
            for explore in ExploreAction:
                actions = TurnActions(interact=interact, explore=explore)
                self.assertEqual(decode_actions(encode_actions(actions)), actions)

    def test_rejects_out_of_range_and_negative_actions(self) -> None:
        with self.assertRaises(ValueError):
            decode_actions([-1, 0])
        with self.assertRaises(ValueError):
            decode_actions([0, ACTION_SHAPE[1]])

    def test_masks_impossible_actions_at_blank_corner(self) -> None:
        world = World(
            WorldConfig(width=2, height=2, fruit_tree_density=0.0),
            ((TileType.BLANK, TileType.BLANK), (TileType.BLANK, TileType.BLANK)),
        )
        state = Agent(1, Position(0, 0)).state(world)

        masks = action_masks(state, width=2, height=2)

        np.testing.assert_array_equal(
            masks,
            np.asarray([False, False, True, True, False, True, False, True, True]),
        )


if __name__ == "__main__":
    unittest.main()
