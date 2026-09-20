import unittest

import numpy as np

from jevu.agent import Agent
from jevu.rl.observation import ObservationSpec, encode_observation
from jevu.world import Position, TileType, World, WorldConfig


class RLObservationTests(unittest.TestCase):
    def test_encodes_visible_tiles_without_revealing_hidden_tiles(self) -> None:
        world = World(
            WorldConfig(width=3, height=3),
            (
                (TileType.BLANK, TileType.BLANK, TileType.FRUIT_TREE),
                (TileType.BLANK, TileType.FRUIT_TREE, TileType.BLANK),
                (TileType.FRUIT_TREE, TileType.BLANK, TileType.FRUIT_TREE),
            ),
        )
        state = Agent(1, Position(1, 1), hunger=5).state(world)
        spec = ObservationSpec(width=3, height=3, max_turns=20)

        observation = encode_observation(state, spec)

        self.assertTrue(spec.space().contains(observation))
        self.assertEqual(int(observation["map"][0].sum()), 5)
        self.assertEqual(observation["map"][0, 0, 0], 0.0)
        self.assertEqual(observation["map"][1, 0, 2], 0.0)
        self.assertEqual(observation["map"][5, 1, 1], 1.0)
        np.testing.assert_allclose(observation["scalars"], [0.5, 0.0, 0.5, 0.5])

    def test_smaller_world_is_padded_to_the_model_canvas(self) -> None:
        world = World.generate(WorldConfig(width=2, height=2))
        state = Agent(1, Position(0, 0)).state(world)

        observation = encode_observation(
            state, ObservationSpec(width=3, height=3, max_turns=10)
        )

        self.assertEqual(observation["map"].shape, (6, 3, 3))
        self.assertEqual(observation["map"][:, 2, :].sum(), 0.0)
        self.assertEqual(observation["map"][:, :, 2].sum(), 0.0)

    def test_rejects_world_dimensions_larger_than_model(self) -> None:
        world = World.generate(WorldConfig(width=4, height=2))
        state = Agent(1, Position(0, 0)).state(world)

        with self.assertRaisesRegex(ValueError, "world dimensions exceed"):
            encode_observation(state, ObservationSpec(width=3, height=2, max_turns=10))


if __name__ == "__main__":
    unittest.main()
