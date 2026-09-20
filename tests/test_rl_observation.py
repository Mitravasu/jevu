import unittest

import numpy as np

from jevu.agent import Agent
from jevu.rl.observation import CHANNEL_NAMES, ObservationSpec, encode_observation
from jevu.world import Position, TileType, World, WorldConfig


class RLObservationTests(unittest.TestCase):
    def test_encodes_only_one_tile_orthogonal_view(self) -> None:
        world = World(
            WorldConfig(width=3, height=3),
            (
                (TileType.FRUIT_TREE, TileType.BLANK, TileType.FRUIT_TREE),
                (TileType.BLANK, TileType.FRUIT_TREE, TileType.BLANK),
                (TileType.FRUIT_TREE, TileType.BLANK, TileType.FRUIT_TREE),
            ),
        )
        state = Agent(1, Position(1, 1), hunger=5).state(
            world,
            agent_positions={Position(1, 1): 1, Position(2, 1): 2},
        )
        spec = ObservationSpec(max_turns=20)

        observation = encode_observation(state, spec)

        self.assertTrue(spec.space().contains(observation))
        self.assertEqual(observation["map"].shape, (10, 3, 3))
        self.assertEqual(int(observation["map"][0].sum()), 5)
        self.assertEqual(observation["map"][0, 0, 0], 0.0)
        self.assertEqual(observation["map"][2, 0, 0], 0.0)
        self.assertEqual(observation["map"][6, 1, 2], 1.0)
        self.assertEqual(observation["map"][7, 1, 1], 1.0)
        np.testing.assert_allclose(observation["scalars"], [0.5, 0.0])

    def test_encodes_recent_visit_counts_for_visible_tiles(self) -> None:
        world = World.generate(
            WorldConfig(width=3, height=3, fruit_tree_density=0.0)
        )
        current = Position(1, 1)
        left = Position(0, 1)
        state = Agent(
            1,
            current,
            recent_positions=(left, current, left, left, current),
        ).state(world)

        observation = encode_observation(state, ObservationSpec(max_turns=20))
        visits = observation["map"][CHANNEL_NAMES.index("recent_visit_count")]

        self.assertAlmostEqual(visits[1, 0], 0.3)
        self.assertAlmostEqual(visits[1, 1], 0.2)
        self.assertEqual(visits[1, 2], 0.0)

    def test_marks_only_visible_unclaimed_frontier_tiles(self) -> None:
        world = World.generate(
            WorldConfig(width=3, height=3, fruit_tree_density=0.0)
        )
        current = Position(1, 1)
        left = Position(0, 1)
        right = Position(2, 1)
        state = Agent(1, current).state(
            world,
            tile_claims={current: 1, right: 2},
        )

        observation = encode_observation(state, ObservationSpec(max_turns=20))
        frontier = observation["map"][CHANNEL_NAMES.index("unclaimed_frontier")]

        self.assertEqual(frontier[1, 0], 1.0)
        self.assertEqual(frontier[1, 1], 0.0)
        self.assertEqual(frontier[1, 2], 0.0)
        self.assertEqual(state.adjacent_tile_claims["left"], None)
        self.assertEqual(state.adjacent_tile_claims["right"], "A2")

    def test_same_local_state_matches_across_locations_and_world_sizes(self) -> None:
        small = World(
            WorldConfig(width=3, height=3),
            tuple((TileType.BLANK,) * 3 for _ in range(3)),
        )
        large = World(
            WorldConfig(width=25, height=25),
            tuple((TileType.BLANK,) * 25 for _ in range(25)),
        )
        first = Agent(1, Position(1, 1)).state(small)
        second = Agent(1, Position(12, 12)).state(large)
        spec = ObservationSpec(max_turns=10)

        first_observation = encode_observation(first, spec)
        second_observation = encode_observation(second, spec)

        np.testing.assert_array_equal(
            first_observation["map"], second_observation["map"]
        )
        np.testing.assert_array_equal(
            first_observation["scalars"], second_observation["scalars"]
        )

    def test_marks_visible_world_boundaries(self) -> None:
        world = World.generate(WorldConfig(width=2, height=2))
        state = Agent(1, Position(0, 0)).state(world)

        observation = encode_observation(state, ObservationSpec(max_turns=10))

        self.assertEqual(observation["map"][1, 0, 1], 1.0)
        self.assertEqual(observation["map"][1, 1, 0], 1.0)
        self.assertEqual(observation["map"][1, 1, 2], 0.0)
        self.assertEqual(observation["map"][1, 2, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
