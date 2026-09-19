import unittest

from jevu.world import Position, TileType, World, WorldConfig


class WorldGenerationTests(unittest.TestCase):
    def test_same_seed_generates_same_world(self) -> None:
        config = WorldConfig(width=8, height=6, fruit_tree_density=0.25, seed=42)

        self.assertEqual(World.generate(config), World.generate(config))

    def test_different_seeds_generate_different_worlds(self) -> None:
        first = World.generate(WorldConfig(seed=1))
        second = World.generate(WorldConfig(seed=2))

        self.assertNotEqual(first.tiles, second.tiles)

    def test_density_extremes(self) -> None:
        blank_world = World.generate(WorldConfig(fruit_tree_density=0.0))
        tree_world = World.generate(WorldConfig(fruit_tree_density=1.0))

        self.assertTrue(all(tile is TileType.BLANK for row in blank_world.tiles for tile in row))
        self.assertTrue(
            all(tile is TileType.FRUIT_TREE for row in tree_world.tiles for tile in row)
        )

    def test_adjacent_tiles_exclude_diagonals_and_boundaries(self) -> None:
        world = World.generate(WorldConfig(width=3, height=3, seed=7))

        self.assertEqual(set(world.adjacent_tiles(Position(0, 0))), {"down", "right"})
        self.assertEqual(
            set(world.adjacent_tiles(Position(1, 1))),
            {"up", "down", "left", "right"},
        )

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WorldConfig(width=0)
        with self.assertRaises(ValueError):
            WorldConfig(fruit_tree_density=1.1)


if __name__ == "__main__":
    unittest.main()
