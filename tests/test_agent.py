import unittest

from jevu.agent import Agent
from jevu.rules import INITIAL_HUNGER
from jevu.world import Position, World, WorldConfig


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World.generate(WorldConfig(width=5, height=4, seed=42))

    def test_state_contains_local_observation(self) -> None:
        agent = Agent(number=1, position=Position(2, 2))
        state = agent.state(self.world)

        self.assertEqual(state.id, "A1")
        self.assertEqual(state.position, Position(2, 2))
        self.assertEqual(state.current_tile, self.world.tile_at(Position(2, 2)))
        self.assertEqual(state.current_tile_cooldown, 0)
        self.assertEqual(
            state.adjacent_tiles,
            self.world.adjacent_tiles(Position(2, 2)),
        )
        self.assertEqual(state.hunger, INITIAL_HUNGER)
        self.assertEqual(state.inventory.food, 0)

    def test_state_contains_observable_tree_cooldowns(self) -> None:
        agent = Agent(number=1, position=Position(2, 2))
        adjacent_position = self.world.adjacent_positions(agent.position)["up"]

        state = agent.state(
            self.world,
            {
                agent.position: 2,
                adjacent_position: 1,
            },
        )

        self.assertEqual(state.current_tile_cooldown, 2)
        self.assertEqual(state.adjacent_tile_cooldowns["up"], 1)

    def test_rendering_displays_agent_id(self) -> None:
        agent = Agent(number=1, position=Position(0, 0))

        self.assertTrue(self.world.render_ascii([agent]).startswith("A1 "))

if __name__ == "__main__":
    unittest.main()
