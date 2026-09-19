import unittest

from jevu.agent import Agent, place_agents
from jevu.world import Position, World, WorldConfig


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World.generate(WorldConfig(width=5, height=4, seed=42))

    def test_placement_is_deterministic(self) -> None:
        self.assertEqual(place_agents(self.world, 3), place_agents(self.world, 3))

    def test_agents_have_unique_positions_and_numbered_ids(self) -> None:
        agents = place_agents(self.world, 3)

        self.assertEqual([agent.id for agent in agents], ["A1", "A2", "A3"])
        self.assertEqual(len({agent.position for agent in agents}), 3)

    def test_state_contains_local_observation(self) -> None:
        agent = Agent(number=1, position=Position(2, 2))
        state = agent.state(self.world)

        self.assertEqual(state.id, "A1")
        self.assertEqual(state.position, Position(2, 2))
        self.assertEqual(state.current_tile, self.world.tile_at(Position(2, 2)))
        self.assertEqual(
            state.adjacent_tiles,
            self.world.adjacent_tiles(Position(2, 2)),
        )
        self.assertEqual(state.hunger, 10)
        self.assertEqual(state.carried_fruit, 0)

    def test_rendering_displays_agent_id(self) -> None:
        agent = Agent(number=1, position=Position(0, 0))

        self.assertTrue(self.world.render_ascii([agent]).startswith("A1 "))

    def test_too_many_agents_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            place_agents(self.world, 21)


if __name__ == "__main__":
    unittest.main()
