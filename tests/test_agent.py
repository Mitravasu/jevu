import unittest

from jevu.agent import Agent
from jevu.personalities import Personality
from jevu.rules import INITIAL_HUNGER
from jevu.world import Position, World, WorldConfig


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World.generate(WorldConfig(width=5, height=4, seed=42))

    def test_state_contains_local_observation(self) -> None:
        agent = Agent(number=1, position=Position(2, 2))
        state = agent.state(self.world)

        self.assertEqual(state.id, "A1")
        self.assertEqual(state.world_width, 5)
        self.assertEqual(state.world_height, 4)
        self.assertEqual(state.position, Position(2, 2))
        self.assertEqual(state.current_tile, self.world.tile_at(Position(2, 2)))
        self.assertEqual(state.current_tile_cooldown, 0)
        self.assertEqual(
            state.adjacent_tiles,
            self.world.adjacent_tiles(Position(2, 2)),
        )
        self.assertEqual(state.hunger, INITIAL_HUNGER)
        self.assertEqual(state.inventory.food, 0)
        self.assertEqual(state.personality, Personality.ADAPTIVE_SURVIVOR)

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

    def test_state_contains_observable_tile_claims(self) -> None:
        agent = Agent(number=1, position=Position(2, 2))
        adjacent_position = self.world.adjacent_positions(agent.position)["up"]

        state = agent.state(
            self.world,
            tile_claims={agent.position: 1, adjacent_position: 2},
        )

        self.assertEqual(state.current_tile_claim, "A1")
        self.assertEqual(state.adjacent_tile_claims["up"], "A2")
        self.assertEqual(len(state.claimed_territory), 1)
        claimed = state.claimed_territory[0]
        self.assertEqual(claimed.tile.position, agent.position)
        self.assertEqual(claimed.tile.claim, "A1")
        self.assertEqual(
            claimed.adjacent_tiles["up"].position,
            adjacent_position,
        )
        self.assertEqual(claimed.adjacent_tiles["up"].claim, "A2")

    def test_rendering_displays_agent_id(self) -> None:
        agent = Agent(number=1, position=Position(0, 0))

        self.assertTrue(self.world.render_ascii([agent]).startswith("A1 "))


if __name__ == "__main__":
    unittest.main()
