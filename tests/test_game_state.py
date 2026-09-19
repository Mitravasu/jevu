import unittest
from contextlib import redirect_stdout
from io import StringIO

from jevu.game_state import GameState
from jevu.world import WorldConfig


class GameStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WorldConfig(width=5, height=4, seed=42)

    def test_create_generates_world_and_agents(self) -> None:
        game_state = GameState.create(self.config, agent_count=3)

        self.assertEqual(game_state.world.config, self.config)
        self.assertEqual([agent.id for agent in game_state.agents], ["A1", "A2", "A3"])
        self.assertEqual(len({agent.position for agent in game_state.agents}), 3)

    def test_initial_setup_is_deterministic(self) -> None:
        self.assertEqual(
            GameState.create(self.config, agent_count=3),
            GameState.create(self.config, agent_count=3),
        )

    def test_too_many_agents_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GameState.create(self.config, agent_count=21)

    def test_rendering_includes_every_agent(self) -> None:
        rendered = GameState.create(self.config, agent_count=3).render_ascii()

        self.assertIn("A1", rendered)
        self.assertIn("A2", rendered)
        self.assertIn("A3", rendered)

    def test_run_outputs_the_game(self) -> None:
        game_state = GameState.create(self.config, agent_count=1)
        output = StringIO()

        with redirect_stdout(output):
            game_state.run()

        self.assertEqual(output.getvalue().strip(), game_state.render_ascii())


if __name__ == "__main__":
    unittest.main()
