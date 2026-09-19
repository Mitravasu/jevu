import unittest
from contextlib import redirect_stdout
from dataclasses import replace
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
            game_state.run(log_directory=None)

        rendered_output = output.getvalue()
        self.assertIn("Start state:\n" + game_state.render_ascii(), rendered_output)
        self.assertIn("Final state:\n" + game_state.render_ascii(), rendered_output)
        self.assertEqual(rendered_output.count("Agent states:"), 2)
        self.assertIn(game_state.render_agent_states(), rendered_output)

    def test_run_simulates_requested_number_of_turns(self) -> None:
        blank_config = WorldConfig(
            width=5,
            height=4,
            fruit_tree_density=0.0,
            seed=42,
        )
        game_state = GameState.create(blank_config, agent_count=2)

        with redirect_stdout(StringIO()):
            final_state = game_state.run(max_turns=3, log_directory=None)

        self.assertEqual(final_state.turn, 3)
        self.assertEqual(len(final_state.action_log), 6)
        self.assertTrue(all(agent.hunger == 7 for agent in final_state.agents))
        first_entry = final_state.action_log[0]
        self.assertEqual(first_entry.start_position, game_state.agents[0].position)
        self.assertEqual(first_entry.start_hunger, 10)
        self.assertEqual(first_entry.end_hunger, 9)
        self.assertEqual(first_entry.start_food, 0)
        self.assertIn("position=(", str(first_entry))
        self.assertIn("hunger=10 -> 9", str(first_entry))
        self.assertIn("food=", str(first_entry))

    def test_simulation_is_deterministic(self) -> None:
        first = GameState.create(self.config, agent_count=2)
        second = GameState.create(self.config, agent_count=2)

        with redirect_stdout(StringIO()):
            first_result = first.run(max_turns=3, log_directory=None)
            second_result = second.run(max_turns=3, log_directory=None)

        self.assertEqual(first_result, second_result)

    def test_agents_die_when_hunger_reaches_zero(self) -> None:
        game_state = GameState.create(self.config, agent_count=1)
        starving_agent = replace(game_state.agents[0], hunger=1)
        game_state = replace(game_state, agents=(starving_agent,))

        with redirect_stdout(StringIO()):
            final_state = game_state.run(max_turns=1, log_directory=None)

        self.assertEqual(final_state.agents, ())

    def test_negative_max_turns_are_rejected(self) -> None:
        game_state = GameState.create(self.config)

        with self.assertRaises(ValueError):
            game_state.run(max_turns=-1, log_directory=None)


if __name__ == "__main__":
    unittest.main()
