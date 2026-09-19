import os
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import AgentState
from jevu.game_state import GameState
from jevu.pygame_view import (
    AGENT_COLORS,
    BLANK_COLOR,
    TREE_COLOR,
    build_summary_lines,
    run_pygame,
)
from jevu.world import WorldConfig
from jevu.rules import HUNGER_LOSS_PER_TURN, MIN_HUNGER


def select_actions(_: AgentState) -> TurnActions:
    return TurnActions(
        interact=InteractAction.HARVEST,
        explore=ExploreAction.UP,
    )


class PygameViewTests(unittest.TestCase):
    def test_palette_uses_requested_distinct_colors(self) -> None:
        self.assertGreater(BLANK_COLOR[1], BLANK_COLOR[0])
        self.assertGreater(TREE_COLOR[0], TREE_COLOR[1])
        self.assertEqual(len(AGENT_COLORS), len(set(AGENT_COLORS)))

    def test_game_can_render_without_a_visible_display(self) -> None:
        game_state = GameState.create(
            WorldConfig(width=3, height=2, seed=42),
            agent_count=2,
        )

        with redirect_stdout(StringIO()):
            final_state = run_pygame(
                game_state,
                max_turns=1,
                action_selector=select_actions,
                turns_per_second=1000,
                log_directory=None,
                hold_open=False,
            )

        self.assertEqual(final_state.turn, 1)

    def test_summary_reports_when_all_agents_die(self) -> None:
        initial_state = GameState.create(
            WorldConfig(width=3, height=2, seed=42),
            agent_count=1,
        )
        initial_state = replace(
            initial_state,
            agents=(
                replace(
                    initial_state.agents[0],
                    hunger=MIN_HUNGER + HUNGER_LOSS_PER_TURN,
                ),
            ),
        )

        with redirect_stdout(StringIO()):
            final_state = initial_state.run(
                max_turns=10,
                log_directory=None,
                action_selector=select_actions,
            )

        self.assertEqual(
            build_summary_lines(initial_state, final_state),
            ("All agents died", "A1 survived 1 turn"),
        )

    def test_summary_reports_survivors_at_turn_limit(self) -> None:
        initial_state = GameState.create(
            WorldConfig(width=3, height=2, seed=42),
            agent_count=1,
        )

        with redirect_stdout(StringIO()):
            final_state = initial_state.run(
                max_turns=2,
                log_directory=None,
                action_selector=select_actions,
            )

        self.assertEqual(
            build_summary_lines(initial_state, final_state),
            ("Maximum turns reached", "A1 survived 2 turns (alive)"),
        )


if __name__ == "__main__":
    unittest.main()
