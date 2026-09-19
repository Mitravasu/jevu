import os
import unittest
from contextlib import redirect_stdout
from io import StringIO

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from jevu.game_state import GameState
from jevu.pygame_view import AGENT_COLORS, BLANK_COLOR, TREE_COLOR, run_pygame
from jevu.world import WorldConfig


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
                turns_per_second=1000,
                log_directory=None,
                hold_open=False,
            )

        self.assertEqual(final_state.turn, 1)


if __name__ == "__main__":
    unittest.main()
