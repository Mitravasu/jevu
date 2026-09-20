import argparse
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import Mock, patch

import main


class MainTests(unittest.TestCase):
    def test_rl_backend_requires_bundle(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--agent-backend", "rl"])

    def test_jev_backend_rejects_bundle(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--bundle", "model-bundle"])

    @patch("jevu.rl.selector.RLActionSelector.from_bundle")
    @patch("main.GameState.create")
    def test_rl_bundle_replaces_jev_selector(self, create_game, load_selector) -> None:
        game_state = Mock()
        selector = Mock()
        create_game.return_value = game_state
        load_selector.return_value = selector
        args = argparse.Namespace(
            agent_backend="rl",
            bundle="model-bundle",
            device="mps",
            width=10,
            height=10,
            tree_density=0.15,
            seed=42,
            agents=1,
            max_turns=20,
            model="unused",
            pygame=False,
            turns_per_second=2.0,
        )

        with patch("main.parse_args", return_value=args):
            main.main()

        load_selector.assert_called_once_with("model-bundle", device="mps")
        game_state.run.assert_called_once_with(
            max_turns=20,
            action_selector=selector,
        )


if __name__ == "__main__":
    unittest.main()
