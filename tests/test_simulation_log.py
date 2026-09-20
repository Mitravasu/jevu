import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import AgentState
from jevu.decision import ChoiceTrace, TurnDecision
from jevu.game_state import GameState
from jevu.world import WorldConfig


def select_actions(_: AgentState) -> TurnDecision:
    actions = TurnActions(
        interact=InteractAction.HARVEST,
        explore=ExploreAction.UP,
    )
    return TurnDecision(
        actions=actions,
        interact=ChoiceTrace(
            choice="harvest",
            probabilities={"harvest": 0.8, "eat": 0.1, "claim": 0.1},
            confidence=0.7,
        ),
        explore=ChoiceTrace(
            choice="up",
            probabilities={"up": 0.7, "down": 0.1, "left": 0.1, "right": 0.1},
            confidence=0.6,
        ),
    )


class SimulationLogTests(unittest.TestCase):
    def test_run_writes_structured_log_named_with_parameters(self) -> None:
        game_state = GameState.create(
            WorldConfig(
                width=4,
                height=3,
                fruit_tree_density=0.25,
                seed=42,
            ),
            agent_count=2,
        )

        with TemporaryDirectory() as temporary_directory:
            with redirect_stdout(StringIO()):
                game_state.run(
                    max_turns=2,
                    log_directory=temporary_directory,
                    action_selector=select_actions,
                )

            log_files = list(Path(temporary_directory).glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            self.assertTrue(
                log_files[0].name.startswith(
                    "w4_h3_trees0p25_seed42_agents2_turns2_"
                )
            )

            records = [
                json.loads(line)
                for line in log_files[0].read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[0]["type"], "simulation_start")
        self.assertEqual(
            records[0]["agents"][0]["personality"],
            game_state.agents[0].personality.value,
        )
        self.assertEqual(records[-1]["type"], "simulation_end")
        self.assertEqual(
            [record["type"] for record in records[1:-1]],
            ["action", "action", "action", "action"],
        )
        self.assertEqual(records[1]["start"]["food"], 0)
        self.assertEqual(records[1]["automatic_harvest_food"], 0)
        self.assertIn("position", records[1]["end"])
        self.assertEqual(
            records[1]["decision"]["interact"]["probabilities"],
            {"harvest": 0.8, "eat": 0.1, "claim": 0.1},
        )
        self.assertEqual(records[1]["decision"]["explore"]["confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
