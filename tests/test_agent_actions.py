import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jevu.agent_actions import extract_agent_actions


class AgentActionExtractionTests(unittest.TestCase):
    def test_extracts_only_requested_agent_as_sentences(self) -> None:
        records = [
            {
                "type": "simulation_start",
                "world": [["blank", "fruit_tree"]],
            },
            {
                "type": "action",
                "turn": 1,
                "agent_id": "A1",
                "actions": {"interact": "harvest", "explore": "left"},
                "start": {
                    "position": {"x": 1, "y": 0},
                    "hunger": 10,
                    "food": 0,
                },
                "end": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 9,
                    "food": 1,
                },
            },
            {
                "type": "action",
                "turn": 1,
                "agent_id": "A2",
                "actions": {"interact": "claim", "explore": "right"},
                "start": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 10,
                    "food": 0,
                },
                "end": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 9,
                    "food": 0,
                },
            },
        ]

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "simulation.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            sentences = extract_agent_actions(log_path, "A1")

        self.assertEqual(
            sentences,
            (
                "Turn 1: Agent A1 harvested food from the fruit tree tile at "
                "(1, 0), then moved left to (0, 0).",
            ),
        )

    def test_reports_when_movement_did_not_change_position(self) -> None:
        records = [
            {"type": "simulation_start", "world": [["blank"]]},
            {
                "type": "action",
                "turn": 2,
                "agent_id": "A1",
                "actions": {"interact": "claim", "explore": "up"},
                "start": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 9,
                    "food": 0,
                },
                "end": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 8,
                    "food": 0,
                },
            },
        ]

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "simulation.jsonl"
            log_path.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[1]))

            sentences = extract_agent_actions(log_path, "A1")

        self.assertEqual(
            sentences[0],
            "Turn 2: Agent A1 tried to claim the blank tile at (0, 0), then "
            "tried to move up but stayed at (0, 0).",
        )

    def test_includes_jev_probability_trace_when_available(self) -> None:
        records = [
            {"type": "simulation_start", "world": [["blank"]]},
            {
                "type": "action",
                "turn": 1,
                "agent_id": "A1",
                "actions": {"interact": "claim", "explore": "up"},
                "start": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 10,
                    "food": 0,
                },
                "end": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 9,
                    "food": 0,
                },
                "automatic_harvest_food": 1,
                "decision": {
                    "interact": {
                        "choice": "claim",
                        "probabilities": {"harvest": 0.1, "eat": 0.2, "claim": 0.7},
                        "confidence": 0.6,
                    },
                    "explore": {
                        "choice": "up",
                        "probabilities": {"up": 0.7, "down": 0.3},
                        "confidence": 0.5,
                    },
                },
            },
        ]

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "simulation.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            sentence = extract_agent_actions(log_path, "A1")[0]

        self.assertIn("harvest 10.0%, eat 20.0%, claim 70.0%", sentence)
        self.assertIn("confidence 60.0%", sentence)
        self.assertIn("up 70.0%, down 30.0%", sentence)
        self.assertIn("automatically received 1 food", sentence)

    def test_describes_none_and_stay_actions_naturally(self) -> None:
        records = [
            {"type": "simulation_start", "world": [["blank"]]},
            {
                "type": "action",
                "turn": 1,
                "agent_id": "A1",
                "actions": {"interact": "none", "explore": "stay"},
                "start": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 10,
                    "food": 0,
                },
                "end": {
                    "position": {"x": 0, "y": 0},
                    "hunger": 9,
                    "food": 0,
                },
            },
        ]

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "simulation.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            sentence = extract_agent_actions(log_path, "A1")[0]

        self.assertEqual(
            sentence,
            "Turn 1: Agent A1 did nothing on the blank tile at (0, 0), then "
            "stayed at (0, 0).",
        )


if __name__ == "__main__":
    unittest.main()
