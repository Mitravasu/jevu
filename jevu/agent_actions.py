"""Extract one agent's actions from a simulation log as sentences."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _position_text(position: dict[str, int]) -> str:
    return f"({position['x']}, {position['y']})"


def _interaction_text(record: dict[str, Any], tile: str) -> str:
    action = record["actions"]["interact"]
    tile_name = tile.replace("_", " ")

    if action == "harvest":
        if record["end"]["food"] > record["start"]["food"]:
            return f"harvested food from the {tile_name} tile"
        return f"tried to harvest the {tile_name} tile"
    if action == "eat":
        if record["end"]["food"] < record["start"]["food"]:
            return f"ate food on the {tile_name} tile"
        return f"tried to eat on the {tile_name} tile"
    if action == "claim":
        return f"tried to claim the {tile_name} tile"
    if action == "none":
        return f"did nothing on the {tile_name} tile"
    return f"selected {action} on the {tile_name} tile"


def _action_sentence(
    record: dict[str, Any],
    world: list[list[str]],
    personality: str | None = None,
) -> str:
    start = record["start"]["position"]
    end = record["end"]["position"]
    tile = world[start["y"]][start["x"]]
    interaction = _interaction_text(record, tile)
    direction = record["actions"]["explore"]

    if direction == "stay":
        movement = f"then stayed at {_position_text(start)}"
    elif start == end:
        movement = (
            f"then tried to move {direction} but stayed at {_position_text(start)}"
        )
    else:
        movement = f"then moved {direction} to {_position_text(end)}"

    automatic_harvest_food = record.get("automatic_harvest_food", 0)
    automatic_harvest = ""
    if automatic_harvest_food:
        automatic_harvest = (
            f" It automatically received {automatic_harvest_food} food from "
            "claimed fruit trees."
        )

    agent_label = f"Agent {record['agent_id']}"
    if personality is not None:
        agent_label += f" ({personality.replace('_', ' ')})"
    sentence = (
        f"Turn {record['turn']}: {agent_label} {interaction} at "
        f"{_position_text(start)}, {movement}.{automatic_harvest}"
    )
    decision = record.get("decision")
    if decision is None:
        return sentence

    traces = []
    for name in ("interact", "explore"):
        trace = decision[name]
        probabilities = ", ".join(
            f"{choice} {probability:.1%}"
            for choice, probability in trace["probabilities"].items()
        )
        traces.append(
            f"{name}: {probabilities}; confidence {trace['confidence']:.1%}"
        )
    return f"{sentence} Jev probabilities — {' | '.join(traces)}."


def extract_agent_actions(
    log_path: str | Path,
    agent_id: str,
) -> tuple[str, ...]:
    """Return natural-language action sentences for one agent in a JSONL log."""

    records = [
        json.loads(line)
        for line in Path(log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("type") != "simulation_start":
        raise ValueError("Log must begin with a simulation_start record")

    world = records[0]["world"]
    personalities = {
        agent["id"]: agent.get("personality")
        for agent in records[0].get("agents", [])
    }
    return tuple(
        _action_sentence(record, world, personalities.get(agent_id))
        for record in records
        if record.get("type") == "action" and record.get("agent_id") == agent_id
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print one agent's simulation actions as sentences",
    )
    parser.add_argument("log", type=Path, help="Simulation JSONL log")
    parser.add_argument("agent_id", help="Agent id, for example A1")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sentences = extract_agent_actions(args.log, args.agent_id)
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if not sentences:
        parser.error(f"No actions found for agent {args.agent_id}")
    print("\n".join(sentences))


if __name__ == "__main__":
    main()
