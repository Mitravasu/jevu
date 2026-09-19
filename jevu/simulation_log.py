"""Structured file logging for simulation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from jevu.rules import FRUIT_TREE_COOLDOWN

if TYPE_CHECKING:
    from jevu.agent import Agent
    from jevu.game_state import ActionLogEntry, GameState


def _agent_record(agent: Agent) -> dict[str, Any]:
    return {
        "id": agent.id,
        "position": {"x": agent.position.x, "y": agent.position.y},
        "hunger": agent.hunger,
        "inventory": {"food": agent.inventory.food},
    }


def _action_record(entry: ActionLogEntry) -> dict[str, Any]:
    return {
        "type": "action",
        "turn": entry.turn,
        "agent_id": entry.agent_id,
        "actions": {
            "interact": entry.actions.interact.value,
            "explore": entry.actions.explore.value,
        },
        "start": {
            "position": {
                "x": entry.start_position.x,
                "y": entry.start_position.y,
            },
            "hunger": entry.start_hunger,
            "food": entry.start_food,
        },
        "end": {
            "position": {
                "x": entry.end_position.x,
                "y": entry.end_position.y,
            },
            "hunger": entry.end_hunger,
            "food": entry.end_food,
        },
    }


def _density_label(density: float) -> str:
    return format(density, "g").replace(".", "p")


def write_simulation_log(
    initial_state: GameState,
    final_state: GameState,
    action_entries: Iterable[ActionLogEntry],
    max_turns: int,
    directory: str | Path = "logs",
    timestamp: datetime | None = None,
) -> Path:
    """Write one simulation as JSON Lines and return the created path."""

    timestamp = timestamp or datetime.now(timezone.utc)
    timestamp_label = timestamp.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    config = initial_state.world.config
    filename = (
        f"w{config.width}_h{config.height}_"
        f"trees{_density_label(config.fruit_tree_density)}_"
        f"seed{config.seed}_agents{len(initial_state.agents)}_"
        f"turns{max_turns}_{timestamp_label}.jsonl"
    )

    log_directory = Path(directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / filename

    records = [
        {
            "type": "simulation_start",
            "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat(),
            "parameters": {
                "width": config.width,
                "height": config.height,
                "fruit_tree_density": config.fruit_tree_density,
                "seed": config.seed,
                "agents": len(initial_state.agents),
                "max_turns": max_turns,
                "fruit_tree_cooldown": FRUIT_TREE_COOLDOWN,
            },
            "world": [
                [tile.value for tile in row] for row in initial_state.world.tiles
            ],
            "agents": [_agent_record(agent) for agent in initial_state.agents],
        },
        *(_action_record(entry) for entry in action_entries),
        {
            "type": "simulation_end",
            "turn": final_state.turn,
            "agents": [_agent_record(agent) for agent in final_state.agents],
            "fruit_tree_cooldowns": [
                {
                    "position": {"x": position.x, "y": position.y},
                    "turns_remaining": cooldown,
                }
                for position, cooldown in final_state.fruit_tree_cooldowns.items()
            ],
            "tile_claims": [
                {
                    "position": {"x": position.x, "y": position.y},
                    "agent_id": f"A{agent_number}",
                }
                for position, agent_number in final_state.tile_claims.items()
            ],
        },
    ]

    with log_path.open("x", encoding="utf-8") as log_file:
        for record in records:
            log_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    return log_path
