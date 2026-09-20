"""Versioned metadata shared by RL training and inference."""

from __future__ import annotations

import json
from pathlib import Path

from jevu.rules import (
    FOOD_EAT_COST,
    FOOD_HARVEST_AMOUNT,
    FOOD_HUNGER_RESTORE,
    FRUIT_TREE_COOLDOWN,
    HUNGER_LOSS_PER_TURN,
    INITIAL_HUNGER,
    MAX_HUNGER,
    MIN_HUNGER,
)

BUNDLE_VERSION = 1
MODEL_FILENAME = "final_model.zip"
METADATA_FILENAME = "metadata.json"


def game_rules_metadata() -> dict[str, int]:
    return {
        "min_hunger": MIN_HUNGER,
        "max_hunger": MAX_HUNGER,
        "initial_hunger": INITIAL_HUNGER,
        "hunger_loss_per_turn": HUNGER_LOSS_PER_TURN,
        "food_harvest_amount": FOOD_HARVEST_AMOUNT,
        "fruit_tree_cooldown": FRUIT_TREE_COOLDOWN,
        "food_eat_cost": FOOD_EAT_COST,
        "food_hunger_restore": FOOD_HUNGER_RESTORE,
    }


def write_json(path: Path, value: object) -> None:
    """Atomically replace a JSON artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
