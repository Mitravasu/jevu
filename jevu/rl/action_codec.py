"""Stable conversion between model actions and JevU actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import AgentState
from jevu.rules import FOOD_EAT_COST, MAX_HUNGER
from jevu.world import TileType

INTERACT_ACTIONS = (
    InteractAction.HARVEST,
    InteractAction.EAT,
    InteractAction.CLAIM,
    InteractAction.NONE,
)
EXPLORE_ACTIONS = (
    ExploreAction.UP,
    ExploreAction.DOWN,
    ExploreAction.LEFT,
    ExploreAction.RIGHT,
    ExploreAction.STAY,
)
ACTION_SHAPE = (len(INTERACT_ACTIONS), len(EXPLORE_ACTIONS))


def encode_actions(actions: TurnActions) -> np.ndarray:
    return np.asarray(
        [
            INTERACT_ACTIONS.index(actions.interact),
            EXPLORE_ACTIONS.index(actions.explore),
        ],
        dtype=np.int64,
    )


def decode_actions(action: Sequence[int] | np.ndarray) -> TurnActions:
    values = np.asarray(action, dtype=np.int64).reshape(-1)
    if values.size != 2:
        raise ValueError(
            "An RL action must contain interaction and exploration indices"
        )
    interact_index, explore_index = (int(values[0]), int(values[1]))
    if not 0 <= interact_index < len(INTERACT_ACTIONS) or not 0 <= explore_index < len(
        EXPLORE_ACTIONS
    ):
        raise ValueError(f"RL action is outside the action space: {values.tolist()}")
    try:
        return TurnActions(
            interact=INTERACT_ACTIONS[interact_index],
            explore=EXPLORE_ACTIONS[explore_index],
        )
    except IndexError as exc:
        raise ValueError(
            f"RL action is outside the action space: {values.tolist()}"
        ) from exc


def action_masks(state: AgentState, width: int, height: int) -> np.ndarray:
    """Return concatenated masks expected by MaskablePPO for MultiDiscrete."""

    interact = np.asarray(
        [
            state.current_tile is TileType.FRUIT_TREE
            and state.current_tile_cooldown == 0,
            state.inventory.food >= FOOD_EAT_COST and state.hunger < MAX_HUNGER,
            state.current_tile_claim is None,
            True,
        ],
        dtype=np.bool_,
    )
    explore = np.asarray(
        [
            state.position.y > 0,
            state.position.y + 1 < height,
            state.position.x > 0,
            state.position.x + 1 < width,
            True,
        ],
        dtype=np.bool_,
    )
    return np.concatenate((interact, explore))


def action_mapping() -> dict[str, list[str]]:
    return {
        "interact": [action.value for action in INTERACT_ACTIONS],
        "explore": [action.value for action in EXPLORE_ACTIONS],
    }
