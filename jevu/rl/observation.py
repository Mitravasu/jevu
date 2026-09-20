"""Agent-centered observable-state encoding for RL policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from jevu.agent import RECENT_POSITION_WINDOW, AgentState
from jevu.rules import FOOD_HARVEST_AMOUNT, FRUIT_TREE_COOLDOWN, MAX_HUNGER
from jevu.world import Position, TileType

OBSERVATION_SCHEMA_VERSION = 3
WINDOW_SIZE = 3
WINDOW_CENTER = 1
CHANNEL_NAMES = (
    "known",
    "boundary",
    "fruit_tree",
    "tree_cooldown",
    "owned_by_self",
    "claimed_by_other",
    "other_agent",
    "current_position",
    "recent_visit_count",
    "unclaimed_frontier",
)
_OFFSETS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """A world-size-independent, one-tile egocentric observation schema."""

    max_turns: int

    @property
    def max_food(self) -> int:
        return max(1, self.max_turns * FOOD_HARVEST_AMOUNT)

    def space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "map": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(CHANNEL_NAMES), WINDOW_SIZE, WINDOW_SIZE),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

    def metadata(self) -> dict[str, object]:
        return {
            "version": OBSERVATION_SCHEMA_VERSION,
            "view": "egocentric_orthogonal_one_tile",
            "window_size": WINDOW_SIZE,
            "max_turns": self.max_turns,
            "max_food": self.max_food,
            "recent_position_window": RECENT_POSITION_WINDOW,
            "channels": list(CHANNEL_NAMES),
            "scalars": ["hunger", "food"],
        }


def encode_observation(
    state: AgentState,
    spec: ObservationSpec,
) -> dict[str, np.ndarray]:
    """Encode only the current and orthogonally adjacent visible tiles."""

    grid = np.zeros(
        (len(CHANNEL_NAMES), WINDOW_SIZE, WINDOW_SIZE),
        dtype=np.float32,
    )
    owned_positions = {
        claimed.tile.position for claimed in state.claimed_territory
    }
    recent_visit_counts = {
        position: state.recent_positions.count(position)
        for position in set(state.recent_positions)
    }

    def is_unclaimed_frontier(position: Position) -> bool:
        # Kept as a local feature: only visible candidate tiles are encoded.
        return any(
            Position(position.x + dx, position.y + dy) in owned_positions
            for dx, dy in _OFFSETS.values()
        )

    def add_visible(
        row: int,
        column: int,
        position: Position,
        tile: TileType,
        cooldown: int,
        claim: str | None,
        other_agent: str | None,
    ) -> None:
        grid[0, row, column] = 1.0
        grid[2, row, column] = float(tile is TileType.FRUIT_TREE)
        grid[3, row, column] = min(
            1.0, cooldown / max(1, FRUIT_TREE_COOLDOWN)
        )
        grid[4, row, column] = float(claim == state.id)
        grid[5, row, column] = float(claim is not None and claim != state.id)
        grid[6, row, column] = float(other_agent is not None)
        grid[8, row, column] = min(
            1.0,
            recent_visit_counts.get(position, 0) / RECENT_POSITION_WINDOW,
        )
        grid[9, row, column] = float(
            claim is None and is_unclaimed_frontier(position)
        )

    add_visible(
        WINDOW_CENTER,
        WINDOW_CENTER,
        state.position,
        state.current_tile,
        state.current_tile_cooldown,
        state.current_tile_claim,
        None,
    )
    grid[7, WINDOW_CENTER, WINDOW_CENTER] = 1.0

    for direction, (dx, dy) in _OFFSETS.items():
        row = WINDOW_CENTER + dy
        column = WINDOW_CENTER + dx
        if direction not in state.adjacent_tiles:
            grid[1, row, column] = 1.0
            continue
        add_visible(
            row,
            column,
            Position(
                state.position.x + dx,
                state.position.y + dy,
            ),
            state.adjacent_tiles[direction],
            state.adjacent_tile_cooldowns[direction],
            state.adjacent_tile_claims[direction],
            state.adjacent_agents[direction],
        )

    scalars = np.asarray(
        [
            state.hunger / max(1, MAX_HUNGER),
            min(state.inventory.food, spec.max_food) / spec.max_food,
        ],
        dtype=np.float32,
    )
    return {"map": grid, "scalars": scalars}


def empty_observation(spec: ObservationSpec) -> dict[str, np.ndarray]:
    del spec
    return {
        "map": np.zeros(
            (len(CHANNEL_NAMES), WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32
        ),
        "scalars": np.zeros(2, dtype=np.float32),
    }
