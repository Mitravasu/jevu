"""Observable-state encoding for RL policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from jevu.agent import AgentState, TileObservation
from jevu.rules import FOOD_HARVEST_AMOUNT, FRUIT_TREE_COOLDOWN, MAX_HUNGER
from jevu.world import Position, TileType

OBSERVATION_SCHEMA_VERSION = 1
CHANNEL_NAMES = (
    "known",
    "fruit_tree",
    "tree_cooldown",
    "owned_by_self",
    "claimed_by_other",
    "current_position",
)


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    width: int
    height: int
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
                    shape=(len(CHANNEL_NAMES), self.height, self.width),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

    def metadata(self) -> dict[str, object]:
        return {
            "version": OBSERVATION_SCHEMA_VERSION,
            "width": self.width,
            "height": self.height,
            "max_turns": self.max_turns,
            "max_food": self.max_food,
            "channels": list(CHANNEL_NAMES),
            "scalars": ["hunger", "food", "x", "y"],
        }


def encode_observation(
    state: AgentState,
    spec: ObservationSpec,
) -> dict[str, np.ndarray]:
    """Encode observable and remembered tiles without exposing hidden world state."""

    if state.world_width > spec.width or state.world_height > spec.height:
        raise ValueError(
            "Agent state world dimensions exceed the model observation schema: "
            f"state={state.world_width}x{state.world_height}, "
            f"model={spec.width}x{spec.height}"
        )

    grid = np.zeros(
        (len(CHANNEL_NAMES), spec.height, spec.width),
        dtype=np.float32,
    )

    def add_tile(tile: TileObservation) -> None:
        if not (
            0 <= tile.position.x < spec.width and 0 <= tile.position.y < spec.height
        ):
            raise ValueError(
                f"Observed tile is outside configured world: {tile.position}"
            )
        x, y = tile.position.x, tile.position.y
        grid[0, y, x] = 1.0
        grid[1, y, x] = float(tile.tile is TileType.FRUIT_TREE)
        grid[2, y, x] = min(1.0, tile.cooldown / max(1, FRUIT_TREE_COOLDOWN))
        grid[3, y, x] = float(tile.claim == state.id)
        grid[4, y, x] = float(tile.claim is not None and tile.claim != state.id)

    add_tile(
        TileObservation(
            position=state.position,
            tile=state.current_tile,
            cooldown=state.current_tile_cooldown,
            claim=state.current_tile_claim,
        )
    )
    offsets = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    for direction, tile in state.adjacent_tiles.items():
        dx, dy = offsets[direction]
        add_tile(
            TileObservation(
                position=Position(state.position.x + dx, state.position.y + dy),
                tile=tile,
                cooldown=state.adjacent_tile_cooldowns[direction],
                claim=state.adjacent_tile_claims[direction],
            )
        )
    for claimed in state.claimed_territory:
        add_tile(claimed.tile)
        for adjacent in claimed.adjacent_tiles.values():
            add_tile(adjacent)
    grid[5, state.position.y, state.position.x] = 1.0

    scalars = np.asarray(
        [
            state.hunger / max(1, MAX_HUNGER),
            min(state.inventory.food, spec.max_food) / spec.max_food,
            state.position.x / max(1, spec.width - 1),
            state.position.y / max(1, spec.height - 1),
        ],
        dtype=np.float32,
    )
    return {"map": grid, "scalars": scalars}


def empty_observation(spec: ObservationSpec) -> dict[str, np.ndarray]:
    return {
        "map": np.zeros(
            (len(CHANNEL_NAMES), spec.height, spec.width), dtype=np.float32
        ),
        "scalars": np.zeros(4, dtype=np.float32),
    }
