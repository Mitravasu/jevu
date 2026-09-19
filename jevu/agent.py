"""Agent identity, placement, and observable state."""

from __future__ import annotations

from dataclasses import dataclass, field

from jevu.inventory import InventoryState
from jevu.rules import INITIAL_HUNGER, MAX_HUNGER, MIN_HUNGER
from jevu.world import Position, TileType, World


@dataclass(frozen=True, slots=True)
class AgentState:
    """A snapshot of everything currently observable about an agent."""

    id: str
    position: Position
    current_tile: TileType
    adjacent_tiles: dict[str, TileType]
    hunger: int
    inventory: InventoryState


@dataclass(frozen=True, slots=True)
class Agent:
    """An agent placed in the world."""

    number: int
    position: Position
    hunger: int = INITIAL_HUNGER
    inventory: InventoryState = field(default_factory=InventoryState)

    def __post_init__(self) -> None:
        if self.number <= 0:
            raise ValueError("Agent number must be positive")
        if not MIN_HUNGER <= self.hunger <= MAX_HUNGER:
            raise ValueError(
                f"Agent hunger must be between {MIN_HUNGER} and {MAX_HUNGER}"
            )

    @property
    def id(self) -> str:
        return f"A{self.number}"

    def state(self, world: World) -> AgentState:
        """Return the agent's current state and local observation."""

        return AgentState(
            id=self.id,
            position=self.position,
            current_tile=world.tile_at(self.position),
            adjacent_tiles=world.adjacent_tiles(self.position),
            hunger=self.hunger,
            inventory=self.inventory,
        )
