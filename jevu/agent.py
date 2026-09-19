"""Agent identity, placement, and observable state."""

from __future__ import annotations

from dataclasses import dataclass

from jevu.world import Position, TileType, World


@dataclass(frozen=True, slots=True)
class AgentState:
    """A snapshot of everything currently observable about an agent."""

    id: str
    position: Position
    current_tile: TileType
    adjacent_tiles: dict[str, TileType]
    hunger: int
    carried_fruit: int


@dataclass(frozen=True, slots=True)
class Agent:
    """An agent placed in the world."""

    number: int
    position: Position
    hunger: int = 10
    carried_fruit: int = 0

    def __post_init__(self) -> None:
        if self.number <= 0:
            raise ValueError("Agent number must be positive")
        if not 0 <= self.hunger <= 10:
            raise ValueError("Agent hunger must be between 0 and 10")
        if self.carried_fruit < 0:
            raise ValueError("Carried fruit cannot be negative")

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
            carried_fruit=self.carried_fruit,
        )
