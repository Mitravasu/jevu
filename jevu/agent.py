"""Agent identity, placement, and observable state."""

from __future__ import annotations

from collections.abc import Mapping
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
    current_tile_cooldown: int
    current_tile_claim: str | None
    adjacent_tiles: dict[str, TileType]
    adjacent_tile_cooldowns: dict[str, int]
    adjacent_tile_claims: dict[str, str | None]
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

    def state(
        self,
        world: World,
        fruit_tree_cooldowns: Mapping[Position, int] | None = None,
        tile_claims: Mapping[Position, int] | None = None,
    ) -> AgentState:
        """Return the agent's current state and local observation."""

        cooldowns = fruit_tree_cooldowns or {}
        claims = tile_claims or {}
        adjacent_positions = world.adjacent_positions(self.position)

        return AgentState(
            id=self.id,
            position=self.position,
            current_tile=world.tile_at(self.position),
            current_tile_cooldown=cooldowns.get(self.position, 0),
            current_tile_claim=(
                f"A{claims[self.position]}" if self.position in claims else None
            ),
            adjacent_tiles={
                direction: world.tile_at(position)
                for direction, position in adjacent_positions.items()
            },
            adjacent_tile_cooldowns={
                direction: cooldowns.get(position, 0)
                for direction, position in adjacent_positions.items()
            },
            adjacent_tile_claims={
                direction: f"A{claims[position]}" if position in claims else None
                for direction, position in adjacent_positions.items()
            },
            hunger=self.hunger,
            inventory=self.inventory,
        )
