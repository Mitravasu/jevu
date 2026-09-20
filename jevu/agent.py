"""Agent identity, placement, and observable state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from jevu.inventory import InventoryState
from jevu.personalities import Personality
from jevu.rules import INITIAL_HUNGER, MAX_HUNGER, MIN_HUNGER
from jevu.world import Position, TileType, World


@dataclass(frozen=True, slots=True)
class TileObservation:
    """One visible tile and its dynamic simulation state."""

    position: Position
    tile: TileType
    cooldown: int
    claim: str | None


@dataclass(frozen=True, slots=True)
class ClaimedTileObservation:
    """An owned tile and the tiles visible from it."""

    tile: TileObservation
    adjacent_tiles: dict[str, TileObservation]


@dataclass(frozen=True, slots=True)
class AgentState:
    """A snapshot of everything currently observable about an agent."""

    id: str
    world_width: int
    world_height: int
    position: Position
    current_tile: TileType
    current_tile_cooldown: int
    current_tile_claim: str | None
    adjacent_tiles: dict[str, TileType]
    adjacent_tile_cooldowns: dict[str, int]
    adjacent_tile_claims: dict[str, str | None]
    adjacent_agents: dict[str, str | None]
    claimed_territory: tuple[ClaimedTileObservation, ...]
    hunger: int
    inventory: InventoryState
    personality: Personality


@dataclass(frozen=True, slots=True)
class Agent:
    """An agent placed in the world."""

    number: int
    position: Position
    hunger: int = INITIAL_HUNGER
    inventory: InventoryState = field(default_factory=InventoryState)
    personality: Personality = Personality.ADAPTIVE_SURVIVOR

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
        agent_positions: Mapping[Position, int] | None = None,
    ) -> AgentState:
        """Return the agent's current state and local observation."""

        cooldowns = fruit_tree_cooldowns or {}
        claims = tile_claims or {}
        visible_agents = agent_positions or {}
        adjacent_positions = world.adjacent_positions(self.position)

        def observe(position: Position) -> TileObservation:
            return TileObservation(
                position=position,
                tile=world.tile_at(position),
                cooldown=cooldowns.get(position, 0),
                claim=f"A{claims[position]}" if position in claims else None,
            )

        claimed_territory = tuple(
            ClaimedTileObservation(
                tile=observe(position),
                adjacent_tiles={
                    direction: observe(adjacent_position)
                    for direction, adjacent_position in world.adjacent_positions(
                        position
                    ).items()
                },
            )
            for position, owner_number in sorted(
                claims.items(),
                key=lambda item: (item[0].y, item[0].x),
            )
            if owner_number == self.number
        )

        return AgentState(
            id=self.id,
            world_width=world.config.width,
            world_height=world.config.height,
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
            adjacent_agents={
                direction: (
                    f"A{visible_agents[position]}"
                    if position in visible_agents
                    and visible_agents[position] != self.number
                    else None
                )
                for direction, position in adjacent_positions.items()
            },
            claimed_territory=claimed_territory,
            hunger=self.hunger,
            inventory=self.inventory,
            personality=self.personality,
        )
