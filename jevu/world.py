"""Deterministic grid-world generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from jevu.agent import Agent


class TileType(StrEnum):
    """Contents of one world tile."""

    BLANK = "blank"
    FRUIT_TREE = "fruit_tree"


@dataclass(frozen=True, slots=True)
class Position:
    """Zero-based coordinates in the world."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Parameters used to generate a world."""

    width: int = 10
    height: int = 10
    fruit_tree_density: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World width and height must be positive")
        if not 0.0 <= self.fruit_tree_density <= 1.0:
            raise ValueError("Fruit tree density must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class World:
    """A generated world stored as rows of tiles."""

    config: WorldConfig
    tiles: tuple[tuple[TileType, ...], ...]

    @classmethod
    def generate(cls, config: WorldConfig) -> World:
        """Generate a deterministic world from a configuration."""

        random = Random(config.seed)
        tiles = tuple(
            tuple(
                TileType.FRUIT_TREE
                if random.random() < config.fruit_tree_density
                else TileType.BLANK
                for _ in range(config.width)
            )
            for _ in range(config.height)
        )
        return cls(config=config, tiles=tiles)

    def contains(self, position: Position) -> bool:
        """Return whether a position lies inside the world."""

        return (
            0 <= position.x < self.config.width
            and 0 <= position.y < self.config.height
        )

    def tile_at(self, position: Position) -> TileType:
        """Return the tile at a position."""

        if not self.contains(position):
            raise IndexError(f"Position is outside the world: {position}")
        return self.tiles[position.y][position.x]

    def adjacent_tiles(self, position: Position) -> dict[str, TileType]:
        """Return visible orthogonal neighbors, excluding world boundaries."""

        return {
            direction: self.tile_at(neighbor)
            for direction, neighbor in self.adjacent_positions(position).items()
        }

    def adjacent_positions(self, position: Position) -> dict[str, Position]:
        """Return orthogonal neighbor positions, excluding world boundaries."""

        if not self.contains(position):
            raise IndexError(f"Position is outside the world: {position}")

        candidates = {
            "up": Position(position.x, position.y - 1),
            "down": Position(position.x, position.y + 1),
            "left": Position(position.x - 1, position.y),
            "right": Position(position.x + 1, position.y),
        }
        return {
            direction: neighbor
            for direction, neighbor in candidates.items()
            if self.contains(neighbor)
        }

    def render_ascii(self, agents: Iterable[Agent] = ()) -> str:
        """Render the world for quick terminal inspection."""

        symbols = {TileType.BLANK: ".", TileType.FRUIT_TREE: "T"}
        agent_labels = {agent.position: agent.id for agent in agents}
        return "\n".join(
            " ".join(
                agent_labels.get(Position(x, y), symbols[tile])
                for x, tile in enumerate(row)
            )
            for y, row in enumerate(self.tiles)
        )
