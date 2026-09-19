"""Complete state and initial setup for a JevU game."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from jevu.agent import Agent
from jevu.world import Position, World, WorldConfig


@dataclass(frozen=True, slots=True)
class GameState:
    """The world and all agents currently in it."""

    world: World
    agents: tuple[Agent, ...]

    @classmethod
    def create(cls, world_config: WorldConfig, agent_count: int = 1) -> GameState:
        """Generate a world and deterministically place its agents."""

        world = World.generate(world_config)
        agents = cls._place_agents(world, agent_count)
        return cls(world=world, agents=agents)

    @staticmethod
    def _place_agents(world: World, count: int) -> tuple[Agent, ...]:
        available_positions = [
            Position(x, y)
            for y in range(world.config.height)
            for x in range(world.config.width)
        ]
        if count < 0:
            raise ValueError("Agent count cannot be negative")
        if count > len(available_positions):
            raise ValueError("Agent count cannot exceed the number of world tiles")

        random = Random(world.config.seed)
        positions = random.sample(available_positions, count)
        return tuple(
            Agent(number=number, position=position)
            for number, position in enumerate(positions, start=1)
        )

    def render_ascii(self) -> str:
        """Render the world with all agents overlaid."""

        return self.world.render_ascii(self.agents)

    def run(self) -> None:
        """Run the game from its current state."""

        print(self.render_ascii())
