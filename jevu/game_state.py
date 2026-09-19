"""Complete state and initial setup for a JevU game."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

from jevu.actions import ExploreAction, InteractAction, TurnActions, take_turn
from jevu.agent import Agent
from jevu.simulation_log import write_simulation_log
from jevu.world import Position, World, WorldConfig


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    """Actions selected by one agent during one turn."""

    turn: int
    agent_id: str
    actions: TurnActions
    start_position: Position
    end_position: Position
    start_hunger: int
    end_hunger: int
    start_food: int
    end_food: int

    def __str__(self) -> str:
        return (
            f"Turn {self.turn} - {self.agent_id}: "
            f"position=({self.start_position.x}, {self.start_position.y}) -> "
            f"({self.end_position.x}, {self.end_position.y}), "
            f"hunger={self.start_hunger} -> {self.end_hunger}, "
            f"food={self.start_food} -> {self.end_food}, "
            f"interact={self.actions.interact.value}, "
            f"explore={self.actions.explore.value}"
        )


@dataclass(frozen=True, slots=True)
class GameState:
    """The world and all agents currently in it."""

    world: World
    agents: tuple[Agent, ...]
    turn: int = 0
    action_log: tuple[ActionLogEntry, ...] = ()

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

    def render_agent_states(self) -> str:
        """Render the position, hunger, and inventory of every living agent."""

        if not self.agents:
            return "(none)"
        return "\n".join(
            f"{agent.id}: (x={agent.position.x}, y={agent.position.y}), "
            f"hunger={agent.hunger}, food={agent.inventory.food}"
            for agent in self.agents
        )

    def _run_turn(self) -> GameState:
        """Run one deterministic random turn for every living agent."""

        turn = self.turn + 1
        random = Random(f"jevu:{self.world.config.seed}:turn:{turn}")
        occupied_positions = {agent.position for agent in self.agents}
        updated_agents: list[Agent] = []
        new_log_entries: list[ActionLogEntry] = []

        for agent in self.agents:
            occupied_positions.remove(agent.position)
            actions = TurnActions(
                interact=random.choice(tuple(InteractAction)),
                explore=random.choice(tuple(ExploreAction)),
            )
            updated_agent = take_turn(agent, self.world, actions)

            if updated_agent.position in occupied_positions:
                updated_agent = replace(updated_agent, position=agent.position)

            updated_agent = replace(
                updated_agent,
                hunger=max(0, updated_agent.hunger - 1),
            )
            new_log_entries.append(
                ActionLogEntry(
                    turn=turn,
                    agent_id=agent.id,
                    actions=actions,
                    start_position=agent.position,
                    end_position=updated_agent.position,
                    start_hunger=agent.hunger,
                    end_hunger=updated_agent.hunger,
                    start_food=agent.inventory.food,
                    end_food=updated_agent.inventory.food,
                )
            )

            if updated_agent.hunger > 0:
                updated_agents.append(updated_agent)
                occupied_positions.add(updated_agent.position)

        return replace(
            self,
            agents=tuple(updated_agents),
            turn=turn,
            action_log=self.action_log + tuple(new_log_entries),
        )

    def run(
        self,
        max_turns: int = 0,
        log_directory: str | Path | None = "logs",
    ) -> GameState:
        """Run a simulation and return its final state."""

        if max_turns < 0:
            raise ValueError("Maximum turns cannot be negative")

        game_state = self
        initial_log_length = len(self.action_log)
        for _ in range(max_turns):
            game_state = game_state._run_turn()

        for entry in game_state.action_log[initial_log_length:]:
            print(entry)

        print("\nStart state:")
        print(self.render_ascii())
        print("Agent states:")
        print(self.render_agent_states())
        print("\nFinal state:")
        print(game_state.render_ascii())
        print("Agent states:")
        print(game_state.render_agent_states())
        if log_directory is not None:
            log_path = write_simulation_log(
                initial_state=self,
                final_state=game_state,
                action_entries=game_state.action_log[initial_log_length:],
                max_turns=max_turns,
                directory=log_directory,
            )
            print(f"\nLog file: {log_path}")
        return game_state
