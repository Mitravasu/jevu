"""Complete state and initial setup for a JevU game."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random

from jevu.actions import InteractAction, TurnActions, claim_tile, take_turn
from jevu.agent import Agent, AgentState
from jevu.decision import ActionSelection, TurnDecision
from jevu.personalities import personality_for_agent
from jevu.rules import (
    FOOD_HARVEST_AMOUNT,
    FRUIT_TREE_COOLDOWN,
    HUNGER_LOSS_PER_TURN,
    MIN_HUNGER,
)
from jevu.simulation_log import write_simulation_log
from jevu.world import Position, TileType, World, WorldConfig


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
    automatic_harvest_food: int = 0
    decision: TurnDecision | None = None

    def __str__(self) -> str:
        summary = (
            f"Turn {self.turn} - {self.agent_id}: "
            f"position=({self.start_position.x}, {self.start_position.y}) -> "
            f"({self.end_position.x}, {self.end_position.y}), "
            f"hunger={self.start_hunger} -> {self.end_hunger}, "
            f"food={self.start_food} -> {self.end_food}, "
            f"interact={self.actions.interact.value}, "
            f"explore={self.actions.explore.value}"
        )
        if self.automatic_harvest_food:
            summary += f", automatic_harvest_food={self.automatic_harvest_food}"
        if self.decision is None:
            return summary
        interact_probability = self.decision.interact.probabilities[
            self.decision.interact.choice
        ]
        explore_probability = self.decision.explore.probabilities[
            self.decision.explore.choice
        ]
        return (
            f"{summary}, "
            f"interact_probability={interact_probability:.3f}, "
            f"interact_confidence={self.decision.interact.confidence:.3f}, "
            f"explore_probability={explore_probability:.3f}, "
            f"explore_confidence={self.decision.explore.confidence:.3f}"
        )


@dataclass(frozen=True, slots=True)
class GameState:
    """The world and all agents currently in it."""

    world: World
    agents: tuple[Agent, ...]
    turn: int = 0
    action_log: tuple[ActionLogEntry, ...] = ()
    fruit_tree_cooldowns: dict[Position, int] = field(default_factory=dict)
    tile_claims: dict[Position, int] = field(default_factory=dict)

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
            Agent(
                number=number,
                position=position,
                personality=personality_for_agent(number),
            )
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
            f"personality={agent.personality.value}, "
            f"hunger={agent.hunger}, food={agent.inventory.food}"
            for agent in self.agents
        )

    def _run_turn(
        self,
        action_selector: Callable[[AgentState], ActionSelection],
    ) -> GameState:
        """Run one selected turn for every living agent."""

        turn = self.turn + 1
        automatic_food: dict[int, int] = {}
        newly_harvested: set[Position] = set()
        living_agent_numbers = {agent.number for agent in self.agents}
        for position, owner_number in self.tile_claims.items():
            if owner_number not in living_agent_numbers:
                continue
            if self.world.tile_at(position) is not TileType.FRUIT_TREE:
                continue
            if self.fruit_tree_cooldowns.get(position, 0) > 0:
                continue
            automatic_food[owner_number] = (
                automatic_food.get(owner_number, 0) + FOOD_HARVEST_AMOUNT
            )
            newly_harvested.add(position)

        turn_agents = tuple(
            replace(
                agent,
                inventory=agent.inventory.add_food(automatic_food[agent.number]),
            )
            if agent.number in automatic_food
            else agent
            for agent in self.agents
        )
        occupied_positions = {agent.position for agent in turn_agents}
        updated_agents: list[Agent] = []
        new_log_entries: list[ActionLogEntry] = []
        tile_claims = dict(self.tile_claims)

        for agent in turn_agents:
            occupied_positions.remove(agent.position)
            active_cooldowns = {
                position: cooldown
                for position, cooldown in self.fruit_tree_cooldowns.items()
                if cooldown > 0
            }
            if FRUIT_TREE_COOLDOWN > 0:
                active_cooldowns.update(
                    {
                        position: FRUIT_TREE_COOLDOWN
                        for position in newly_harvested
                    }
                )
            selection = action_selector(
                agent.state(self.world, active_cooldowns, tile_claims)
            )
            decision = selection if isinstance(selection, TurnDecision) else None
            actions = decision.actions if decision is not None else selection
            if actions.interact is InteractAction.CLAIM:
                tile_claims = claim_tile(agent, self.world, tile_claims)
            updated_agent = take_turn(
                agent,
                self.world,
                actions,
                fruit_tree_available=active_cooldowns.get(agent.position, 0) == 0,
            )
            if (
                actions.interact is InteractAction.HARVEST
                and updated_agent.inventory.food > agent.inventory.food
            ):
                newly_harvested.add(agent.position)

            if updated_agent.position in occupied_positions:
                updated_agent = replace(updated_agent, position=agent.position)

            updated_agent = replace(
                updated_agent,
                hunger=max(
                    MIN_HUNGER,
                    updated_agent.hunger - HUNGER_LOSS_PER_TURN,
                ),
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
                    automatic_harvest_food=automatic_food.get(agent.number, 0),
                    decision=decision,
                )
            )

            if updated_agent.hunger > MIN_HUNGER:
                updated_agents.append(updated_agent)
                occupied_positions.add(updated_agent.position)

        remaining_cooldowns = {
            position: cooldown - 1
            for position, cooldown in self.fruit_tree_cooldowns.items()
            if cooldown > 1
        }
        if FRUIT_TREE_COOLDOWN > 0:
            remaining_cooldowns.update(
                {
                    position: FRUIT_TREE_COOLDOWN
                    for position in newly_harvested
                }
            )

        return replace(
            self,
            agents=tuple(updated_agents),
            turn=turn,
            action_log=self.action_log + tuple(new_log_entries),
            fruit_tree_cooldowns=remaining_cooldowns,
            tile_claims=tile_claims,
        )

    def run(
        self,
        max_turns: int = 0,
        log_directory: str | Path | None = "logs",
        state_callback: Callable[[GameState], bool] | None = None,
        *,
        action_selector: Callable[[AgentState], ActionSelection],
    ) -> GameState:
        """Run a simulation and return its final state."""

        if max_turns < 0:
            raise ValueError("Maximum turns cannot be negative")

        game_state = self
        initial_log_length = len(self.action_log)
        should_continue = state_callback(self) if state_callback else True
        for _ in range(max_turns if should_continue else 0):
            if not game_state.agents:
                break
            game_state = game_state._run_turn(action_selector)
            if state_callback is not None and not state_callback(game_state):
                break

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
