"""Complete state and initial setup for a JevU game."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random

from jevu.actions import InteractAction, TurnActions, claim_tile, take_turn
from jevu.agent import RECENT_POSITION_WINDOW, Agent, AgentState
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
class TurnPreparation:
    """Automatic effects applied before agents choose actions for a turn."""

    agents: tuple[Agent, ...]
    automatic_food: dict[int, int]
    newly_harvested: frozenset[Position]


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """Result of applying one agent's action inside a shared world turn."""

    entry: ActionLogEntry
    agent_number: int
    survived: bool
    newly_claimed_tiles: int


class TurnSession:
    """Incrementally execute one world turn in deterministic agent order."""

    def __init__(self, game_state: GameState) -> None:
        self._game_state = game_state
        preparation = game_state._prepare_turn()
        self._turn = game_state.turn + 1
        self._automatic_food = preparation.automatic_food
        self._newly_harvested = set(preparation.newly_harvested)
        self._turn_agents = preparation.agents
        self._occupied_positions = {agent.position for agent in self._turn_agents}
        self._updated_agents: list[Agent] = []
        self._new_log_entries: list[ActionLogEntry] = []
        self._tile_claims = dict(game_state.tile_claims)
        self._index = 0

    @property
    def complete(self) -> bool:
        return self._index >= len(self._turn_agents)

    @property
    def tile_claims(self) -> dict[Position, int]:
        return dict(self._tile_claims)

    @property
    def current_agent(self) -> Agent:
        if self.complete:
            raise RuntimeError("Every agent has already acted this turn")
        return self._turn_agents[self._index]

    def current_state(self) -> AgentState:
        """Return the state the next agent observes before choosing an action."""

        agent = self.current_agent
        visible_agents = {
            other.position: other.number
            for other in (
                *self._updated_agents,
                *self._turn_agents[self._index :],
            )
        }
        return agent.state(
            self._game_state.world,
            self._game_state._active_cooldowns(self._newly_harvested),
            self._tile_claims,
            visible_agents,
        )

    def apply(self, selection: ActionSelection) -> AgentTurnResult:
        """Apply one selection and advance to the next agent in this turn."""

        agent = self.current_agent
        active_cooldowns = self._game_state._active_cooldowns(self._newly_harvested)
        self._occupied_positions.remove(agent.position)
        decision = selection if isinstance(selection, TurnDecision) else None
        actions = decision.actions if decision is not None else selection
        claims_before = sum(
            owner == agent.number for owner in self._tile_claims.values()
        )
        if actions.interact is InteractAction.CLAIM:
            self._tile_claims = claim_tile(
                agent,
                self._game_state.world,
                self._tile_claims,
            )
        claims_after = sum(
            owner == agent.number for owner in self._tile_claims.values()
        )
        updated_agent = take_turn(
            agent,
            self._game_state.world,
            actions,
            fruit_tree_available=active_cooldowns.get(agent.position, 0) == 0,
        )
        if (
            actions.interact is InteractAction.HARVEST
            and updated_agent.inventory.food > agent.inventory.food
        ):
            self._newly_harvested.add(agent.position)

        if updated_agent.position in self._occupied_positions:
            updated_agent = replace(updated_agent, position=agent.position)

        position_history = agent.recent_positions or (agent.position,)
        updated_agent = replace(
            updated_agent,
            recent_positions=(
                *position_history,
                updated_agent.position,
            )[-RECENT_POSITION_WINDOW:],
            hunger=max(
                MIN_HUNGER,
                updated_agent.hunger - HUNGER_LOSS_PER_TURN,
            ),
        )
        entry = ActionLogEntry(
            turn=self._turn,
            agent_id=agent.id,
            actions=actions,
            start_position=agent.position,
            end_position=updated_agent.position,
            start_hunger=agent.hunger,
            end_hunger=updated_agent.hunger,
            start_food=agent.inventory.food,
            end_food=updated_agent.inventory.food,
            automatic_harvest_food=self._automatic_food.get(agent.number, 0),
            decision=decision,
        )
        self._new_log_entries.append(entry)
        survived = updated_agent.hunger > MIN_HUNGER
        if survived:
            self._updated_agents.append(updated_agent)
            self._occupied_positions.add(updated_agent.position)
        self._index += 1
        return AgentTurnResult(
            entry=entry,
            agent_number=agent.number,
            survived=survived,
            newly_claimed_tiles=claims_after - claims_before,
        )

    def finish(self) -> GameState:
        """Build the next immutable game state after every agent has acted."""

        if not self.complete:
            raise RuntimeError("Cannot finish a turn before every agent has acted")
        remaining_cooldowns = {
            position: cooldown - 1
            for position, cooldown in self._game_state.fruit_tree_cooldowns.items()
            if cooldown > 1
        }
        if FRUIT_TREE_COOLDOWN > 0:
            remaining_cooldowns.update(
                {position: FRUIT_TREE_COOLDOWN for position in self._newly_harvested}
            )
        return replace(
            self._game_state,
            agents=tuple(self._updated_agents),
            turn=self._turn,
            action_log=(self._game_state.action_log + tuple(self._new_log_entries)),
            fruit_tree_cooldowns=remaining_cooldowns,
            tile_claims=self._tile_claims,
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

    def _prepare_turn(self) -> TurnPreparation:
        """Apply automatic harvests that occur before action selection."""

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

        agents = tuple(
            replace(
                agent,
                inventory=agent.inventory.add_food(automatic_food[agent.number]),
            )
            if agent.number in automatic_food
            else agent
            for agent in self.agents
        )
        return TurnPreparation(
            agents=agents,
            automatic_food=automatic_food,
            newly_harvested=frozenset(newly_harvested),
        )

    def _active_cooldowns(
        self,
        newly_harvested: frozenset[Position] | set[Position],
    ) -> dict[Position, int]:
        active_cooldowns = {
            position: cooldown
            for position, cooldown in self.fruit_tree_cooldowns.items()
            if cooldown > 0
        }
        if FRUIT_TREE_COOLDOWN > 0:
            active_cooldowns.update(
                {position: FRUIT_TREE_COOLDOWN for position in newly_harvested}
            )
        return active_cooldowns

    def next_single_agent_state(self) -> AgentState:
        """Preview the exact state a sole agent will use for its next decision."""

        if len(self.agents) != 1:
            raise ValueError("Next-turn preview requires exactly one living agent")
        return self.start_turn().current_state()

    def start_turn(self) -> TurnSession:
        """Start an incremental turn shared by simulation and RL training."""

        return TurnSession(self)

    def step(
        self,
        action_selector: Callable[[AgentState], ActionSelection],
    ) -> GameState:
        """Advance every living agent by one deterministic simulation turn."""

        turn = self.start_turn()
        while not turn.complete:
            turn.apply(action_selector(turn.current_state()))
        return turn.finish()

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
            game_state = game_state.step(action_selector)
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
