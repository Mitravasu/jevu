"""Action types and functions for applying one agent turn."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from jevu.agent import Agent
from jevu.world import Position, TileType, World

FOOD_FULLNESS_RESTORE = 2


class ExploreAction(StrEnum):
    """Movement actions available during the explore part of a turn."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class InteractAction(StrEnum):
    """Actions available during the interact part of a turn."""

    HARVEST = "harvest"
    EAT = "eat"


type Action = ExploreAction | InteractAction


@dataclass(frozen=True, slots=True)
class TurnActions:
    """The explore and interact actions selected for one turn."""

    interact: InteractAction
    explore: ExploreAction


def explore(agent: Agent, world: World, action: ExploreAction) -> Agent:
    """Move an agent one tile, or leave it in place at a world boundary."""

    offsets = {
        ExploreAction.UP: (0, -1),
        ExploreAction.DOWN: (0, 1),
        ExploreAction.LEFT: (-1, 0),
        ExploreAction.RIGHT: (1, 0),
    }
    dx, dy = offsets[action]
    destination = Position(agent.position.x + dx, agent.position.y + dy)
    if not world.contains(destination):
        return agent
    return replace(agent, position=destination)


def interact(agent: Agent, world: World, action: InteractAction) -> Agent:
    """Apply a harvest or eat action to an agent."""

    if action is InteractAction.HARVEST:
        if world.tile_at(agent.position) is TileType.FRUIT_TREE:
            return replace(agent, inventory=agent.inventory.add_food())
        return agent

    if agent.inventory.food == 0 or agent.hunger == 10:
        return agent
    return replace(
        agent,
        inventory=agent.inventory.consume_food(),
        hunger=min(10, agent.hunger + FOOD_FULLNESS_RESTORE),
    )


def take_turn(agent: Agent, world: World, actions: TurnActions) -> Agent:
    """Interact with the current tile first, then move."""

    updated_agent = interact(agent, world, actions.interact)
    return explore(updated_agent, world, actions.explore)
