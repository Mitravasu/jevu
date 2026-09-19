"""Action types and functions for applying one agent turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from jevu.agent import Agent
from jevu.rules import (
    FOOD_EAT_COST,
    FOOD_HARVEST_AMOUNT,
    FOOD_HUNGER_RESTORE,
    MAX_HUNGER,
)
from jevu.world import Position, TileType, World


class ExploreAction(StrEnum):
    """Movement actions available during the explore part of a turn."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


class InteractAction(StrEnum):
    """Actions available during the interact part of a turn."""

    HARVEST = "harvest"
    EAT = "eat"
    CLAIM = "claim"
    NONE = "none"


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
        ExploreAction.STAY: (0, 0),
    }
    dx, dy = offsets[action]
    destination = Position(agent.position.x + dx, agent.position.y + dy)
    if not world.contains(destination):
        return agent
    return replace(agent, position=destination)


def interact(
    agent: Agent,
    world: World,
    action: InteractAction,
    *,
    fruit_tree_available: bool = True,
) -> Agent:
    """Apply a harvest or eat action to an agent."""

    if action is InteractAction.HARVEST:
        if (
            fruit_tree_available
            and world.tile_at(agent.position) is TileType.FRUIT_TREE
        ):
            return replace(
                agent,
                inventory=agent.inventory.add_food(FOOD_HARVEST_AMOUNT),
            )
        return agent

    if action is InteractAction.CLAIM:
        return agent

    if action is InteractAction.NONE:
        return agent

    if agent.inventory.food < FOOD_EAT_COST or agent.hunger == MAX_HUNGER:
        return agent
    return replace(
        agent,
        inventory=agent.inventory.consume_food(FOOD_EAT_COST),
        hunger=min(MAX_HUNGER, agent.hunger + FOOD_HUNGER_RESTORE),
    )


def claim_tile(
    agent: Agent,
    world: World,
    tile_claims: Mapping[Position, int],
) -> dict[Position, int]:
    """Return claims with the agent's current tile claimed when eligible."""

    if agent.position in tile_claims:
        return dict(tile_claims)
    if world.tile_at(agent.position) not in {TileType.BLANK, TileType.FRUIT_TREE}:
        return dict(tile_claims)
    return dict(tile_claims) | {agent.position: agent.number}


def take_turn(
    agent: Agent,
    world: World,
    actions: TurnActions,
    *,
    fruit_tree_available: bool = True,
) -> Agent:
    """Interact with the current tile first, then move."""

    updated_agent = interact(
        agent,
        world,
        actions.interact,
        fruit_tree_available=fruit_tree_available,
    )
    return explore(updated_agent, world, actions.explore)
