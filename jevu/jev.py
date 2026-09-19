"""Jev-backed action selection for one agent state."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol

from typesafe_sdk import Choice, TypeSafeClient

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import AgentState
from jevu.rules import (
    FOOD_EAT_COST,
    FOOD_HARVEST_AMOUNT,
    FOOD_HUNGER_RESTORE,
    MAX_HUNGER,
    MIN_HUNGER,
)

DEFAULT_MODEL = "jev-1.13.0"


class SystemOneClient(Protocol):
    """The part of the TypeSafe client used by the action selector."""

    def system_one(
        self,
        state: object,
        questions: Mapping[str, Choice],
    ) -> SystemOneResult: ...

    def close(self) -> None: ...


class ChoiceResult(Protocol):
    choice: str


class SystemOneResult(Protocol):
    choices: Mapping[str, ChoiceResult]


INTERACT_QUESTION = Choice(
    instructions=(
        "Which interaction should the agent take before moving to improve its "
        "chance of surviving? Use only `agent_state`."
    ),
    criteria={
        InteractAction.HARVEST.value: (
            f"Collect {FOOD_HARVEST_AMOUNT} food when "
            "`agent_state.current_tile` is `fruit_tree`; on any other tile "
            "this has no effect."
        ),
        InteractAction.EAT.value: (
            f"Consume {FOOD_EAT_COST} food to restore "
            f"{FOOD_HUNGER_RESTORE} hunger points when "
            f"`agent_state.inventory.food` is at least {FOOD_EAT_COST} and "
            f"hunger is below {MAX_HUNGER}; otherwise this has no effect."
        ),
    },
)

EXPLORE_QUESTION = Choice(
    instructions=(
        "Which direction should the agent move after interacting to improve its "
        "chance of surviving? Use `agent_state.adjacent_tiles`; a missing "
        "direction is a world boundary and leaves the agent in place."
    ),
    criteria={
        action.value: f"Move one tile {action.value}."
        for action in ExploreAction
    },
)

QUESTIONS = {
    "interact": INTERACT_QUESTION,
    "explore": EXPLORE_QUESTION,
}


def _state_payload(state: AgentState) -> dict[str, object]:
    """Convert domain state into the JSON state Jev evaluates."""

    return {
        "agent_state": {
            "id": state.id,
            "position": {"x": state.position.x, "y": state.position.y},
            "current_tile": state.current_tile.value,
            "adjacent_tiles": {
                direction: tile.value
                for direction, tile in state.adjacent_tiles.items()
            },
            "hunger": state.hunger,
            "hunger_scale": {
                str(MIN_HUNGER): "the agent starves after this turn",
                str(MAX_HUNGER): "the agent is fully fed",
            },
            "inventory": {"food": state.inventory.food},
        }
    }


class JevActionSelector:
    """Select actions with Jev without applying them to the game."""

    def __init__(
        self,
        client: SystemOneClient | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._owns_client = client is None
        if client is None:
            try:
                api_key = os.environ["JEV_API_TOKEN"]
            except KeyError:
                raise RuntimeError("JEV_API_TOKEN is required to run JevU") from None
            client = TypeSafeClient(api_key=api_key, model=model)
        self._client = client

    def __call__(self, state: AgentState) -> TurnActions:
        result = self._client.system_one(
            state=_state_payload(state),
            questions=QUESTIONS,
        )
        return TurnActions(
            interact=InteractAction(result.choices["interact"].choice),
            explore=ExploreAction(result.choices["explore"].choice),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JevActionSelector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
