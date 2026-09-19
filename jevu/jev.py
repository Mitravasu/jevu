"""Jev-backed action selection for one agent state."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol

from typesafe_sdk import Choice, TypeSafeClient

from jevu.action_choice_data import (
    EXPLORE_CRITERIA,
    EXPLORE_INSTRUCTIONS,
    INTERACT_CRITERIA,
    INTERACT_INSTRUCTIONS,
)
from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import AgentState
from jevu.rules import MAX_HUNGER, MIN_HUNGER

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
    instructions=INTERACT_INSTRUCTIONS,
    criteria={
        action.value: description
        for action, description in INTERACT_CRITERIA.items()
    },
)

EXPLORE_QUESTION = Choice(
    instructions=EXPLORE_INSTRUCTIONS,
    criteria={
        action.value: description
        for action, description in EXPLORE_CRITERIA.items()
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
            "current_tile_cooldown": state.current_tile_cooldown,
            "current_tile_claim": state.current_tile_claim,
            "adjacent_tiles": {
                direction: tile.value
                for direction, tile in state.adjacent_tiles.items()
            },
            "adjacent_tile_cooldowns": state.adjacent_tile_cooldowns,
            "adjacent_tile_claims": state.adjacent_tile_claims,
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
