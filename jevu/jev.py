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
from jevu.agent import AgentState, TileObservation
from jevu.agent_goals import AGENT_GOALS, GAME_RULES
from jevu.decision import ChoiceTrace, TurnDecision
from jevu.personalities import PERSONALITY_PROFILES
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
    probabilities: Mapping[str, float]
    confidence: float


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

    personality = PERSONALITY_PROFILES[state.personality]

    def tile_payload(tile: TileObservation) -> dict[str, object]:
        return {
            "position": {"x": tile.position.x, "y": tile.position.y},
            "tile": tile.tile.value,
            "cooldown": tile.cooldown,
            "claim": tile.claim,
        }

    return {
        "goals": list(AGENT_GOALS),
        "rules": list(GAME_RULES),
        "agent_state": {
            "id": state.id,
            "personality": {
                "id": state.personality.value,
                "name": personality.name,
                "description": personality.description,
                "behavior_priorities": list(personality.behavior_priorities),
            },
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
            "claimed_territory": [
                {
                    "tile": tile_payload(claimed.tile),
                    "adjacent_tiles": {
                        direction: tile_payload(adjacent)
                        for direction, adjacent in claimed.adjacent_tiles.items()
                    },
                }
                for claimed in state.claimed_territory
            ],
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

    def __call__(self, state: AgentState) -> TurnDecision:
        result = self._client.system_one(
            state=_state_payload(state),
            questions=QUESTIONS,
        )
        actions = TurnActions(
            interact=InteractAction(result.choices["interact"].choice),
            explore=ExploreAction(result.choices["explore"].choice),
        )
        return TurnDecision(
            actions=actions,
            interact=self._choice_trace(result.choices["interact"]),
            explore=self._choice_trace(result.choices["explore"]),
        )

    @staticmethod
    def _choice_trace(result: ChoiceResult) -> ChoiceTrace:
        return ChoiceTrace(
            choice=result.choice,
            probabilities=dict(result.probabilities),
            confidence=result.confidence,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JevActionSelector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
