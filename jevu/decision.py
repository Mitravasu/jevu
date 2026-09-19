"""Typed action decisions and their model traces."""

from __future__ import annotations

from dataclasses import dataclass

from jevu.actions import TurnActions


@dataclass(frozen=True, slots=True)
class ChoiceTrace:
    """One selected choice with its full probability distribution."""

    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """Selected turn actions and the judgments that produced them."""

    actions: TurnActions
    interact: ChoiceTrace
    explore: ChoiceTrace


type ActionSelection = TurnActions | TurnDecision
