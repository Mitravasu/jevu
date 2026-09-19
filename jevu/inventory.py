"""Inventory state carried by an agent."""

from __future__ import annotations

from dataclasses import dataclass, replace

from jevu.rules import INITIAL_FOOD, MIN_FOOD


@dataclass(frozen=True, slots=True)
class InventoryState:
    """The resources currently carried by an agent."""

    food: int = INITIAL_FOOD

    def __post_init__(self) -> None:
        if self.food < MIN_FOOD:
            raise ValueError(f"Food cannot be less than {MIN_FOOD}")

    def add_food(self, amount: int) -> InventoryState:
        if amount < MIN_FOOD:
            raise ValueError(f"Food amount cannot be less than {MIN_FOOD}")
        return replace(self, food=self.food + amount)

    def consume_food(self, amount: int) -> InventoryState:
        if amount < MIN_FOOD:
            raise ValueError(f"Food amount cannot be less than {MIN_FOOD}")
        if self.food < amount:
            return self
        return replace(self, food=self.food - amount)
