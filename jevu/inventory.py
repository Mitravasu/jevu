"""Inventory state carried by an agent."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class InventoryState:
    """The resources currently carried by an agent."""

    food: int = 0

    def __post_init__(self) -> None:
        if self.food < 0:
            raise ValueError("Food cannot be negative")

    def add_food(self, amount: int = 1) -> InventoryState:
        if amount < 0:
            raise ValueError("Food amount cannot be negative")
        return replace(self, food=self.food + amount)

    def consume_food(self) -> InventoryState:
        if self.food == 0:
            return self
        return replace(self, food=self.food - 1)
