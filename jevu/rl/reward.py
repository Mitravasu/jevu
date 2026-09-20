"""Auditable reward calculation for the two agent goals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from jevu.rl.config import RewardConfig


@dataclass(frozen=True, slots=True)
class TransitionEvents:
    survived: bool
    newly_claimed_tiles: int
    ineffective_actions: int


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    survival: float
    territory: float
    death: float
    ineffective_action: float

    @property
    def total(self) -> float:
        return self.survival + self.territory + self.death + self.ineffective_action

    def to_dict(self) -> dict[str, float]:
        return asdict(self) | {"total": self.total}


def calculate_reward(
    events: TransitionEvents,
    config: RewardConfig,
) -> RewardBreakdown:
    """Score survival and incremental territory without rewarding proxy behavior."""

    if events.newly_claimed_tiles < 0 or events.ineffective_actions < 0:
        raise ValueError("Transition event counts cannot be negative")
    return RewardBreakdown(
        survival=config.survival_per_turn if events.survived else 0.0,
        territory=events.newly_claimed_tiles * config.claim_tile,
        death=0.0 if events.survived else config.death,
        ineffective_action=events.ineffective_actions * config.ineffective_action,
    )
