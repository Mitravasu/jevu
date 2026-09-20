"""Deterministic fixed-suite evaluation for saved RL policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from statistics import fmean

from jevu.agent import AgentState
from jevu.decision import ActionSelection
from jevu.game_state import GameState
from jevu.rules import MIN_HUNGER
from jevu.world import WorldConfig


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and normalized score for one world."""

    size: int
    seed: int
    max_turns: int
    agent_count: int
    completed_turns: int
    survivors: int
    survived_agent_turns: int
    claimed_tiles: int
    survival_ratio: float
    claim_ratio: float
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_case(
    action_selector: Callable[[AgentState], ActionSelection],
    *,
    size: int,
    seed: int,
    agent_count: int,
    max_turns: int,
    tree_density: float,
    survival_weight: float = 1.0,
    claim_weight: float = 1.0,
) -> EvaluationResult:
    """Run one square world without logs or console rendering."""

    if size <= 0 or size > 25:
        raise ValueError("Evaluation size must be between 1 and 25")
    if agent_count <= 0 or agent_count > size * size:
        raise ValueError("Agent count must fit within the evaluation world")
    if max_turns <= 0:
        raise ValueError("Maximum turns must be positive")
    if survival_weight < 0 or claim_weight < 0:
        raise ValueError("Score weights cannot be negative")
    weight_total = survival_weight + claim_weight
    if weight_total <= 0:
        raise ValueError("At least one score weight must be positive")

    game_state = GameState.create(
        WorldConfig(
            width=size,
            height=size,
            fruit_tree_density=tree_density,
            seed=seed,
        ),
        agent_count=agent_count,
    )
    for _ in range(max_turns):
        if not game_state.agents:
            break
        game_state = game_state.step(action_selector)

    survived_agent_turns = sum(
        entry.end_hunger > MIN_HUNGER for entry in game_state.action_log
    )
    survival_ratio = survived_agent_turns / (max_turns * agent_count)
    claimed_tiles = len(game_state.tile_claims)
    claim_ratio = claimed_tiles / (size * size)
    score = 100.0 * (
        survival_weight * survival_ratio + claim_weight * claim_ratio
    ) / weight_total
    return EvaluationResult(
        size=size,
        seed=seed,
        max_turns=max_turns,
        agent_count=agent_count,
        completed_turns=game_state.turn,
        survivors=len(game_state.agents),
        survived_agent_turns=survived_agent_turns,
        claimed_tiles=claimed_tiles,
        survival_ratio=survival_ratio,
        claim_ratio=claim_ratio,
        score=score,
    )


def summarize(results: Iterable[EvaluationResult]) -> dict[str, float | int]:
    """Return macro averages so every evaluated world has equal influence."""

    values = tuple(results)
    if not values:
        raise ValueError("Cannot summarize an empty evaluation")
    return {
        "episodes": len(values),
        "average_score": fmean(result.score for result in values),
        "average_survival_ratio": fmean(
            result.survival_ratio for result in values
        ),
        "average_claim_ratio": fmean(result.claim_ratio for result in values),
        "total_survived_agent_turns": sum(
            result.survived_agent_turns for result in values
        ),
        "total_claimed_tiles": sum(result.claimed_tiles for result in values),
    }
