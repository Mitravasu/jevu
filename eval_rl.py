"""Evaluate a saved RL bundle over fixed seeds and world sizes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from jevu.rl.bundle import METADATA_FILENAME, write_json
from jevu.rl.evaluation import EvaluationResult, evaluate_case, summarize
from jevu.rl.selector import RLActionSelector

DEFAULT_SEEDS = (7, 42, 101, 202, 999)
MAX_EVALUATION_SIZE = 25


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a saved RL policy on fixed seeds and world sizes"
    )
    parser.add_argument("--bundle", required=True, help="Completed RL bundle directory")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Square world sizes (default: 5, 10, 15, 20, 25 when supported)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-size", type=int, default=MAX_EVALUATION_SIZE)
    parser.add_argument("--agents", type=int, help="Agents per world (default: training value)")
    parser.add_argument(
        "--max-turns", type=int, help="Turn limit (default: training value)"
    )
    parser.add_argument(
        "--tree-density", type=float, help="Tree density (default: training value)"
    )
    parser.add_argument("--survival-weight", type=float, default=1.0)
    parser.add_argument("--claim-weight", type=float, default=1.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", help="JSON result path (default: inside the bundle)")
    args = parser.parse_args(argv)
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    if not 1 <= args.max_size <= MAX_EVALUATION_SIZE:
        parser.error(f"--max-size must be between 1 and {MAX_EVALUATION_SIZE}")
    if args.sizes and any(
        size < 1 or size > MAX_EVALUATION_SIZE for size in args.sizes
    ):
        parser.error(f"--sizes values must be between 1 and {MAX_EVALUATION_SIZE}")
    if args.agents is not None and args.agents <= 0:
        parser.error("--agents must be positive")
    if args.max_turns is not None and args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if args.tree_density is not None and not 0 <= args.tree_density <= 1:
        parser.error("--tree-density must be between 0 and 1")
    if args.survival_weight < 0 or args.claim_weight < 0:
        parser.error("score weights cannot be negative")
    if args.survival_weight + args.claim_weight <= 0:
        parser.error("at least one score weight must be positive")
    return args


def _training_environment(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        environment = metadata["configuration"]["environment"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Bundle metadata has no training environment") from exc
    if not isinstance(environment, dict):
        raise TypeError("Bundle training environment metadata is invalid")
    return environment


def _default_sizes(maximum: int, agent_count: int) -> list[int]:
    minimum = math.ceil(math.sqrt(agent_count))
    sizes = [size for size in range(5, maximum + 1, 5) if size >= minimum]
    if maximum >= minimum and maximum not in sizes:
        sizes.append(maximum)
    if not sizes:
        raise ValueError(
            f"No world up to {maximum}x{maximum} can fit {agent_count} agents"
        )
    return sizes


def _print_results(
    results: list[EvaluationResult], summary: dict[str, float | int]
) -> None:
    print(
        "\nSize  Seed   Survivors  Agent turns  Claims  Survival  Coverage  Score"
    )
    for result in results:
        print(
            f"{result.size:>4}  {result.seed:>5}  "
            f"{result.survivors:>5}/{result.agent_count:<5}  "
            f"{result.survived_agent_turns:>11}  {result.claimed_tiles:>6}  "
            f"{result.survival_ratio:>7.1%}  {result.claim_ratio:>7.1%}  "
            f"{result.score:>5.1f}"
        )

    by_size: dict[int, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        by_size[result.size].append(result)
    print("\nAverage by size")
    for size, size_results in sorted(by_size.items()):
        size_summary = summarize(size_results)
        print(
            f"  {size:>2}x{size:<2}: score={size_summary['average_score']:.1f}, "
            f"survival={size_summary['average_survival_ratio']:.1%}, "
            f"coverage={size_summary['average_claim_ratio']:.1%}"
        )
    print(
        f"\nOverall score: {summary['average_score']:.1f}/100 "
        f"(survival={summary['average_survival_ratio']:.1%}, "
        f"coverage={summary['average_claim_ratio']:.1%})"
    )


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    bundle = Path(args.bundle)
    metadata_path = bundle / METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"Incomplete RL model bundle: {bundle}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())
    training = _training_environment(metadata)
    selector = RLActionSelector.from_bundle(bundle, device=args.device)

    model_maximum = min(
        selector.observation_spec.width,
        selector.observation_spec.height,
        args.max_size,
        MAX_EVALUATION_SIZE,
    )
    agent_count = args.agents or int(training["agent_count"])
    max_turns = args.max_turns or int(training["max_turns"])
    tree_density = (
        args.tree_density
        if args.tree_density is not None
        else float(training["fruit_tree_density"])
    )
    sizes = sorted(set(args.sizes or _default_sizes(model_maximum, agent_count)))
    for size in sizes:
        if size > model_maximum:
            raise ValueError(
                f"Size {size} exceeds this model's {model_maximum}x{model_maximum} "
                "evaluation limit"
            )
        if agent_count > size * size:
            raise ValueError(f"{agent_count} agents do not fit in a {size}x{size} world")

    cases = [(size, seed) for size in sizes for seed in args.seeds]
    results = [
        evaluate_case(
            selector,
            size=size,
            seed=seed,
            agent_count=agent_count,
            max_turns=max_turns,
            tree_density=tree_density,
            survival_weight=args.survival_weight,
            claim_weight=args.claim_weight,
        )
        for size, seed in tqdm(
            cases,
            desc="Evaluating",
            unit="world",
            disable=args.no_progress,
        )
    ]
    summary = summarize(results)
    _print_results(results, summary)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(args.output)
        if args.output
        else bundle / "evaluations" / f"evaluation-{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "bundle": str(bundle),
            "created_at": datetime.now(UTC).isoformat(),
            "settings": {
                "sizes": sizes,
                "seeds": args.seeds,
                "agent_count": agent_count,
                "max_turns": max_turns,
                "tree_density": tree_density,
                "survival_weight": args.survival_weight,
                "claim_weight": args.claim_weight,
                "deterministic": True,
            },
            "score_definition": (
                "100 * weighted mean of survived agent-turn ratio and claimed "
                "world-tile ratio"
            ),
            "summary": summary,
            "episodes": [result.to_dict() for result in results],
        },
    )
    print(f"Evaluation file: {output}")
    return output


if __name__ == "__main__":
    main()
