import argparse
from collections.abc import Callable, Sequence
from contextlib import ExitStack

from jevu.agent import AgentState
from jevu.decision import ActionSelection
from jevu.game_state import GameState
from jevu.jev import DEFAULT_MODEL, JevActionSelector
from jevu.world import WorldConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JevU game")
    parser.add_argument(
        "--agent-backend",
        choices=("jev", "rl"),
        default="jev",
        help="Action selector to use (default: jev)",
    )
    parser.add_argument(
        "--bundle",
        help="Completed RL model bundle directory; required with --agent-backend rl",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Device for RL inference (default: auto)",
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--tree-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--turns-per-second", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.agent_backend == "rl" and args.bundle is None:
        parser.error("--bundle is required with --agent-backend rl")
    if args.agent_backend == "jev" and args.bundle is not None:
        parser.error("--bundle requires --agent-backend rl")
    return args


def run_simulation(
    game_state: GameState,
    args: argparse.Namespace,
    action_selector: Callable[[AgentState], ActionSelection],
) -> None:
    """Run the configured text or pygame simulation with one selector."""

    if args.pygame:
        from jevu.pygame_view import run_pygame

        run_pygame(
            game_state,
            max_turns=args.max_turns,
            action_selector=action_selector,
            turns_per_second=args.turns_per_second,
        )
    else:
        game_state.run(
            max_turns=args.max_turns,
            action_selector=action_selector,
        )


def main() -> None:
    args = parse_args()
    game_state = GameState.create(
        WorldConfig(
            width=args.width,
            height=args.height,
            fruit_tree_density=args.tree_density,
            seed=args.seed,
        ),
        agent_count=args.agents,
    )
    with ExitStack() as stack:
        if args.agent_backend == "rl":
            from jevu.rl.selector import RLActionSelector

            action_selector = RLActionSelector.from_bundle(
                args.bundle,
                device=args.device,
            )
        else:
            action_selector = stack.enter_context(JevActionSelector(model=args.model))
        run_simulation(game_state, args, action_selector)


if __name__ == "__main__":
    main()
