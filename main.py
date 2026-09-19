import argparse

from jevu.game_state import GameState
from jevu.jev import DEFAULT_MODEL, JevActionSelector
from jevu.world import WorldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JevU game")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--tree-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--turns-per-second", type=float, default=2.0)
    return parser.parse_args()


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
    with JevActionSelector(model=args.model) as action_selector:
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


if __name__ == "__main__":
    main()
