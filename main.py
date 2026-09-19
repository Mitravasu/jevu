import argparse

from jevu.game_state import GameState
from jevu.world import WorldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JevU game")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--tree-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=1)
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
    game_state.run()


if __name__ == "__main__":
    main()
