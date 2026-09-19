import argparse

from jevu.agent import place_agents
from jevu.world import World, WorldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a JevU world")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--tree-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world = World.generate(
        WorldConfig(
            width=args.width,
            height=args.height,
            fruit_tree_density=args.tree_density,
            seed=args.seed,
        )
    )
    agents = place_agents(world, args.agents)
    print(world.render_ascii(agents))


if __name__ == "__main__":
    main()
