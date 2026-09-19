import unittest

from jevu.actions import (
    ExploreAction,
    InteractAction,
    TurnActions,
    explore,
    interact,
    take_turn,
)
from jevu.agent import Agent
from jevu.world import Position, TileType, World, WorldConfig


class ActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(
            config=WorldConfig(width=2, height=2, seed=42),
            tiles=(
                (TileType.BLANK, TileType.FRUIT_TREE),
                (TileType.BLANK, TileType.BLANK),
            ),
        )

    def test_explore_moves_in_each_direction(self) -> None:
        center = Agent(number=1, position=Position(1, 1))

        self.assertEqual(
            explore(center, self.world, ExploreAction.UP).position,
            Position(1, 0),
        )
        self.assertEqual(
            explore(center, self.world, ExploreAction.LEFT).position,
            Position(0, 1),
        )

    def test_explore_stays_in_place_at_boundary(self) -> None:
        agent = Agent(number=1, position=Position(0, 0))

        self.assertEqual(explore(agent, self.world, ExploreAction.UP), agent)
        self.assertEqual(explore(agent, self.world, ExploreAction.LEFT), agent)

    def test_harvest_collects_fruit_only_on_tree(self) -> None:
        blank_agent = Agent(number=1, position=Position(0, 0))
        tree_agent = Agent(number=1, position=Position(1, 0))

        self.assertEqual(
            interact(blank_agent, self.world, InteractAction.HARVEST).carried_fruit,
            0,
        )
        self.assertEqual(
            interact(tree_agent, self.world, InteractAction.HARVEST).carried_fruit,
            1,
        )

    def test_eat_consumes_fruit_and_restores_one_hunger(self) -> None:
        agent = Agent(
            number=1,
            position=Position(0, 0),
            hunger=8,
            carried_fruit=2,
        )

        updated = interact(agent, self.world, InteractAction.EAT)

        self.assertEqual(updated.hunger, 9)
        self.assertEqual(updated.carried_fruit, 1)

    def test_eat_does_nothing_without_need_or_fruit(self) -> None:
        hungry_agent = Agent(number=1, position=Position(0, 0), hunger=8)
        full_agent = Agent(
            number=1,
            position=Position(0, 0),
            hunger=10,
            carried_fruit=1,
        )

        self.assertEqual(
            interact(hungry_agent, self.world, InteractAction.EAT),
            hungry_agent,
        )
        self.assertEqual(
            interact(full_agent, self.world, InteractAction.EAT),
            full_agent,
        )

    def test_turn_interacts_before_moving(self) -> None:
        agent = Agent(number=1, position=Position(1, 0))
        actions = TurnActions(
            explore=ExploreAction.LEFT,
            interact=InteractAction.HARVEST,
        )

        updated = take_turn(agent, self.world, actions)

        self.assertEqual(updated.position, Position(0, 0))
        self.assertEqual(updated.carried_fruit, 1)


if __name__ == "__main__":
    unittest.main()
