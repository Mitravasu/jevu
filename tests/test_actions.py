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
from jevu.inventory import InventoryState
from jevu.rules import (
    FOOD_EAT_COST,
    FOOD_HARVEST_AMOUNT,
    FOOD_HUNGER_RESTORE,
    MAX_HUNGER,
)
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
        top_left_agent = Agent(number=1, position=Position(0, 0))
        bottom_right_agent = Agent(number=2, position=Position(1, 1))

        self.assertEqual(
            explore(top_left_agent, self.world, ExploreAction.UP),
            top_left_agent,
        )
        self.assertEqual(
            explore(top_left_agent, self.world, ExploreAction.LEFT),
            top_left_agent,
        )
        self.assertEqual(
            explore(bottom_right_agent, self.world, ExploreAction.DOWN),
            bottom_right_agent,
        )
        self.assertEqual(
            explore(bottom_right_agent, self.world, ExploreAction.RIGHT),
            bottom_right_agent,
        )

    def test_harvest_collects_fruit_only_on_tree(self) -> None:
        blank_agent = Agent(number=1, position=Position(0, 0))
        tree_agent = Agent(number=1, position=Position(1, 0))

        self.assertEqual(
            interact(blank_agent, self.world, InteractAction.HARVEST).inventory.food,
            0,
        )
        self.assertEqual(
            interact(tree_agent, self.world, InteractAction.HARVEST).inventory.food,
            FOOD_HARVEST_AMOUNT,
        )

    def test_eat_consumes_food_and_restores_configured_hunger(self) -> None:
        agent = Agent(
            number=1,
            position=Position(0, 0),
            hunger=MAX_HUNGER - FOOD_HUNGER_RESTORE,
            inventory=InventoryState(food=FOOD_EAT_COST * 2),
        )

        updated = interact(agent, self.world, InteractAction.EAT)

        self.assertEqual(updated.hunger, MAX_HUNGER)
        self.assertEqual(updated.inventory.food, FOOD_EAT_COST)

    def test_eat_caps_hunger_at_configured_maximum(self) -> None:
        agent = Agent(
            number=1,
            position=Position(0, 0),
            hunger=MAX_HUNGER - 1,
            inventory=InventoryState(food=FOOD_EAT_COST),
        )

        updated = interact(agent, self.world, InteractAction.EAT)

        self.assertEqual(updated.hunger, MAX_HUNGER)
        self.assertEqual(updated.inventory.food, 0)

    def test_eat_does_nothing_without_need_or_fruit(self) -> None:
        hungry_agent = Agent(
            number=1,
            position=Position(0, 0),
            hunger=MAX_HUNGER - FOOD_HUNGER_RESTORE,
        )
        full_agent = Agent(
            number=1,
            position=Position(0, 0),
            hunger=MAX_HUNGER,
            inventory=InventoryState(food=FOOD_EAT_COST),
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
        self.assertEqual(updated.inventory.food, FOOD_HARVEST_AMOUNT)


if __name__ == "__main__":
    unittest.main()
