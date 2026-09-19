import unittest

from jevu.inventory import InventoryState
from jevu.rules import FOOD_EAT_COST, FOOD_HARVEST_AMOUNT


class InventoryStateTests(unittest.TestCase):
    def test_add_food_increments_food(self) -> None:
        self.assertEqual(
            InventoryState().add_food(FOOD_HARVEST_AMOUNT).food,
            FOOD_HARVEST_AMOUNT,
        )

    def test_consume_food_decrements_food(self) -> None:
        inventory = InventoryState(food=FOOD_EAT_COST * 2)

        self.assertEqual(
            inventory.consume_food(FOOD_EAT_COST).food,
            FOOD_EAT_COST,
        )

    def test_consume_food_does_nothing_when_empty(self) -> None:
        inventory = InventoryState()

        self.assertEqual(inventory.consume_food(FOOD_EAT_COST), inventory)

    def test_negative_food_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InventoryState(food=-1)


if __name__ == "__main__":
    unittest.main()
