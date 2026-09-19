import unittest

from jevu.inventory import InventoryState


class InventoryStateTests(unittest.TestCase):
    def test_add_food_increments_food(self) -> None:
        self.assertEqual(InventoryState().add_food().food, 1)

    def test_consume_food_decrements_food(self) -> None:
        self.assertEqual(InventoryState(food=2).consume_food().food, 1)

    def test_consume_food_does_nothing_when_empty(self) -> None:
        inventory = InventoryState()

        self.assertEqual(inventory.consume_food(), inventory)

    def test_negative_food_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InventoryState(food=-1)


if __name__ == "__main__":
    unittest.main()
