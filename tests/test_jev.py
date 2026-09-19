import unittest
from unittest.mock import patch
from types import SimpleNamespace

from jevu.action_choice_data import EXPLORE_CRITERIA, INTERACT_CRITERIA
from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import Agent
from jevu.jev import DEFAULT_MODEL, INTERACT_QUESTION, JevActionSelector
from jevu.rules import (
    FOOD_EAT_COST,
    FOOD_HARVEST_AMOUNT,
    FOOD_HUNGER_RESTORE,
    FRUIT_TREE_COOLDOWN,
)
from jevu.world import Position, TileType, World, WorldConfig


class FakeClient:
    def __init__(self) -> None:
        self.state = None
        self.questions = None
        self.closed = False

    def system_one(self, state, questions):
        self.state = state
        self.questions = questions
        return SimpleNamespace(
            choices={
                "interact": SimpleNamespace(choice="eat"),
                "explore": SimpleNamespace(choice="left"),
            }
        )

    def close(self) -> None:
        self.closed = True


class JevActionSelectorTests(unittest.TestCase):
    def test_choice_data_covers_every_action(self) -> None:
        self.assertEqual(set(INTERACT_CRITERIA), set(InteractAction))
        self.assertEqual(set(EXPLORE_CRITERIA), set(ExploreAction))

    @patch.dict("os.environ", {"JEV_API_TOKEN": "test-token"})
    @patch("jevu.jev.TypeSafeClient")
    def test_uses_the_configured_model(self, client_type) -> None:
        client_type.return_value = FakeClient()

        selector = JevActionSelector(model="jev-test")

        client_type.assert_called_once_with(
            api_key="test-token",
            model="jev-test",
        )
        selector.close()

    def test_default_model_is_pinned(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "jev-1.13.0")

    def test_question_criteria_use_gameplay_constants(self) -> None:
        harvest = INTERACT_QUESTION.criteria[InteractAction.HARVEST.value]
        eat = INTERACT_QUESTION.criteria[InteractAction.EAT.value]

        self.assertIn(str(FOOD_HARVEST_AMOUNT), harvest)
        self.assertIn(str(FRUIT_TREE_COOLDOWN), harvest)
        self.assertIn(str(FOOD_EAT_COST), eat)
        self.assertIn(str(FOOD_HUNGER_RESTORE), eat)
        self.assertIn(InteractAction.CLAIM.value, INTERACT_QUESTION.criteria)

    def test_selects_both_actions_from_one_agent_state_request(self) -> None:
        world = World(
            config=WorldConfig(width=2, height=2),
            tiles=(
                (TileType.BLANK, TileType.FRUIT_TREE),
                (TileType.BLANK, TileType.BLANK),
            ),
        )
        client = FakeClient()
        selector = JevActionSelector(client=client)

        actions = selector(Agent(number=1, position=Position(1, 1)).state(world))

        self.assertEqual(
            actions,
            TurnActions(
                interact=InteractAction.EAT,
                explore=ExploreAction.LEFT,
            ),
        )
        self.assertEqual(set(client.questions), {"interact", "explore"})
        self.assertEqual(client.state["agent_state"]["id"], "A1")
        self.assertEqual(client.state["agent_state"]["current_tile_cooldown"], 0)
        self.assertIsNone(client.state["agent_state"]["current_tile_claim"])
        self.assertEqual(
            client.state["agent_state"]["adjacent_tiles"],
            {"up": "fruit_tree", "left": "blank"},
        )

    def test_does_not_close_an_injected_client(self) -> None:
        client = FakeClient()

        with JevActionSelector(client=client):
            pass

        self.assertFalse(client.closed)


if __name__ == "__main__":
    unittest.main()
