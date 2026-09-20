import unittest
from unittest.mock import patch
from types import SimpleNamespace

from jevu.action_choice_data import EXPLORE_CRITERIA, INTERACT_CRITERIA
from jevu.agent_goals import AGENT_GOALS, GAME_RULES
from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.agent import Agent
from jevu.jev import (
    DEFAULT_MODEL,
    EXPLORE_QUESTION,
    INTERACT_QUESTION,
    JevActionSelector,
)
from jevu.personalities import PERSONALITY_PROFILES, Personality
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
                "interact": SimpleNamespace(
                    choice="eat",
                    probabilities={
                        "harvest": 0.1,
                        "eat": 0.6,
                        "claim": 0.2,
                        "none": 0.1,
                    },
                    confidence=0.6,
                ),
                "explore": SimpleNamespace(
                    choice="left",
                    probabilities={
                        "up": 0.1,
                        "down": 0.1,
                        "left": 0.7,
                        "right": 0.05,
                        "stay": 0.05,
                    },
                    confidence=0.5,
                ),
            }
        )

    def close(self) -> None:
        self.closed = True


class JevActionSelectorTests(unittest.TestCase):
    def test_choice_data_covers_every_action(self) -> None:
        self.assertEqual(set(INTERACT_CRITERIA), set(InteractAction))
        self.assertEqual(set(EXPLORE_CRITERIA), set(ExploreAction))

    def test_stay_does_not_imply_a_survival_advantage(self) -> None:
        stay = EXPLORE_CRITERIA[ExploreAction.STAY]

        self.assertIn("only when every available move is less useful", stay)
        self.assertIn("does not conserve hunger", stay)

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

        agent = Agent(
            number=1,
            position=Position(1, 1),
            personality=Personality.FRONTIER_EXPLORER,
        )
        decision = selector(agent.state(world))

        self.assertEqual(
            decision.actions,
            TurnActions(
                interact=InteractAction.EAT,
                explore=ExploreAction.LEFT,
            ),
        )
        self.assertEqual(
            decision.interact.probabilities,
            {"harvest": 0.1, "eat": 0.6, "claim": 0.2, "none": 0.1},
        )
        self.assertEqual(decision.interact.confidence, 0.6)
        self.assertEqual(set(client.questions), {"interact", "explore"})
        self.assertEqual(client.state["goals"], list(AGENT_GOALS))
        self.assertEqual(client.state["rules"], list(GAME_RULES))
        self.assertEqual(client.state["agent_state"]["id"], "A1")
        profile = PERSONALITY_PROFILES[Personality.FRONTIER_EXPLORER]
        self.assertEqual(
            client.state["agent_state"]["personality"],
            {
                "id": Personality.FRONTIER_EXPLORER.value,
                "name": profile.name,
                "description": profile.description,
                "behavior_priorities": list(profile.behavior_priorities),
            },
        )
        self.assertEqual(client.state["agent_state"]["current_tile_cooldown"], 0)
        self.assertIsNone(client.state["agent_state"]["current_tile_claim"])
        self.assertEqual(client.state["agent_state"]["claimed_territory"], [])
        self.assertEqual(
            client.state["agent_state"]["adjacent_tiles"],
            {"up": "fruit_tree", "left": "blank"},
        )

    def test_serializes_claimed_tiles_and_their_neighbors(self) -> None:
        world = World(
            config=WorldConfig(width=2, height=1),
            tiles=((TileType.FRUIT_TREE, TileType.BLANK),),
        )
        agent = Agent(number=1, position=Position(0, 0))
        client = FakeClient()

        JevActionSelector(client=client)(
            agent.state(
                world,
                fruit_tree_cooldowns={Position(0, 0): 2},
                tile_claims={Position(0, 0): 1},
            )
        )

        territory = client.state["agent_state"]["claimed_territory"]
        self.assertEqual(
            territory,
            [
                {
                    "tile": {
                        "position": {"x": 0, "y": 0},
                        "tile": "fruit_tree",
                        "cooldown": 2,
                        "claim": "A1",
                    },
                    "adjacent_tiles": {
                        "right": {
                            "position": {"x": 1, "y": 0},
                            "tile": "blank",
                            "cooldown": 0,
                            "claim": None,
                        }
                    },
                }
            ],
        )

    def test_goals_are_state_not_question_instructions(self) -> None:
        self.assertNotIn("claim as many", INTERACT_QUESTION.instructions.lower())
        self.assertNotIn("claim as many", EXPLORE_QUESTION.instructions.lower())
        self.assertIn("Claim as many tiles as possible.", AGENT_GOALS)

    def test_does_not_close_an_injected_client(self) -> None:
        client = FakeClient()

        with JevActionSelector(client=client):
            pass

        self.assertFalse(client.closed)


if __name__ == "__main__":
    unittest.main()
