import unittest

from jevu.actions import ExploreAction, InteractAction, TurnActions
from jevu.rl.evaluation import EvaluationResult, evaluate_case, summarize


def stay_and_claim(_state):
    return TurnActions(InteractAction.CLAIM, ExploreAction.STAY)


class RLEvaluationTests(unittest.TestCase):
    def test_case_scores_survival_and_claim_coverage(self) -> None:
        result = evaluate_case(
            stay_and_claim,
            size=2,
            seed=7,
            agent_count=1,
            max_turns=2,
            tree_density=0.0,
        )

        self.assertEqual(result.survived_agent_turns, 2)
        self.assertEqual(result.claimed_tiles, 1)
        self.assertEqual(result.survival_ratio, 1.0)
        self.assertEqual(result.claim_ratio, 0.25)
        self.assertEqual(result.score, 62.5)

    def test_case_supports_worlds_larger_than_25(self) -> None:
        result = evaluate_case(
            stay_and_claim,
            size=50,
            seed=7,
            agent_count=1,
            max_turns=1,
            tree_density=0.0,
        )

        self.assertEqual(result.size, 50)
        self.assertEqual(result.claimed_tiles, 1)
        self.assertEqual(result.claim_ratio, 1 / 2500)

    def test_summary_macro_averages_episodes(self) -> None:
        first = EvaluationResult(5, 1, 10, 1, 10, 1, 10, 5, 1.0, 0.2, 60.0)
        second = EvaluationResult(10, 2, 10, 1, 5, 0, 5, 10, 0.5, 0.1, 30.0)

        summary = summarize([first, second])

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["average_score"], 45.0)
        self.assertEqual(summary["average_survival_ratio"], 0.75)
        self.assertAlmostEqual(summary["average_claim_ratio"], 0.15)


if __name__ == "__main__":
    unittest.main()
