import unittest

from jevu.personalities import (
    PERSONALITY_ORDER,
    PERSONALITY_PROFILES,
    Personality,
    personality_for_agent,
)


class PersonalityTests(unittest.TestCase):
    def test_every_personality_has_a_profile(self) -> None:
        self.assertEqual(set(PERSONALITY_PROFILES), set(Personality))

    def test_assignment_is_deterministic_and_cycles(self) -> None:
        self.assertEqual(personality_for_agent(1), Personality.CARETAKER_DIPLOMAT)
        self.assertEqual(personality_for_agent(11), Personality.TYRANT_PLANNER)
        self.assertEqual(
            personality_for_agent(len(PERSONALITY_ORDER) + 1),
            Personality.CARETAKER_DIPLOMAT,
        )


if __name__ == "__main__":
    unittest.main()
