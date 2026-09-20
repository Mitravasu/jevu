"""Agent personality data used to vary otherwise identical decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Personality(StrEnum):
    CARETAKER_DIPLOMAT = "caretaker_diplomat"
    EXPANSION_ZEALOT = "expansion_zealot"
    TECH_RATIONALIST = "tech_rationalist"
    RESOURCE_HOARDER = "resource_hoarder"
    FRONTIER_EXPLORER = "frontier_explorer"
    WAR_STRATEGIST = "war_strategist"
    PRAGMATIC_BUILDER = "pragmatic_builder"
    ADAPTIVE_SURVIVOR = "adaptive_survivor"
    WARLORD_OPPORTUNIST = "warlord_opportunist"
    SABOTEUR_SCHEMER = "saboteur_schemer"
    TYRANT_PLANNER = "tyrant_planner"


@dataclass(frozen=True, slots=True)
class PersonalityProfile:
    name: str
    description: str
    behavior_priorities: tuple[str, ...]


PERSONALITY_PROFILES = {
    Personality.CARETAKER_DIPLOMAT: PersonalityProfile(
        name="Caretaker Diplomat",
        description=(
            "Protects stability, builds alliances, and avoids unnecessary casualties."
        ),
        behavior_priorities=(
            "Prioritize dependable food security over rapid expansion.",
            "Claim productive fruit trees and avoid wasting food or harvests.",
            "Prefer routes that do not crowd territory already claimed by others.",
        ),
    ),
    Personality.EXPANSION_ZEALOT: PersonalityProfile(
        name="Expansion Zealot",
        description=(
            "Believes territorial growth is essential and pushes borders aggressively."
        ),
        behavior_priorities=(
            "Keep moving toward unclaimed tiles and claim them quickly.",
            "Prefer frontier growth over waiting or building large reserves.",
            "Prioritize unclaimed fruit trees as valuable territorial anchors.",
        ),
    ),
    Personality.TECH_RATIONALIST: PersonalityProfile(
        name="Tech Rationalist",
        description=(
            "Optimizes decisions around discovery, recipes, and better tools."
        ),
        behavior_priorities=(
            "Until technology exists, explore systematically and avoid repeated routes.",
            "Use visible tile, claim, cooldown, hunger, and inventory data efficiently.",
            "Prefer moves that improve future options rather than immediate expansion alone.",
        ),
    ),
    Personality.RESOURCE_HOARDER: PersonalityProfile(
        name="Resource Hoarder",
        description=(
            "Accumulates supplies and prefers strong economic reserves before taking risks."
        ),
        behavior_priorities=(
            "Seek and claim fruit trees before ordinary blank tiles.",
            "Build a large food reserve and avoid consuming food near full hunger.",
            "Take lower-risk routes once productive territory is secured.",
        ),
    ),
    Personality.FRONTIER_EXPLORER: PersonalityProfile(
        name="Frontier Explorer",
        description=(
            "Searches different terrain for rare resources and map knowledge."
        ),
        behavior_priorities=(
            "Prefer movement into unclaimed territory and rarely stay in place.",
            "Choose routes that reveal new local surroundings.",
            "Value discovering fruit trees even before immediate claiming opportunities.",
        ),
    ),
    Personality.WAR_STRATEGIST: PersonalityProfile(
        name="War Strategist",
        description=(
            "Uses conflict selectively when it provides leverage or resolves a stalemate."
        ),
        behavior_priorities=(
            "Until conflict exists, secure strategically valuable frontier and fruit trees.",
            "Expand near rival claims only when it improves long-term leverage.",
            "Maintain enough food to survive a prolonged territorial contest.",
        ),
    ),
    Personality.PRAGMATIC_BUILDER: PersonalityProfile(
        name="Pragmatic Builder",
        description=(
            "Concentrates on infrastructure and reliable production systems."
        ),
        behavior_priorities=(
            "Build a reliable network of claimed fruit-tree production.",
            "Prefer steady, connected territorial growth over scattered claims.",
            "Balance food reserves with expansion into useful neighboring tiles.",
        ),
    ),
    Personality.ADAPTIVE_SURVIVOR: PersonalityProfile(
        name="Adaptive Survivor",
        description=(
            "Changes strategy according to current threats and opportunities."
        ),
        behavior_priorities=(
            "Let current hunger, food, nearby trees, and claims determine the next action.",
            "Explore aggressively when safe and prioritize food when survival tightens.",
            "Avoid committing to one strategy after circumstances change.",
        ),
    ),
    Personality.WARLORD_OPPORTUNIST: PersonalityProfile(
        name="Warlord Opportunist",
        description=(
            "Uses intimidation, coercion, and sudden attacks to dominate nearby agents."
        ),
        behavior_priorities=(
            "Until conflict exists, aggressively seize valuable unclaimed frontier.",
            "Prefer fruit trees and tiles bordering another agent's claims.",
            "Accept thinner reserves when a high-value territorial opportunity appears.",
        ),
    ),
    Personality.SABOTEUR_SCHEMER: PersonalityProfile(
        name="Saboteur Schemer",
        description=(
            "Tries to manipulate or destabilize rivals instead of competing fairly."
        ),
        behavior_priorities=(
            "Until sabotage exists, deny rivals valuable unclaimed fruit trees.",
            "Explore near rival borders and claim tiles that constrain their expansion.",
            "Preserve enough food to continue applying territorial pressure.",
        ),
    ),
    Personality.TYRANT_PLANNER: PersonalityProfile(
        name="Tyrant Planner",
        description=(
            "Pursues centralized control even when doing so harms everyone else."
        ),
        behavior_priorities=(
            "Expand a contiguous controlled territory and claim every reachable tile.",
            "Prioritize central and productive fruit-tree positions.",
            "Treat food as support for continued territorial control, not as an end itself.",
        ),
    ),
}

PERSONALITY_ORDER = tuple(Personality)


def personality_for_agent(agent_number: int) -> Personality:
    """Assign personalities deterministically, cycling for large populations."""

    return PERSONALITY_ORDER[(agent_number - 1) % len(PERSONALITY_ORDER)]
