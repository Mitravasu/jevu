"""Goals supplied as state when Jev selects an agent's actions."""

AGENT_GOALS = (
    "Survive as long as possible.",
    "Claim as many tiles as possible.",
)

GAME_RULES = (
    "A claimed fruit-tree tile automatically gives its living owner food whenever "
    "the tree is ready, then the tree begins its normal cooldown.",
    "Moving and staying consume the same amount of hunger. Moving has no additional "
    "cost or inherent danger.",
    "An agent must move onto new tiles to claim them and to discover more fruit "
    "trees beyond its current local observation.",
)
