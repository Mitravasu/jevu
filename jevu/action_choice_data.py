"""Model-facing descriptions of the actions Jev can select."""

from jevu.actions import ExploreAction, InteractAction
from jevu.rules import (
    FOOD_EAT_COST,
    FOOD_HARVEST_AMOUNT,
    FOOD_HUNGER_RESTORE,
    FRUIT_TREE_COOLDOWN,
    MAX_HUNGER,
)

INTERACT_INSTRUCTIONS = (
    "Which interaction should the agent take before moving? Use `goals` and "
    "`agent_state`."
)

INTERACT_CRITERIA = {
    InteractAction.HARVEST: (
        f"Collect {FOOD_HARVEST_AMOUNT} food when "
        "`agent_state.current_tile` is `fruit_tree`; on any other tile "
        "or when `agent_state.current_tile_cooldown` is above 0, this has "
        f"no effect. A harvested tree cools down for {FRUIT_TREE_COOLDOWN} "
        "complete turns."
    ),
    InteractAction.EAT: (
        f"Consume {FOOD_EAT_COST} food to restore "
        f"{FOOD_HUNGER_RESTORE} hunger points when "
        f"`agent_state.inventory.food` is at least {FOOD_EAT_COST} and "
        f"hunger is below {MAX_HUNGER}; otherwise this has no effect."
    ),
    InteractAction.CLAIM: (
        "Claim the current blank or fruit-tree tile when "
        "`agent_state.current_tile_claim` is null. A claimed tile cannot "
        "be claimed again by any agent."
    ),
    InteractAction.NONE: (
        "Do nothing when harvesting, eating, and claiming would all have no "
        "useful effect."
    ),
}

EXPLORE_INSTRUCTIONS = (
    "Which direction should the agent move after interacting? Use `goals`, "
    "`agent_state.adjacent_tiles`, and `agent_state.adjacent_tile_claims`; a "
    "missing direction is a world boundary and leaves the agent in place."
)

EXPLORE_CRITERIA = {
    action: (
        "Remain on the current tile only when every available move is less useful. "
        "Staying does not conserve hunger and cannot discover or claim new tiles."
        if action is ExploreAction.STAY
        else f"Move one tile {action.value}."
    )
    for action in ExploreAction
}
