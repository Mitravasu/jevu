"""Gymnasium adapter over the authoritative JevU simulation."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jevu.actions import ExploreAction, InteractAction
from jevu.game_state import ActionLogEntry, GameState, TurnSession
from jevu.rl.action_codec import ACTION_SHAPE, action_masks, decode_actions
from jevu.rl.config import EnvironmentConfig, RewardConfig
from jevu.rl.observation import (
    ObservationSpec,
    empty_observation,
    encode_observation,
)
from jevu.rl.reward import TransitionEvents, calculate_reward
from jevu.world import WorldConfig


def _ineffective_action_count(
    entry: ActionLogEntry,
    newly_claimed_tiles: int,
) -> int:
    ineffective = 0
    if entry.actions.interact is InteractAction.HARVEST:
        ineffective += int(entry.end_food <= entry.start_food)
    elif entry.actions.interact is InteractAction.EAT:
        ineffective += int(entry.end_food >= entry.start_food)
    elif entry.actions.interact is InteractAction.CLAIM:
        ineffective += int(newly_claimed_tiles == 0)
    if (
        entry.actions.explore is not ExploreAction.STAY
        and entry.end_position == entry.start_position
    ):
        ineffective += 1
    return ineffective


class JevUGymEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """A shared-policy episode that cycles through agents in one world."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        environment_config: EnvironmentConfig,
        reward_config: RewardConfig,
    ) -> None:
        self.environment_config = environment_config
        self.reward_config = reward_config
        self.observation_spec = ObservationSpec(
            max_turns=environment_config.max_turns,
        )
        self.observation_space = self.observation_spec.space()
        self.action_space = spaces.MultiDiscrete(np.asarray(ACTION_SHAPE))
        self.game_state: GameState | None = None
        self._turn_session: TurnSession | None = None
        self._agent_decisions = 0
        self._episode_reward = 0.0
        self._reward_components = {
            "survival": 0.0,
            "territory": 0.0,
            "death": 0.0,
            "ineffective_action": 0.0,
        }
        self._world_seed: int | None = None
        self._episode_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._episode_index = 0
        self._episode_index += 1
        self._world_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.game_state = GameState.create(
            WorldConfig(
                width=self.environment_config.width,
                height=self.environment_config.height,
                fruit_tree_density=self.environment_config.fruit_tree_density,
                seed=self._world_seed,
            ),
            agent_count=self.environment_config.agent_count,
        )
        self._turn_session = self.game_state.start_turn()
        self._agent_decisions = 0
        self._episode_reward = 0.0
        self._reward_components = {
            "survival": 0.0,
            "territory": 0.0,
            "death": 0.0,
            "ineffective_action": 0.0,
        }
        return self._observation(), {
            "episode_index": self._episode_index,
            "world_seed": self._world_seed,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.game_state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._turn_session is None or self._turn_session.complete:
            raise RuntimeError("step() cannot be called after an episode has ended")

        actions = decode_actions(action)
        agent_state = self._turn_session.current_state()
        agent_number = self._turn_session.current_agent.number
        claims_before = sum(
            owner == agent_number for owner in self._turn_session.tile_claims.values()
        )
        turn_result = self._turn_session.apply(actions)
        claims_after = sum(
            owner == agent_number for owner in self._turn_session.tile_claims.values()
        )
        newly_claimed = claims_after - claims_before
        survived = turn_result.survived
        events = TransitionEvents(
            survived=survived,
            newly_claimed_tiles=newly_claimed,
            ineffective_actions=_ineffective_action_count(
                turn_result.entry,
                newly_claimed,
            ),
        )
        breakdown = calculate_reward(events, self.reward_config)
        reward = breakdown.total
        self._episode_reward += reward
        for key, value in breakdown.to_dict().items():
            if key != "total":
                self._reward_components[key] += value
        self._agent_decisions += 1

        turn_complete = self._turn_session.complete
        terminated = False
        truncated = False
        if turn_complete:
            self.game_state = self._turn_session.finish()
            terminated = not self.game_state.agents
            truncated = (
                not terminated
                and self.game_state.turn >= self.environment_config.max_turns
            )
            if not terminated and not truncated:
                self._turn_session = self.game_state.start_turn()

        info: dict[str, Any] = {
            "episode_index": self._episode_index,
            "world_seed": self._world_seed,
            "acting_agent_id": agent_state.id,
            "acting_agent_number": agent_number,
            "world_turn": turn_result.entry.turn,
            "reward_components": breakdown.to_dict(),
            "newly_claimed_tiles": newly_claimed,
            "agent_claimed_tiles": claims_after,
            "survived": survived,
        }
        if terminated or truncated:
            assert self.game_state is not None
            claims_by_agent = {
                f"A{agent.number}": sum(
                    owner == agent.number
                    for owner in self.game_state.tile_claims.values()
                )
                for agent in self.game_state.agents
            }
            info["episode_summary"] = {
                "reward": self._episode_reward,
                "length": self._agent_decisions,
                "agent_decisions": self._agent_decisions,
                "world_turns": self.game_state.turn,
                "claimed_tiles": len(self.game_state.tile_claims),
                "claims_by_survivor": claims_by_agent,
                "survivors": len(self.game_state.agents),
                "terminal_cause": ("all_agents_dead" if terminated else "max_turns"),
                "reward_components": dict(self._reward_components),
                "world_seed": self._world_seed,
            }
        observation = (
            empty_observation(self.observation_spec)
            if terminated
            else self._observation()
        )
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._turn_session is None or self._turn_session.complete:
            return np.ones(sum(ACTION_SHAPE), dtype=np.bool_)
        state = self._turn_session.current_state()
        return action_masks(
            state,
            self.environment_config.width,
            self.environment_config.height,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        if self._turn_session is None or self._turn_session.complete:
            return empty_observation(self.observation_spec)
        state = self._turn_session.current_state()
        return encode_observation(state, self.observation_spec)
