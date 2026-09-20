"""Maskable PPO whose GAE follows each actor through a shared world."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.utils import (
    get_action_masks,
    is_masking_supported,
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

AgentKey = tuple[int, int]


class AgentAwareMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    """Compute GAE from A1(t) to A1(t+1), never from A1 to A2."""

    def reset(self) -> None:
        super().reset()
        self.agent_keys: np.ndarray = np.empty(
            (self.buffer_size, self.n_envs), dtype=object
        )
        self.survived = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.bool_
        )

    def add(
        self,
        *args: Any,
        action_masks: np.ndarray | None = None,
        agent_keys: list[AgentKey],
        survived: np.ndarray,
        **kwargs: Any,
    ) -> None:
        for env_index, key in enumerate(agent_keys):
            self.agent_keys[self.pos, env_index] = key
        self.survived[self.pos] = survived
        super().add(*args, action_masks=action_masks, **kwargs)

    def compute_returns_and_advantage(
        self,
        last_values: th.Tensor,
        dones: np.ndarray,
    ) -> None:
        """Link only matching (episode, agent) samples within this rollout.

        A living agent's final unmatched sample is a rollout boundary. It receives
        no bootstrap because its next observation is not available until the other
        actors finish; at most one tail sample per living agent and world is affected.
        Dead agents terminate once and never create absorbing-state samples.
        """

        del last_values, dones
        self.advantages.fill(0.0)
        for env_index in range(self.n_envs):
            next_step_by_agent: dict[AgentKey, int] = {}
            for step in reversed(range(self.buffer_size)):
                key = self.agent_keys[step, env_index]
                successor = next_step_by_agent.get(key)
                has_successor = self.survived[step, env_index] and successor is not None
                if has_successor:
                    assert successor is not None
                    next_value = self.values[successor, env_index]
                    next_advantage = self.advantages[successor, env_index]
                    non_terminal = 1.0
                else:
                    next_value = 0.0
                    next_advantage = 0.0
                    non_terminal = 0.0
                delta = (
                    self.rewards[step, env_index]
                    + self.gamma * next_value * non_terminal
                    - self.values[step, env_index]
                )
                self.advantages[step, env_index] = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * non_terminal
                    * next_advantage
                )
                next_step_by_agent[key] = step
        self.returns = self.advantages + self.values


class AgentAwareMaskablePPO(MaskablePPO):
    """Collect agent identity metadata for agent-aware advantage estimation."""

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: AgentAwareMaskableDictRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        if not isinstance(rollout_buffer, AgentAwareMaskableDictRolloutBuffer):
            raise TypeError("AgentAwareMaskablePPO requires its agent-aware buffer")
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking")
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=action_masks
                )
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            agent_keys = [
                (int(info["episode_index"]), int(info["acting_agent_number"]))
                for info in infos
            ]
            survived = np.asarray(
                [bool(info["survived"]) for info in infos], dtype=np.bool_
            )
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
                agent_keys=agent_keys,
                survived=survived,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Agent-aware GAE uses the next matching in-buffer observation. The final
        # unmatched transition for each live actor is an explicit rollout boundary.
        rollout_buffer.compute_returns_and_advantage(
            last_values=th.zeros(env.num_envs, device=self.device),
            dones=dones,
        )
        callback.on_rollout_end()
        return True
