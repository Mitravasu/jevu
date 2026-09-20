"""Load a trained RL model behind JevU's action-selector interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from jevu.actions import TurnActions
from jevu.agent import AgentState
from jevu.rl.action_codec import action_mapping, action_masks, decode_actions
from jevu.rl.bundle import (
    BUNDLE_VERSION,
    METADATA_FILENAME,
    MODEL_FILENAME,
    game_rules_metadata,
)
from jevu.rl.device import resolve_device
from jevu.rl.observation import (
    OBSERVATION_SCHEMA_VERSION,
    ObservationSpec,
    encode_observation,
)


class RLActionSelector:
    """Choose typed JevU actions with a saved MaskablePPO policy."""

    def __init__(
        self,
        model: MaskablePPO,
        observation_spec: ObservationSpec,
        *,
        deterministic: bool = True,
    ) -> None:
        self._model = model
        self._observation_spec = observation_spec
        self._deterministic = deterministic

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        deterministic: bool = True,
        device: str = "auto",
    ) -> RLActionSelector:
        bundle = Path(bundle_path)
        metadata_path = bundle / METADATA_FILENAME
        model_path = bundle / MODEL_FILENAME
        if not metadata_path.is_file() or not model_path.is_file():
            raise ValueError(f"Incomplete RL model bundle: {bundle}")
        metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        if metadata.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError("Unsupported RL model bundle version")
        if metadata.get("status") != "complete":
            raise ValueError("RL model bundle has not completed its reload smoke test")
        if metadata.get("algorithm") != "maskable_ppo":
            raise ValueError("Unsupported RL model algorithm")
        observation = metadata.get("observation_schema", {})
        if observation.get("version") != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                "Incompatible observation schema: expected egocentric version "
                f"{OBSERVATION_SCHEMA_VERSION}, got {observation.get('version')}; "
                "retrain this model with the current trainer"
            )
        if metadata.get("action_mapping") != action_mapping():
            raise ValueError(
                "The model action mapping is incompatible with this JevU version"
            )
        if metadata.get("game_rules") != game_rules_metadata():
            raise ValueError("The model was trained with incompatible game rules")
        spec = ObservationSpec(
            max_turns=int(observation["max_turns"]),
        )
        model = MaskablePPO.load(model_path, device=resolve_device(device))
        return cls(model, spec, deterministic=deterministic)

    def __call__(self, state: AgentState) -> TurnActions:
        observation = encode_observation(state, self._observation_spec)
        masks = action_masks(
            state,
            state.world_width,
            state.world_height,
        )
        action, _ = self._model.predict(
            observation,
            deterministic=self._deterministic,
            action_masks=masks,
        )
        return decode_actions(action)

    @property
    def observation_spec(self) -> ObservationSpec:
        """Return the policy's world-size-independent observation schema."""

        return self._observation_spec
