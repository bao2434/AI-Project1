"""Opponent pool for sampling past frozen checkpoints during PPO training."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional
import torch

from .PPO_Policy import PPO_Policy, encode_observation


class OpponentPool:
    """Stores past frozen policy checkpoints for league / self-play training."""

    def __init__(
        self,
        max_size: int = 10,
        device: torch.device = torch.device("cpu"),
        fallback_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.max_size = max_size
        self.device = device
        self.fallback_fn = fallback_fn
        self.snapshots: List[Dict[str, torch.Tensor]] = []
        # Pre-instantiated frozen model for running checkpoint inference
        self._checkpoint_model: Optional[PPO_Policy] = None

    def __len__(self) -> int:
        return len(self.snapshots)

    def add_snapshot(self, model: PPO_Policy) -> None:
        """Add a CPU state_dict copy of the current policy to the pool."""
        cpu_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        self.snapshots.append(cpu_state)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def get_opponent_fn(
        self,
        snapshot_index: Optional[int] = None,
        deterministic: bool = False,
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Return a callable (observation) -> action_dict using a frozen checkpoint."""
        if not self.snapshots:
            if self.fallback_fn is not None:
                return self.fallback_fn
            # If no snapshot or fallback, default to random pass/moves
            from Agents import ShortestPathAgent
            return ShortestPathAgent()

        if snapshot_index is None:
            snapshot_index = random.randint(0, len(self.snapshots) - 1)

        state_dict = self.snapshots[snapshot_index]

        if self._checkpoint_model is None:
            self._checkpoint_model = PPO_Policy().to(self.device)
            self._checkpoint_model.eval()

        # Load weights into checkpoint model
        self._checkpoint_model.load_state_dict(state_dict)
        self._checkpoint_model.eval()

        def agent_action(obs: Dict[str, Any]) -> Dict[str, Any]:
            with torch.no_grad():
                board, global_features, action_mask = encode_observation(
                    obs, device=self.device
                )
                res = self._checkpoint_model.act(
                    board,
                    global_features,
                    action_mask,
                    deterministic=deterministic,
                )
                action_idx = int(res["action"].item())
                return PPO_Policy.decode_action(action_idx)

        return agent_action
