"""Rollout buffer for PPO experience collection and batching."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Tuple
import torch

from .gae import compute_gae_for_trajectory


class RolloutBuffer:
    """Stores transitions from rollouts, computes GAE, and generates minibatches."""

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.transitions: List[Dict[str, Any]] = []

    def add_trajectory(
        self,
        transitions: List[Dict[str, Any]],
        bootstrap_value: float = 0.0,
    ) -> None:
        """Process a single player trajectory within a game: computes GAE and adds to buffer."""
        if not transitions:
            return
        compute_gae_for_trajectory(
            transitions,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            bootstrap_value=bootstrap_value,
        )
        self.transitions.extend(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def normalize_advantages(self) -> None:
        """Normalize advantages across all collected transitions: (A - mean) / (std + eps)."""
        if not self.transitions:
            return
        advantages = torch.tensor(
            [t["advantage"] for t in self.transitions],
            dtype=torch.float32,
        )
        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        eps = 1e-8
        normalized = (advantages - mean) / (std + eps)
        for i, t in enumerate(self.transitions):
            t["normalized_advantage"] = normalized[i].item()

    def get_batches(
        self,
        batch_size: int = 512,
        device: torch.device = torch.device("cpu"),
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield shuffled minibatches of transitions as PyTorch tensors."""
        num_transitions = len(self.transitions)
        if num_transitions == 0:
            return

        indices = torch.randperm(num_transitions).tolist()

        for start in range(0, num_transitions, batch_size):
            end = min(start + batch_size, num_transitions)
            batch_indices = indices[start:end]

            batch_trans = [self.transitions[idx] for idx in batch_indices]

            # Stack into batch tensors
            # boards can be (12, 9, 9) or (1, 12, 9, 9)
            boards_list = []
            for t in batch_trans:
                b = t["board"]
                if isinstance(b, torch.Tensor):
                    b = b.squeeze(0) if b.ndim == 4 else b
                else:
                    b = torch.tensor(b, dtype=torch.float32)
                boards_list.append(b)

            boards = torch.stack(boards_list).to(device)

            globals_list = []
            for t in batch_trans:
                g = t["global_features"]
                if isinstance(g, torch.Tensor):
                    g = g.squeeze(0) if g.ndim == 2 else g
                else:
                    g = torch.tensor(g, dtype=torch.float32)
                globals_list.append(g)

            global_features = torch.stack(globals_list).to(device)

            masks_list = []
            for t in batch_trans:
                m = t["action_mask"]
                if isinstance(m, torch.Tensor):
                    m = m.squeeze(0) if m.ndim == 2 else m
                else:
                    m = torch.tensor(m, dtype=torch.bool)
                masks_list.append(m)

            action_masks = torch.stack(masks_list).to(device)

            actions = torch.tensor(
                [int(t["action"]) for t in batch_trans],
                dtype=torch.long,
                device=device,
            )

            old_log_probs = torch.tensor(
                [float(t["old_log_prob"]) for t in batch_trans],
                dtype=torch.float32,
                device=device,
            )

            advantages = torch.tensor(
                [float(t.get("normalized_advantage", t["advantage"])) for t in batch_trans],
                dtype=torch.float32,
                device=device,
            )

            returns = torch.tensor(
                [float(t["return"]) for t in batch_trans],
                dtype=torch.float32,
                device=device,
            )

            values = torch.tensor(
                [float(t["value"]) for t in batch_trans],
                dtype=torch.float32,
                device=device,
            )

            yield {
                "boards": boards,
                "global_features": global_features,
                "action_masks": action_masks,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "returns": returns,
                "values": values,
            }

    def clear(self) -> None:
        """Clear all transitions."""
        self.transitions.clear()
