"""Generalized Advantage Estimation (GAE) computation for PPO."""

from __future__ import annotations

from typing import List, Dict, Any


def compute_gae_for_trajectory(
    transitions: List[Dict[str, Any]],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_value: float = 0.0,
) -> None:
    """Calculate GAE advantages and return targets in-place for a trajectory.

    Parameters
    ----------
    transitions:
        List of transition dicts belonging to ONE player in ONE game, in chronological order.
        Each transition must have:
          - 'reward': float
          - 'value': float (critic estimate V(s_t))
          - 'done': bool
          - 'truncated': bool
    gamma:
        Discount factor (default 0.99).
    gae_lambda:
        GAE lambda parameter (default 0.95).
    bootstrap_value:
        Value estimate V(s_{T}) for the next state after the last transition.
        Must be 0.0 for terminal episodes, and V(s_next) for nonterminal cutoffs.
    """
    if not transitions:
        return

    num_steps = len(transitions)
    last_gae = 0.0

    # Backwards recursion through the trajectory
    for t in reversed(range(num_steps)):
        reward = float(transitions[t]["reward"])
        v_current = float(transitions[t]["value"])
        is_done = bool(transitions[t].get("done", False))

        if t == num_steps - 1:
            v_next = 0.0 if is_done else float(bootstrap_value)
        else:
            v_next = float(transitions[t + 1]["value"])

        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = reward + gamma * v_next - v_current

        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        # If t is terminal, it does not receive advantage from any future step
        if is_done:
            advantage = delta
        else:
            advantage = delta + gamma * gae_lambda * last_gae

        last_gae = advantage

        # Return target: R_t = A_t + V(s_t)
        return_target = advantage + v_current

        transitions[t]["advantage"] = advantage
        transitions[t]["return"] = return_target
