"""Policy evaluation against benchmark opponents."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import torch

from Arena import run_simulation
from Agents.baselines import RandomAgent, ShortestPathAgent, GreedyWallAgent
from .PPO_Policy import PPO_Policy, encode_observation


def create_agent_fn(
    model: PPO_Policy,
    device: torch.device = torch.device("cpu"),
    deterministic: bool = True,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a standalone callable from a PPO_Policy model."""
    model.eval()

    def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            b, g, m = encode_observation(obs, device=device)
            res = model.act(b, g, m, deterministic=deterministic)
            action_idx = int(res["action"].item())
            return PPO_Policy.decode_action(action_idx)

    return agent


def evaluate_policy(
    model: PPO_Policy,
    opponents: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
    num_games_per_opponent: int = 10,
    device: torch.device = torch.device("cpu"),
    deterministic: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a PPO policy against benchmark opponents.

    Parameters
    ----------
    model:
        Trained PPO_Policy model.
    opponents:
        Dictionary of name -> opponent callable. If None, evaluates against
        'ShortestPathAgent' and 'RandomAgent'.
    num_games_per_opponent:
        Total games to play against each opponent (split evenly between Player 0 and Player 1).
    device:
        Torch device for inference.
    deterministic:
        Whether to sample greedily (argmax logits).

    Returns
    -------
    Dictionary of opponent name -> metric dict (win_rate, loss_rate, draw_rate, avg_turns).
    """
    model.eval()
    ppo_fn = create_agent_fn(model, device=device, deterministic=deterministic)

    if opponents is None:
        opponents = {
            "ShortestPathAgent": ShortestPathAgent(),
            "RandomAgent": RandomAgent(),
        }

    results: Dict[str, Dict[str, float]] = {}

    for opp_name, opp_fn in opponents.items():
        wins = 0
        losses = 0
        draws = 0
        total_turns = 0

        games_as_p0 = num_games_per_opponent // 2
        games_as_p1 = num_games_per_opponent - games_as_p0

        # Games as Player 0
        for _ in range(games_as_p0):
            res = run_simulation(ppo_fn, opp_fn, max_turns=500)
            total_turns += res.total_turns
            if res.winner == 0:
                wins += 1
            elif res.winner == 1:
                losses += 1
            else:
                draws += 1

        # Games as Player 1
        for _ in range(games_as_p1):
            res = run_simulation(opp_fn, ppo_fn, max_turns=500)
            total_turns += res.total_turns
            if res.winner == 1:
                wins += 1
            elif res.winner == 0:
                losses += 1
            else:
                draws += 1

        n = num_games_per_opponent
        results[opp_name] = {
            "games": float(n),
            "win_rate": wins / n if n > 0 else 0.0,
            "loss_rate": losses / n if n > 0 else 0.0,
            "draw_rate": draws / n if n > 0 else 0.0,
            "avg_turns": total_turns / n if n > 0 else 0.0,
        }

    return results
