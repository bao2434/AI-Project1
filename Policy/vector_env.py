"""Vectorized parallel game environments for PPO rollout collection."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import torch

from Arena import Environment
from Agents import *
from .PPO_Policy import PPO_Policy, encode_observation
from .opponent_pool import OpponentPool


class ParallelGameEnvs:
    """Manages 16 parallel game environments with mixed opponents."""

    def __init__(
        self,
        num_envs: int = 16,
        opponent_config: Optional[Dict[str, int]] = None,
        opponent_pool: Optional[OpponentPool] = None,
        max_turns: int = 500,
        symmetric_self_play: bool = True,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        draw_reward: float = 0.0,
        step_penalty: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_envs = num_envs
        self.max_turns = max_turns
        self.symmetric_self_play = symmetric_self_play
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.step_penalty = step_penalty
        self.device = device
        self.opponent_pool = opponent_pool or OpponentPool(device=device)

        if opponent_config is None:
            # Default distribution from PPO_TRAINING.md Section 3
            opponent_config = {
                "self": 8,
                "shortest_path": 4,
                "checkpoint_pool": 4,
            }
        self.opponent_config = opponent_config

        # Build list of opponent types per environment (total == num_envs)
        self.env_opponent_types: List[str] = []
        for opp_type, count in opponent_config.items():
            self.env_opponent_types.extend([opp_type] * count)

        # Pad or trim if doesn't match num_envs exactly
        if len(self.env_opponent_types) < num_envs:
            self.env_opponent_types.extend(["self"] * (num_envs - len(self.env_opponent_types)))
        elif len(self.env_opponent_types) > num_envs:
            self.env_opponent_types = self.env_opponent_types[:num_envs]

        # Initialize environment states
        self.envs: List[Environment] = [
            Environment(max_turns=max_turns, starting_player=0)
            for _ in range(num_envs)
        ]
        # Which player is the learning agent in this environment (0 or 1)
        self.learner_players: List[int] = [0 for _ in range(num_envs)]
        # Active opponent functions for non-self environments
        self.active_opponents: List[Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]] = [
            None for _ in range(num_envs)
        ]

        self.game_ids: List[int] = [0 for _ in range(num_envs)]
        # In-progress transitions per player: env_idx -> {0: [...], 1: [...]}
        self.active_trajectories: List[Dict[int, List[Dict[str, Any]]]] = [
            {0: [], 1: []} for _ in range(num_envs)
        ]

        # Metrics collected during games: env_idx -> list of game outcome dicts
        self.completed_game_stats: List[Dict[str, Any]] = []

        # Setup initial opponents
        for env_idx in range(num_envs):
            self._setup_env_for_new_game(env_idx, is_init=True)

    def _create_opponent_callable(self, opp_type: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Instantiate an opponent policy callable by name."""
        if opp_type in ("self", "self_play"):
            return None
        if opp_type == "hard_waller":
            return HardWaller()
        if opp_type == "mcts" or opp_type == "mcts_v7":
            return MCTS_Agent_v7()
        if opp_type == "quoridor_mcts":
            return QuoridorMCTSAgent()
        if opp_type in ("minimax", "minimax_summary"):
            return MinimaxAgentSummary()
        if opp_type in ("wrongway", "wrong_way_bot"):
            return WrongWayBot()
        if opp_type == "checkpoint_pool":
            return self.opponent_pool.get_opponent_fn()
        # Fallback to shortest path
        return HardWaller()

    def _setup_env_for_new_game(self, env_idx: int, is_init: bool = False) -> None:
        """Reset environment and configure starting player and opponent."""
        opp_type = self.env_opponent_types[env_idx]
        if is_init:
            learner = 0
        else:
            # Alternate learner player to avoid first-player bias
            learner = 1 - self.learner_players[env_idx]

        self.learner_players[env_idx] = learner
        self.active_opponents[env_idx] = self._create_opponent_callable(opp_type)
        self.envs[env_idx] = Environment(max_turns=self.max_turns, starting_player=0)
        self.game_ids[env_idx] += 1
        self.active_trajectories[env_idx] = {0: [], 1: []}

        # If opponent starts first, advance through opponent's turn immediately
        if opp_type != "self":
            self._advance_opponent_turns(env_idx)

    def _advance_opponent_turns(self, env_idx: int) -> None:
        """Play opponent turns until it is the learner's turn or game ends."""
        env = self.envs[env_idx]
        opp_fn = self.active_opponents[env_idx]
        learner = self.learner_players[env_idx]

        while not env.done and env.current_player != learner:
            obs = env.get_observation(env.current_player)
            action = opp_fn(obs) if opp_fn is not None else {"type": "pass"}
            env.step(action)

        if env.done:
            self._handle_game_over(env_idx)

    def _handle_game_over(self, env_idx: int) -> None:
        """Assign outcome rewards and finalize trajectories when an episode finishes."""
        env = self.envs[env_idx]
        opp_type = self.env_opponent_types[env_idx]
        winner = env.winner
        turns = env.turn_count

        # Record game stats
        is_self = (opp_type == "self")
        learner = self.learner_players[env_idx]
        if is_self:
            outcome = "draw" if winner is None else f"player_{winner}"
        else:
            if winner == learner:
                outcome = "win"
            elif winner is None:
                outcome = "draw"
            else:
                outcome = "loss"

        self.completed_game_stats.append({
            "env_idx": env_idx,
            "opponent_type": opp_type,
            "winner": winner,
            "outcome": outcome,
            "turns": turns,
            "reason": env.termination_reason,
        })

        # Finalize transitions for each participating player
        traj_dict = self.active_trajectories[env_idx]

        if is_self and self.symmetric_self_play:
            players_to_finalize = [0, 1]
        else:
            players_to_finalize = [learner]

        for p in players_to_finalize:
            p_traj = traj_dict[p]
            if p_traj:
                last_trans = p_traj[-1]
                if winner == p:
                    reward = self.win_reward
                elif winner is None:
                    reward = self.draw_reward
                else:
                    reward = self.loss_reward
                last_trans["reward"] = reward
                last_trans["done"] = True

    def collect_rollout(
        self,
        model: PPO_Policy,
        steps_per_env: int = 512,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Collect steps_per_env learning transitions from each parallel environment.

        Returns
        -------
        all_completed_trajectories:
            List of completed trajectory lists with GAE advantages/returns ready.
        stats:
            Rollout statistics (game counts, win rates, etc.)
        """
        model.eval()

        # Completed trajectories from all environments: [(traj_list, bootstrap_val), ...]
        collected_trajectories: List[Tuple[List[Dict[str, Any]], float]] = []

        # Track transitions collected per environment during this rollout
        env_step_counts = [0 for _ in range(self.num_envs)]
        total_target_transitions = self.num_envs * steps_per_env

        self.completed_game_stats.clear()

        while any(count < steps_per_env for count in env_step_counts):
            # Gather environments that still need transitions and have an active learner turn
            ready_env_indices: List[int] = []
            obs_list: List[Dict[str, Any]] = []

            for env_idx in range(self.num_envs):
                if env_step_counts[env_idx] >= steps_per_env:
                    continue

                env = self.envs[env_idx]
                if env.done:
                    # Trajectories were finalized in _handle_game_over; flush them
                    for p in (0, 1):
                        if self.active_trajectories[env_idx][p]:
                            collected_trajectories.append((self.active_trajectories[env_idx][p], 0.0))
                            self.active_trajectories[env_idx][p] = []
                    self._setup_env_for_new_game(env_idx)
                    env = self.envs[env_idx]

                # Check if it's currently a learning agent's turn
                opp_type = self.env_opponent_types[env_idx]
                curr_player = env.current_player

                if opp_type == "self":
                    # Self-play: learner is active (symmetric or one-sided)
                    if self.symmetric_self_play or curr_player == self.learner_players[env_idx]:
                        ready_env_indices.append(env_idx)
                        obs_list.append(env.get_observation(curr_player))
                else:
                    # Playing against fixed opponent: ensure opponent has finished their turn
                    if curr_player != self.learner_players[env_idx]:
                        self._advance_opponent_turns(env_idx)
                        if self.envs[env_idx].done:
                            for p in (0, 1):
                                if self.active_trajectories[env_idx][p]:
                                    collected_trajectories.append((self.active_trajectories[env_idx][p], 0.0))
                                    self.active_trajectories[env_idx][p] = []
                            self._setup_env_for_new_game(env_idx)
                            continue

                    if not self.envs[env_idx].done and self.envs[env_idx].current_player == self.learner_players[env_idx]:
                        ready_env_indices.append(env_idx)
                        obs_list.append(self.envs[env_idx].get_observation(self.learner_players[env_idx]))

            if not ready_env_indices:
                continue

            # Batch encode observations for all ready environments
            batch_boards = []
            batch_globals = []
            batch_masks = []

            for obs in obs_list:
                b, g, m = encode_observation(obs, device=self.device)
                batch_boards.append(b.squeeze(0))
                batch_globals.append(g.squeeze(0))
                batch_masks.append(m.squeeze(0))

            boards_tensor = torch.stack(batch_boards).to(self.device)
            globals_tensor = torch.stack(batch_globals).to(self.device)
            masks_tensor = torch.stack(batch_masks).to(self.device)

            # Batched model inference without gradients
            with torch.no_grad():
                results = model.act(
                    boards_tensor,
                    globals_tensor,
                    masks_tensor,
                    deterministic=False,
                )

            # Process actions for each environment
            for i, env_idx in enumerate(ready_env_indices):
                action_idx = int(results["action"][i].item())
                log_prob = float(results["log_prob"][i].item())
                value = float(results["value"][i].item())

                env = self.envs[env_idx]
                curr_player = env.current_player
                action_dict = PPO_Policy.decode_action(action_idx)

                # Record transition
                transition = {
                    "board": batch_boards[i].cpu(),
                    "global_features": batch_globals[i].cpu(),
                    "action_mask": batch_masks[i].cpu(),
                    "action": action_idx,
                    "old_log_prob": log_prob,
                    "value": value,
                    "reward": self.step_penalty,
                    "done": False,
                    "truncated": False,
                    "player_id": curr_player,
                    "game_id": self.game_ids[env_idx],
                }

                self.active_trajectories[env_idx][curr_player].append(transition)
                env_step_counts[env_idx] += 1

                # Advance environment
                env.step(action_dict)

                if env.done:
                    self._handle_game_over(env_idx)
                    for p in (0, 1):
                        if self.active_trajectories[env_idx][p]:
                            collected_trajectories.append((self.active_trajectories[env_idx][p], 0.0))
                            self.active_trajectories[env_idx][p] = []
                else:
                    opp_type = self.env_opponent_types[env_idx]
                    if opp_type != "self" and env.current_player != self.learner_players[env_idx]:
                        self._advance_opponent_turns(env_idx)
                        if env.done:
                            for p in (0, 1):
                                if self.active_trajectories[env_idx][p]:
                                    collected_trajectories.append((self.active_trajectories[env_idx][p], 0.0))
                                    self.active_trajectories[env_idx][p] = []

        # Handle rollout cutoff for in-progress games across all environments
        for env_idx in range(self.num_envs):
            env = self.envs[env_idx]
            if not env.done:
                for p in (0, 1):
                    p_traj = self.active_trajectories[env_idx][p]
                    if p_traj:
                        p_traj[-1]["truncated"] = True
                        # Bootstrap from next state value
                        next_obs = env.get_observation(p)
                        with torch.no_grad():
                            nb, ng, nm = encode_observation(next_obs, device=self.device)
                            _, v_boot = model(nb, ng)
                            bootstrap_val = float(v_boot.item())
                        collected_trajectories.append((p_traj, bootstrap_val))
                        self.active_trajectories[env_idx][p] = []

        # Calculate statistics
        rollout_stats = self._summarize_stats()

        return collected_trajectories, rollout_stats

    def _summarize_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for games completed during the rollout."""
        total_games = len(self.completed_game_stats)
        if total_games == 0:
            return {"total_games": 0}

        by_opponent: Dict[str, Dict[str, Any]] = {}
        for game in self.completed_game_stats:
            opp = game["opponent_type"]
            if opp not in by_opponent:
                by_opponent[opp] = {"wins": 0, "losses": 0, "draws": 0, "turns": []}

            outcome = game["outcome"]
            if outcome == "win":
                by_opponent[opp]["wins"] += 1
            elif outcome == "loss":
                by_opponent[opp]["losses"] += 1
            elif outcome == "draw":
                by_opponent[opp]["draws"] += 1
            else:
                # self-play
                by_opponent[opp]["wins"] += 1  # someone won or draw

            by_opponent[opp]["turns"].append(game["turns"])

        summary = {
            "total_games": total_games,
            "by_opponent": {},
        }
        for opp, d in by_opponent.items():
            n = len(d["turns"])
            summary["by_opponent"][opp] = {
                "games": n,
                "win_rate": d["wins"] / n if n > 0 else 0.0,
                "draw_rate": d["draws"] / n if n > 0 else 0.0,
                "loss_rate": d["losses"] / n if n > 0 else 0.0,
                "avg_turns": sum(d["turns"]) / n if n > 0 else 0.0,
            }

        return summary
