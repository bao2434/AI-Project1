"""Vectorized parallel game environments for PPO rollout collection.

Uses multiprocessing with one process per environment so that expensive opponent
searches (Minimax, MCTS, HardWaller) run concurrently across CPU cores without
blocking each other or the main training process.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

from Arena import Environment
from Agents import *
from .PPO_Policy import PPO_Policy, encode_observation
from .opponent_pool import OpponentPool


def _env_worker(
    pipe,
    env_idx: int,
    opp_type: str,
    max_turns: int,
    symmetric_self_play: bool,
):
    """Isolated environment worker running in its own OS process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    opp_fn = None
    if opp_type == "hard_waller":
        opp_fn = HardWaller()
    elif opp_type in ("wrongway", "wrong_way_bot"):
        opp_fn = WrongWayBot()
    elif opp_type in ("mcts", "mcts_v7"):
        opp_fn = MCTS_Agent_v7()
    elif opp_type in ("quoridor_mcts_fast", "quoridor_mcts"):
        opp_fn = QuoridorMCTSAgent()
    elif opp_type in ("minimax_d2", "minimax", "minimax_summary"):
        opp_fn = MinimaxAgentSummary()

    learner_player = 0
    env = Environment(max_turns=max_turns, starting_player=0)
    game_id = 0

    def advance_opponent():
        while not env.done and env.current_player != learner_player:
            obs = env.get_observation(env.current_player)
            action = opp_fn(obs) if opp_fn is not None else {"type": "pass"}
            env.step(action)

    def start_new_game(is_init: bool = False):
        nonlocal learner_player, env, game_id
        if not is_init:
            learner_player = 1 - learner_player
        else:
            learner_player = 0
        env = Environment(max_turns=max_turns, starting_player=0)
        game_id += 1
        if opp_type != "self":
            advance_opponent()

    start_new_game(is_init=True)

    curr_p = env.current_player
    pipe.send({
        "type": "init",
        "env_idx": env_idx,
        "obs": env.get_observation(curr_p),
        "curr_player": curr_p,
        "learner_player": learner_player,
        "done": env.done,
        "game_id": game_id,
    })

    while True:
        try:
            cmd = pipe.recv()
        except (EOFError, KeyboardInterrupt):
            break

        cmd_type = cmd.get("type")
        if cmd_type == "close":
            break

        elif cmd_type == "step":
            action_dict = cmd["action"]
            env.step(action_dict)

            # Advance opponent turns if playing fixed bot
            if opp_type != "self" and not env.done and env.current_player != learner_player:
                advance_opponent()

            game_ended = env.done
            winner = env.winner if game_ended else None
            turns = env.turn_count if game_ended else 0
            reason = env.termination_reason if game_ended else None
            ended_game_id = game_id

            if game_ended:
                start_new_game(is_init=False)

            curr_p = env.current_player
            pipe.send({
                "type": "step_result",
                "env_idx": env_idx,
                "obs": env.get_observation(curr_p),
                "curr_player": curr_p,
                "learner_player": learner_player,
                "done": game_ended,
                "winner": winner,
                "turns": turns,
                "reason": reason,
                "game_id": ended_game_id,
            })

        elif cmd_type == "get_obs":
            curr_p = env.current_player
            pipe.send({
                "type": "obs_result",
                "env_idx": env_idx,
                "obs": env.get_observation(curr_p),
                "curr_player": curr_p,
                "learner_player": learner_player,
            })


def build_opponent_schedule(opponent_config: Dict[str, int], num_envs: int) -> List[str]:
    """Distribute opponent types proportionally across num_envs."""
    total_cfg = sum(opponent_config.values())
    if total_cfg == num_envs:
        schedule = []
        for opp_type, count in opponent_config.items():
            schedule.extend([opp_type] * count)
        return schedule

    items = sorted(opponent_config.items(), key=lambda x: x[1], reverse=True)
    counts = {}
    allocated = 0
    for opp, weight in items:
        cnt = max(1, round(weight / total_cfg * num_envs))
        counts[opp] = cnt
        allocated += cnt

    while allocated > num_envs:
        reduced = False
        for opp, _ in items:
            if counts[opp] > 1:
                counts[opp] -= 1
                allocated -= 1
                reduced = True
                if allocated == num_envs:
                    break
        if not reduced:
            break

    while allocated < num_envs:
        counts[items[0][0]] += 1
        allocated += 1

    schedule = []
    for opp, cnt in counts.items():
        schedule.extend([opp] * cnt)

    return schedule[:num_envs]


class ParallelGameEnvs:
    """Manages parallel game environments running in separate OS processes.

    Avoids waiting on stragglers by continuously stepping any ready environment
    until the total transition target is reached.
    """

    def __init__(
        self,
        num_envs: int = 8,
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
            # Rebalanced default distribution (cheap bots dominant)
            opponent_config = {
                "self": 6,
                "wrong_way_bot": 3,
                "hard_waller": 3,
                "mcts_v7": 2,
                "quoridor_mcts_fast": 1,
                "minimax_d2": 1,
            }
        self.opponent_config = opponent_config
        self.env_opponent_types = build_opponent_schedule(opponent_config, num_envs)

        # Multiprocessing context
        self.ctx = mp.get_context("spawn")
        self.processes: List[mp.Process] = []
        self.parent_pipes: List[Any] = []
        self.worker_states: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self.waiting_for_worker: List[bool] = [False for _ in range(num_envs)]

        # In-progress transitions per env and player: env_idx -> {0: [...], 1: [...]}
        self.active_trajectories: List[Dict[int, List[Dict[str, Any]]]] = [
            {0: [], 1: []} for _ in range(num_envs)
        ]
        self.completed_game_stats: List[Dict[str, Any]] = []

        self._start_workers()

    def _start_workers(self) -> None:
        """Spawn worker processes and establish IPC pipes."""
        for env_idx in range(self.num_envs):
            parent_conn, child_conn = self.ctx.Pipe()
            opp_type = self.env_opponent_types[env_idx]
            proc = self.ctx.Process(
                target=_env_worker,
                args=(
                    child_conn,
                    env_idx,
                    opp_type,
                    self.max_turns,
                    self.symmetric_self_play,
                ),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)
            self.parent_pipes.append(parent_conn)

        # Receive init from all workers
        for env_idx in range(self.num_envs):
            msg = self.parent_pipes[env_idx].recv()
            self.worker_states[env_idx] = {
                "obs": msg["obs"],
                "curr_player": msg["curr_player"],
                "learner_player": msg["learner_player"],
                "game_id": msg["game_id"],
            }

    def _handle_game_over(self, env_idx: int, winner: Optional[int], turns: int, reason: Optional[str], collected_trajectories: List[Any]) -> None:
        """Assign outcome rewards and finalize trajectories when an episode finishes."""
        opp_type = self.env_opponent_types[env_idx]
        learner = self.worker_states[env_idx]["learner_player"]
        is_self = (opp_type == "self")

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
            "reason": reason,
        })

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
                collected_trajectories.append((p_traj, 0.0))
                traj_dict[p] = []

    def collect_rollout(
        self,
        model: PPO_Policy,
        steps_per_env: int = 1024,
        total_transitions: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Collect transitions across environments without waiting on stragglers.

        Fast environments contribute more steps, completely removing the idle-wait.
        Bookkeeping isolates each environment's GAE trajectory.
        """
        model.eval()
        target_total = total_transitions if total_transitions is not None else self.num_envs * steps_per_env

        collected_trajectories: List[Tuple[List[Dict[str, Any]], float]] = []
        self.completed_game_stats.clear()
        total_collected = 0

        while total_collected < target_total:
            # Check for completed steps from workers
            for env_idx in range(self.num_envs):
                if self.waiting_for_worker[env_idx]:
                    pipe = self.parent_pipes[env_idx]
                    if pipe.poll():
                        msg = pipe.recv()
                        self.waiting_for_worker[env_idx] = False
                        self.worker_states[env_idx]["obs"] = msg["obs"]
                        self.worker_states[env_idx]["curr_player"] = msg["curr_player"]
                        self.worker_states[env_idx]["learner_player"] = msg["learner_player"]
                        self.worker_states[env_idx]["game_id"] = msg.get("game_id", 0)

                        if msg["done"]:
                            self._handle_game_over(
                                env_idx,
                                msg["winner"],
                                msg["turns"],
                                msg["reason"],
                                collected_trajectories,
                            )

            # Identify all environments ready for a learner action
            ready_env_indices: List[int] = []
            obs_list: List[Dict[str, Any]] = []

            for env_idx in range(self.num_envs):
                if self.waiting_for_worker[env_idx]:
                    continue
                opp_type = self.env_opponent_types[env_idx]
                curr_p = self.worker_states[env_idx]["curr_player"]
                learner_p = self.worker_states[env_idx]["learner_player"]

                if opp_type == "self":
                    if self.symmetric_self_play or curr_p == learner_p:
                        ready_env_indices.append(env_idx)
                        obs_list.append(self.worker_states[env_idx]["obs"])
                else:
                    if curr_p == learner_p:
                        ready_env_indices.append(env_idx)
                        obs_list.append(self.worker_states[env_idx]["obs"])

            if not ready_env_indices:
                # Wait for at least one worker to send results
                active_pipes = [self.parent_pipes[i] for i in range(self.num_envs) if self.waiting_for_worker[i]]
                if active_pipes:
                    mp.connection.wait(active_pipes, timeout=0.01)
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

            with torch.no_grad():
                results = model.act(
                    boards_tensor,
                    globals_tensor,
                    masks_tensor,
                    deterministic=False,
                )

            # Record transitions and dispatch actions to workers
            for i, env_idx in enumerate(ready_env_indices):
                if total_collected >= target_total:
                    break

                action_idx = int(results["action"][i].item())
                log_prob = float(results["log_prob"][i].item())
                value = float(results["value"][i].item())

                curr_p = self.worker_states[env_idx]["curr_player"]
                action_dict = PPO_Policy.decode_action(action_idx)

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
                    "player_id": curr_p,
                    "game_id": self.worker_states[env_idx]["game_id"],
                }

                self.active_trajectories[env_idx][curr_p].append(transition)
                total_collected += 1

                self.parent_pipes[env_idx].send({"type": "step", "action": action_dict})
                self.waiting_for_worker[env_idx] = True

        # Drain in-flight worker steps
        for env_idx in range(self.num_envs):
            if self.waiting_for_worker[env_idx]:
                pipe = self.parent_pipes[env_idx]
                if pipe.poll(timeout=0.5):
                    msg = pipe.recv()
                    self.waiting_for_worker[env_idx] = False
                    self.worker_states[env_idx]["obs"] = msg["obs"]
                    self.worker_states[env_idx]["curr_player"] = msg["curr_player"]
                    self.worker_states[env_idx]["learner_player"] = msg["learner_player"]
                    if msg["done"]:
                        self._handle_game_over(
                            env_idx,
                            msg["winner"],
                            msg["turns"],
                            msg["reason"],
                            collected_trajectories,
                        )

        # Handle rollout cutoff for in-progress games across all environments
        for env_idx in range(self.num_envs):
            for p in (0, 1):
                p_traj = self.active_trajectories[env_idx][p]
                if p_traj:
                    p_traj[-1]["truncated"] = True
                    # Bootstrap from current state value estimate
                    current_obs = self.worker_states[env_idx]["obs"]
                    with torch.no_grad():
                        nb, ng, _ = encode_observation(current_obs, device=self.device)
                        _, v_boot = model(nb, ng)
                        bootstrap_val = float(v_boot.item())
                    collected_trajectories.append((p_traj, bootstrap_val))
                    self.active_trajectories[env_idx][p] = []

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
                by_opponent[opp]["wins"] += 1

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

    def close(self) -> None:
        """Close IPC pipes and terminate worker processes."""
        for pipe in self.parent_pipes:
            try:
                pipe.send({"type": "close"})
            except Exception:
                pass
        for proc in self.processes:
            try:
                proc.join(timeout=0.5)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        self.parent_pipes.clear()
        self.processes.clear()

    def __del__(self) -> None:
        self.close()
