"""PPO + GAE Trainer for the Corridor Game following PPO_TRAINING.md."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .PPO_Policy import PPO_Policy
from .buffer import RolloutBuffer
from .vector_env import ParallelGameEnvs
from .opponent_pool import OpponentPool
from .evaluator import evaluate_policy


class PPOTrainer:
    """Orchestrates PPO + GAE self-play training for the Corridor game.

    Hyperparameters and defaults adhere directly to PPO_TRAINING.md.
    """

    def __init__(
        self,
        model: Optional[PPO_Policy] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_envs: int = 16,
        steps_per_env: int = 512,
        minibatch_size: int = 512,
        ppo_epochs: int = 4,
        opponent_config: Optional[Dict[str, int]] = None,
        symmetric_self_play: bool = True,
        step_penalty: float = 0.0,
        checkpoint_pool_interval: int = 5,
        save_path: str = "ppo_corridor_model.pt",
        save_interval: int = 5,
        eval_interval: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = (model or PPO_Policy()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs

        self.symmetric_self_play = symmetric_self_play
        self.step_penalty = step_penalty
        self.checkpoint_pool_interval = checkpoint_pool_interval
        self.save_path = save_path
        self.save_interval = save_interval
        self.eval_interval = eval_interval

        # Opponent pool
        self.opponent_pool = OpponentPool(max_size=12, device=self.device)

        # Parallel environments
        self.envs = ParallelGameEnvs(
            num_envs=self.num_envs,
            opponent_config=opponent_config,
            opponent_pool=self.opponent_pool,
            symmetric_self_play=self.symmetric_self_play,
            step_penalty=self.step_penalty,
            device=self.device,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)
        self.total_target_transitions = self.num_envs * self.steps_per_env

        # Training progress counters
        self.current_iteration = 0
        self.total_transitions_collected = 0

    def collect_rollout(self) -> Dict[str, Any]:
        """Collect transitions across environments and compute GAE without waiting on stragglers."""
        self.buffer.clear()
        t0 = time.perf_counter()

        trajectories, game_stats = self.envs.collect_rollout(
            self.model,
            steps_per_env=self.steps_per_env,
            total_transitions=self.total_target_transitions,
        )

        # Add each trajectory to buffer (GAE is calculated per trajectory)
        for traj, boot_val in trajectories:
            self.buffer.add_trajectory(traj, bootstrap_value=boot_val)

        rollout_time = time.perf_counter() - t0
        self.total_transitions_collected += len(self.buffer)

        # Normalize advantages across the rollout
        self.buffer.normalize_advantages()

        return {
            "rollout_time": rollout_time,
            "num_transitions": len(self.buffer),
            "game_stats": game_stats,
        }

    def train_step(self) -> Dict[str, float]:
        """Perform PPO optimization epochs over the collected rollout transitions."""
        self.model.train()
        t0 = time.perf_counter()

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_values: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []

        # Run PPO epochs
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(
                batch_size=self.minibatch_size, device=self.device
            ):
                boards = batch["boards"]
                globals_ = batch["global_features"]
                action_masks = batch["action_masks"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward pass through current network
                logits, values = self.model(boards, globals_)

                # Action masking & categorical distribution
                masked_logits = PPO_Policy.apply_action_mask(logits, action_masks)
                dist = Categorical(logits=masked_logits)

                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Policy probability ratio: r_t(theta) = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Combined loss: -L_clip + 0.5 * L_v - c_e * H
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                # Compute optimization metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_values.append(entropy.item())
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_frac.item())

        opt_time = time.perf_counter() - t0

        # Explained variance: 1 - Var(returns - values) / Var(returns)
        with torch.no_grad():
            all_returns = torch.tensor([t["return"] for t in self.buffer.transitions])
            all_values = torch.tensor([t["value"] for t in self.buffer.transitions])
            var_y = torch.var(all_returns)
            var_diff = torch.var(all_returns - all_values)
            explained_var = (1.0 - var_diff / (var_y + 1e-8)).item()

        return {
            "policy_loss": sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
            "value_loss": sum(value_losses) / len(value_losses) if value_losses else 0.0,
            "entropy": sum(entropy_values) / len(entropy_values) if entropy_values else 0.0,
            "approx_kl": sum(approx_kls) / len(approx_kls) if approx_kls else 0.0,
            "clip_fraction": sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0,
            "explained_variance": explained_var,
            "opt_time": opt_time,
        }

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Periodically save the model and training state to just one file.

        Uses atomic write (write to temp file, then atomic rename) so the file
        is never left corrupted if interrupted.
        """
        save_path = path or self.save_path
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.current_iteration,
            "total_transitions": self.total_transitions_collected,
            "config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range,
                "entropy_coef": self.entropy_coef,
                "value_loss_coef": self.value_loss_coef,
                "max_grad_norm": self.max_grad_norm,
                "num_envs": self.num_envs,
                "steps_per_env": self.steps_per_env,
                "minibatch_size": self.minibatch_size,
                "ppo_epochs": self.ppo_epochs,
                "symmetric_self_play": self.symmetric_self_play,
            },
        }

        temp_path = f"{save_path}.tmp"
        torch.save(checkpoint_data, temp_path)
        os.replace(temp_path, save_path)
        return save_path

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load model and optimizer states from the single saved checkpoint file."""
        load_path = path or self.save_path
        if not os.path.isfile(load_path):
            return False

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iteration = checkpoint.get("iteration", 0)
        self.total_transitions_collected = checkpoint.get("total_transitions", 0)
        return True

    def train(
        self,
        num_iterations: int = 50,
        eval_opponents: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the main training loop for num_iterations rollouts.

        Periodically saves the model to just one file.
        """
        metrics_history: List[Dict[str, Any]] = []

        print(f"=== Starting PPO Training on {self.device} ===")
        print(f"Environments: {self.num_envs} | Steps/Env: {self.steps_per_env} | Minibatch: {self.minibatch_size}")
        print(f"Rollout transitions: {self.num_envs * self.steps_per_env} | PPO Epochs: {self.ppo_epochs}")
        print(f"Saving model to single file: {os.path.abspath(self.save_path)}")
        print(f"Opponent distribution: {self.envs.opponent_config}")
        print("=" * 55)

        for iteration in range(1, num_iterations + 1):
            self.current_iteration += 1
            iter_start = time.perf_counter()

            # 1 & 2. Collect rollout transitions
            rollout_info = self.collect_rollout()

            # 3. Optimize policy and critic
            opt_info = self.train_step()

            iter_time = time.perf_counter() - iter_start

            # 4. Periodically add current policy to opponent pool
            if self.current_iteration % self.checkpoint_pool_interval == 0:
                self.opponent_pool.add_snapshot(self.model)

            # 5. Periodically save model to just one file
            if self.current_iteration % self.save_interval == 0:
                saved_file = self.save_checkpoint()
                save_msg = f"[Saved model to {os.path.basename(saved_file)}]"
            else:
                save_msg = ""

            # Combine metrics
            combined_metrics = {
                "iteration": self.current_iteration,
                "total_transitions": self.total_transitions_collected,
                "rollout_time": rollout_info["rollout_time"],
                "opt_time": opt_info["opt_time"],
                "iter_time": iter_time,
                "policy_loss": opt_info["policy_loss"],
                "value_loss": opt_info["value_loss"],
                "entropy": opt_info["entropy"],
                "approx_kl": opt_info["approx_kl"],
                "clip_fraction": opt_info["clip_fraction"],
                "explained_variance": opt_info["explained_variance"],
                "game_stats": rollout_info["game_stats"],
            }

            # 6. Periodic evaluation against benchmark opponents (rarely, ~every 10 iters, ~5 games)
            eval_results = None
            if eval_opponents and (self.current_iteration % self.eval_interval == 0):
                eval_results = evaluate_policy(
                    self.model,
                    num_games_per_opponent=5,
                    device=self.device,
                    deterministic=True,
                )
                combined_metrics["eval"] = eval_results

            metrics_history.append(combined_metrics)

            # Console logging
            game_summary_str = ""
            if "by_opponent" in rollout_info["game_stats"]:
                parts = []
                for opp, s in rollout_info["game_stats"]["by_opponent"].items():
                    parts.append(f"{opp}: {s['win_rate']:.1%} win ({s['games']}g)")
                game_summary_str = " | ".join(parts)

            print(
                f"Iter {self.current_iteration:3d}/{self.current_iteration + num_iterations - iteration} | "
                f"Trans: {self.total_transitions_collected:6d} | "
                f"Loss(pi): {opt_info['policy_loss']:.4f} | "
                f"Loss(v): {opt_info['value_loss']:.4f} | "
                f"Ent: {opt_info['entropy']:.3f} | "
                f"KL: {opt_info['approx_kl']:.4f} | "
                f"ExpVar: {opt_info['explained_variance']:.3f} | "
                f"Time: {iter_time:.1f}s {save_msg}"
            )
            if game_summary_str:
                print(f"     Games -> {game_summary_str}")
            if eval_results:
                eval_str = " | ".join(
                    f"{opp}: {res['win_rate']:.1%} win ({int(res['games'])}g)"
                    for opp, res in eval_results.items()
                )
                print(f"     [Eval] -> {eval_str}")

        # Always save final model to the single file
        self.save_checkpoint()
        self.close()
        print(f"\nTraining completed! Model saved to single file: {os.path.abspath(self.save_path)}")
        return metrics_history

    def close(self) -> None:
        """Clean up environment workers."""
        if hasattr(self, "envs") and self.envs is not None:
            self.envs.close()
