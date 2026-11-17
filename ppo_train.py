#!/usr/bin/env python3
"""Main training script for PPO + GAE on the 9x9 Corridor Game.

Trains the Actor-Critic policy using self-play and benchmark opponents.
Periodically saves the model to just one file.

Usage example:
    python train.py --iterations 50 --save-interval 5 --model-file ppo_corridor_model.pt
"""

import argparse
import os
import sys
import torch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from Policy import PPO_Policy, PPOTrainer, evaluate_policy, load_trained_model
from Agents import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO+GAE agent on Corridor Game and periodically save to one file."
    )
    # Training duration
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of rollout iterations to train (default: 50).",
    )
    # Model saving to just one file
    parser.add_argument(
        "--model-file",
        "--save-path",
        type=str,
        default="ppo_corridor_model.pt",
        help="Single file path to periodically save the model (default: ppo_corridor_model.pt).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save model to the single file every N iterations (default: 5).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing model file if present.",
    )
    # Parallel environments & rollout
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8, set to CPU core count).",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=1024,
        help="Transitions per environment per rollout (default: 1024, total: 8,192).",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=512,
        help="Minibatch size for PPO updates (default: 512).",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Number of PPO passes over rollout data (default: 4).",
    )
    # Hyperparameters from PPO_TRAINING.md
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="GAE discount factor (default: 0.99).",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda (default: 0.95).",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range epsilon (default: 0.2).",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient (default: 0.01).",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient (default: 0.5).",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=0.0,
        help="Per-decision penalty to discourage stalling (default: 0.0).",
    )
    # Opponent mixture
    parser.add_argument(
        "--self-envs",
        type=int,
        default=8,
        help="Number of environments for self-play (default: 8).",
    )
    parser.add_argument(
        "--baseline-envs",
        type=int,
        default=4,
        help="Number of environments for fixed baseline (default: 4).",
    )
    parser.add_argument(
        "--pool-envs",
        type=int,
        default=4,
        help="Number of environments for past checkpoint pool (default: 4).",
    )
    parser.add_argument(
        "--one-sided",
        action="store_true",
        help="Use one-sided learning instead of symmetric self-play.",
    )
    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Evaluate against benchmarks every N iterations (default: 10).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation of the saved model without training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on (default: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Build opponent configuration dictionary
    opponent_config = {
        "self": 4,                # Self-play against current PPO policy
        "wrong_way_bot": 4,       # Exploration / wrong-way bot
        # "hard_waller": 3,         # Hard waller strategic agent
        # "mcts_v7": 2,             # Monte Carlo Tree Search v7
        # "quoridor_mcts_fast": 1,  # Capped Quoridor MCTS agent (~0.1s)
        # "minimax_d2": 1,          # Capped Minimax depth-2 agent (~0.05-0.1s)
    }

    # Eval-only mode
    # if args.eval_only:
    #     if not os.path.exists(args.model_file):
    #         print(f"Error: Model file '{args.model_file}' not found for evaluation.")
    #         sys.exit(1)
    #     print(f"Loading model from '{args.model_file}' for evaluation...")
    #     model = load_trained_model(args.model_file, device=str(device))
    #     opponents = {
    #         "ShortestPathAgent": ShortestPathAgent(),
    #         "GreedyWallAgent": GreedyWallAgent(),
    #         "RandomAgent": RandomAgent(),
    #     }
    #     results = evaluate_policy(
    #         model,
    #         opponents=opponents,
    #         num_games_per_opponent=20,
    #         device=device,
    #         deterministic=True,
    #     )
    #     print("\n=== Evaluation Results ===")
    #     for opp_name, metrics in results.items():
    #         print(
    #             f"{opp_name:18s} | Win: {metrics['win_rate']:5.1%} | "
    #             f"Loss: {metrics['loss_rate']:5.1%} | Draw: {metrics['draw_rate']:5.1%} | "
    #             f"Avg Turns: {metrics['avg_turns']:.1f}"
    #         )
    #     return

    # Initialize trainer
    print("meow")
    trainer = PPOTrainer(
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_coef,
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        minibatch_size=args.minibatch_size,
        ppo_epochs=args.ppo_epochs,
        opponent_config=opponent_config,
        symmetric_self_play=not args.one_sided,
        step_penalty=args.step_penalty,
        save_path=args.model_file,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        device=device,
    )

    # Resume if requested
    if args.resume:
        if os.path.exists(args.model_file):
            print(f"Resuming training from '{args.model_file}'...")
            trainer.load_checkpoint(args.model_file)
            print(f"Resumed at iteration {trainer.current_iteration}, {trainer.total_transitions_collected} transitions.")
        else:
            print(f"Model file '{args.model_file}' does not exist yet. Starting fresh training.")

    # Run training
    trainer.train(num_iterations=args.iterations, eval_opponents=True)


if __name__ == "__main__":
    main()
