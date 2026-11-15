"""Policy module containing PPO Actor-Critic model, GAE, and trainer."""

from .PPO_Policy import (
    PPO_Policy,
    QuickGELU,
    CLIPAttention,
    CLIPEncoderLayer,
    CLIPVisionTransformer,
    build_blocking_masks,
    calculate_distances,
    normalize_distances,
    encode_observation,
    predict_action,
)
from .gae import compute_gae_for_trajectory
from .buffer import RolloutBuffer
from .opponent_pool import OpponentPool
from .vector_env import ParallelGameEnvs
from .evaluator import evaluate_policy, create_agent_fn
from .ppo_trainer import PPOTrainer


def load_trained_model(
    checkpoint_path: str = "ppo_corridor_model.pt",
    device: str = "cpu",
) -> PPO_Policy:
    """Load a trained PPO_Policy from the saved single checkpoint file."""
    import torch
    dev = torch.device(device)
    model = PPO_Policy().to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


__all__ = [
    "PPO_Policy",
    "QuickGELU",
    "CLIPAttention",
    "CLIPEncoderLayer",
    "CLIPVisionTransformer",
    "build_blocking_masks",
    "calculate_distances",
    "normalize_distances",
    "encode_observation",
    "predict_action",
    "compute_gae_for_trajectory",
    "RolloutBuffer",
    "OpponentPool",
    "ParallelGameEnvs",
    "evaluate_policy",
    "create_agent_fn",
    "PPOTrainer",
    "load_trained_model",
]
