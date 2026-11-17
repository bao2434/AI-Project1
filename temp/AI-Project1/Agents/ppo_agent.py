"""Single-file agent wrapper for the trained PPO model.

Exports choose_action(observation) compatible with Arena.run_simulation,
Arena.Session, and Arena.cli / StudentAgent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import torch

# Ensure workspace root is available
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Policy.PPO_Policy import PPO_Policy, predict_action


def load_model(checkpoint_path: str, device: str = "cpu") -> PPO_Policy:
    dev = torch.device(device)
    model = PPO_Policy().to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

_MODEL: Optional[PPO_Policy] = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> PPO_Policy:
    global _MODEL
    if _MODEL is None:
        model_path = os.getenv("PPO_MODEL_PATH", "ppo_corridor_model.pt")
        candidate_paths = [
            Path(model_path),
            ROOT / model_path,
            ROOT / "ppo_corridor_model.pt",
            ROOT / "Policy" / "ppo_corridor_model.pt",
        ]
        loaded = False
        for p in candidate_paths:
            if p.is_file():
                _MODEL = load_model(str(p), device=_DEVICE)
                loaded = True
                break
        if not loaded:
            # Fallback to randomly initialized architecture
            _MODEL = PPO_Policy().to(_DEVICE)
            _MODEL.eval()
    return _MODEL


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Return a legal action dictionary for the active player."""
    model = get_model()
    return predict_action(model, observation, device=_DEVICE, deterministic=True)


if __name__ == "__main__":
    from Arena import Environment
    env = Environment()
    obs = env.get_observation()
    action = choose_action(obs)
    print("Sample action chosen by PPO agent:", action)
