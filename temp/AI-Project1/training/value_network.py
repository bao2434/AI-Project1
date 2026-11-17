"""Small PyTorch value networks for Wrongway training (not a submission agent).

Place in AI-Project1/training/value_network.py.
Input: the eight 9x9 planes produced by training/state_encoder.py.
Output: a scalar in [-1, 1] from the current player's perspective.
"""

import torch
from torch import nn


class ValueMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 9 * 9, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ValueCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def build_model(name):
    if name == "mlp":
        return ValueMLP()
    if name == "cnn":
        return ValueCNN()
    raise ValueError(f"Unknown architecture: {name}")
