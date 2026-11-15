"""Unit tests for PPO + GAE implementation."""

import unittest
import os
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn as nn

from Arena import Environment
from Policy import (
    PPO_Policy,
    QuickGELU,
    CLIPVisionTransformer,
    encode_observation,
    predict_action,
    compute_gae_for_trajectory,
    RolloutBuffer,
    OpponentPool,
    ParallelGameEnvs,
    PPOTrainer,
    load_trained_model,
)
from Agents import RandomAgent, ShortestPathAgent, GreedyWallAgent


class TestPPOArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = PPO_Policy().to(self.device)

    def test_forward_shapes(self):
        batch_size = 4
        boards = torch.randn(batch_size, 12, 9, 9, device=self.device)
        globals_ = torch.randn(batch_size, 3, device=self.device)

        logits, values = self.model(boards, globals_)
        self.assertEqual(logits.shape, (batch_size, 209))
        self.assertEqual(values.shape, (batch_size,))

    def test_action_masking(self):
        logits = torch.zeros(2, 209)
        mask = torch.zeros(2, 209, dtype=torch.bool)
        mask[0, 5] = True
        mask[1, 88] = True

        masked = PPO_Policy.apply_action_mask(logits, mask)
        self.assertEqual(masked[0, 5].item(), 0.0)
        self.assertTrue(masked[0, 0].item() < -1e30)
        self.assertEqual(masked[1, 88].item(), 0.0)

    def test_action_decoding(self):
        # Move action 0 -> row 0, col 0
        a0 = PPO_Policy.decode_action(0)
        self.assertEqual(a0["type"], "move")
        self.assertEqual(a0["to"], [0, 0])
        self.assertEqual(a0["row"], 0)
        self.assertEqual(a0["col"], 0)

        # Move action 40 -> row 4, col 4
        a40 = PPO_Policy.decode_action(40)
        self.assertEqual(a40["type"], "move")
        self.assertEqual(a40["to"], [4, 4])

        # Wall horizontal 81 -> row 0, col 0
        w_h = PPO_Policy.decode_action(81)
        self.assertEqual(w_h["type"], "wall")
        self.assertEqual(w_h["orientation"], "H")
        self.assertEqual(w_h["row"], 0)
        self.assertEqual(w_h["col"], 0)

        # Wall vertical 145 -> row 0, col 0
        w_v = PPO_Policy.decode_action(145)
        self.assertEqual(w_v["type"], "wall")
        self.assertEqual(w_v["orientation"], "V")
        self.assertEqual(w_v["row"], 0)
        self.assertEqual(w_v["col"], 0)


class TestGAECalculation(unittest.TestCase):
    def test_terminal_trajectory(self):
        """Matches hand-calculated trajectory from PPO_TRAINING.md verification."""
        traj = [
            {"reward": 0.0, "value": 0.2, "done": False, "truncated": False},
            {"reward": 0.0, "value": 0.5, "done": False, "truncated": False},
            {"reward": 1.0, "value": 0.8, "done": True, "truncated": False},
        ]
        compute_gae_for_trajectory(traj, gamma=0.99, gae_lambda=0.95, bootstrap_value=0.0)

        self.assertAlmostEqual(traj[2]["advantage"], 0.2, places=5)
        self.assertAlmostEqual(traj[2]["return"], 1.0, places=5)
        self.assertAlmostEqual(traj[1]["advantage"], 0.4801, places=4)
        self.assertAlmostEqual(traj[1]["return"], 0.9801, places=4)
        self.assertAlmostEqual(traj[0]["advantage"], 0.746534, places=4)
        self.assertAlmostEqual(traj[0]["return"], 0.946534, places=4)

    def test_nonterminal_cutoff_with_bootstrap(self):
        """Matches nonterminal cutoff with V_bootstrap."""
        traj = [
            {"reward": 0.0, "value": 0.3, "done": False, "truncated": False},
            {"reward": 0.0, "value": 0.6, "done": False, "truncated": True},
        ]
        compute_gae_for_trajectory(traj, gamma=0.99, gae_lambda=0.95, bootstrap_value=0.7)

        self.assertAlmostEqual(traj[1]["advantage"], 0.093, places=5)
        self.assertAlmostEqual(traj[1]["return"], 0.693, places=5)
        self.assertAlmostEqual(traj[0]["advantage"], 0.3814665, places=5)
        self.assertAlmostEqual(traj[0]["return"], 0.6814665, places=5)


class TestRolloutBuffer(unittest.TestCase):
    def test_buffer_batching_and_normalization(self):
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        traj = [
            {
                "board": torch.zeros(12, 9, 9),
                "global_features": torch.zeros(3),
                "action_mask": torch.ones(209, dtype=torch.bool),
                "action": 40,
                "old_log_prob": -1.0,
                "value": 0.1,
                "reward": 1.0,
                "done": True,
                "truncated": False,
            }
        ]
        buffer.add_trajectory(traj, bootstrap_value=0.0)
        self.assertEqual(len(buffer), 1)

        buffer.normalize_advantages()
        batches = list(buffer.get_batches(batch_size=1, device=torch.device("cpu")))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["boards"].shape, (1, 12, 9, 9))
        self.assertEqual(batches[0]["actions"].shape, (1,))


class TestCheckpointing(unittest.TestCase):
    def test_save_to_single_file_and_load(self):
        save_file = "test_single_checkpoint.pt"
        try:
            trainer = PPOTrainer(
                num_envs=2,
                steps_per_env=8,
                minibatch_size=8,
                save_path=save_file,
                device=torch.device("cpu"),
            )
            trainer.current_iteration = 7
            trainer.total_transitions_collected = 112
            trainer.save_checkpoint()

            self.assertTrue(os.path.isfile(save_file))

            # Load into fresh trainer
            new_trainer = PPOTrainer(
                num_envs=2,
                steps_per_env=8,
                minibatch_size=8,
                save_path=save_file,
                device=torch.device("cpu"),
            )
            success = new_trainer.load_checkpoint()
            self.assertTrue(success)
            self.assertEqual(new_trainer.current_iteration, 7)
            self.assertEqual(new_trainer.total_transitions_collected, 112)

            # Load into standalone model
            model = load_trained_model(save_file, device="cpu")
            self.assertIsInstance(model, PPO_Policy)
        finally:
            if os.path.exists(save_file):
                os.remove(save_file)


if __name__ == "__main__":
    unittest.main()
