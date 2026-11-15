"""Baseline agents for the Corridor (Wrongway) game.

Provides standard reference opponents:
- RandomAgent: Selects uniformly at random among legal moves and walls.
- ShortestPathAgent: Greedily moves along the shortest path to the goal row.
- GreedyWallAgent: Balances shortest-path movement with defensive wall placement.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class RandomAgent:
    """Agent that selects randomly among all legal actions."""

    def __init__(self, wall_prob: float = 0.2, seed: Optional[int] = None):
        self.wall_prob = wall_prob
        self.rng = random.Random(seed)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        legal_moves = obs.get("legal_moves", [])
        legal_walls = obs.get("legal_walls", [])
        walls_remaining = obs.get("you", {}).get("walls_remaining", 0)

        # Decide whether to place a wall if walls are available
        if walls_remaining > 0 and legal_walls and self.rng.random() < self.wall_prob:
            wall = self.rng.choice(legal_walls)
            return {
                "type": "wall",
                "orientation": wall["orientation"],
                "row": wall["row"],
                "col": wall["col"],
            }

        if legal_moves:
            move = self.rng.choice(legal_moves)
            return {"type": "move", "to": list(move)}

        if legal_walls and walls_remaining > 0:
            wall = self.rng.choice(legal_walls)
            return {
                "type": "wall",
                "orientation": wall["orientation"],
                "row": wall["row"],
                "col": wall["col"],
            }

        return {"type": "pass"}


class ShortestPathAgent:
    """Agent that greedily moves to minimize the distance to its goal."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        legal_moves = obs.get("legal_moves", [])
        if not legal_moves:
            return {"type": "pass"}

        distances = obs.get("distances", [])
        best_dist = 9999
        best_moves: List[List[int]] = []

        for m in legal_moves:
            r, c = m[0], m[1]
            d = distances[r][c] if 0 <= r < len(distances) and 0 <= c < len(distances[0]) else 9999
            if d < 0:
                d = 9999
            if d < best_dist:
                best_dist = d
                best_moves = [m]
            elif d == best_dist:
                best_moves.append(m)

        chosen = self.rng.choice(best_moves) if best_moves else legal_moves[0]
        return {"type": "move", "to": list(chosen)}


class GreedyWallAgent:
    """Agent that moves along the shortest path, but places walls when opponent is ahead."""

    def __init__(self, wall_threshold: int = 2, seed: Optional[int] = None):
        self.wall_threshold = wall_threshold
        self.rng = random.Random(seed)
        self.shortest_path_agent = ShortestPathAgent(seed=seed)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        legal_moves = obs.get("legal_moves", [])
        legal_walls = obs.get("legal_walls", [])
        you = obs.get("you", {})
        opp = obs.get("opponent", {})

        walls_remaining = you.get("walls_remaining", 0)
        distances = obs.get("distances", [])

        # Estimate own shortest distance
        yr, yc = you.get("position", [0, 0])
        own_dist = distances[yr][yc] if 0 <= yr < len(distances) and 0 <= yc < len(distances[0]) else 99

        # If opponent is close to goal or ahead, consider placing a wall
        opp_r, opp_c = opp.get("position", [0, 0])
        opp_goal = opp.get("goal_row", 0)
        opp_dist_naive = abs(opp_r - opp_goal)

        if walls_remaining > 0 and legal_walls and (opp_dist_naive <= self.wall_threshold or opp_dist_naive < own_dist):
            # Pick a wall near opponent
            candidate_walls = [
                w for w in legal_walls
                if abs(w["row"] - opp_r) <= 1 and abs(w["col"] - opp_c) <= 1
            ]
            if candidate_walls and self.rng.random() < 0.6:
                w = self.rng.choice(candidate_walls)
                return {
                    "type": "wall",
                    "orientation": w["orientation"],
                    "row": w["row"],
                    "col": w["col"],
                }

        return self.shortest_path_agent(obs)
