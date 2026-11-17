from preprocessing import _compute_distance_map
from MeowMeowBot.value_model import ValueCNN
import torch

from collections import deque
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

class MeowMeowBot:
    def __init__(self, value_model_dir:str='./models/value_model.pt') -> None:
        self._recent_positions: List[Tuple[int, int]] = []
        self._last_match_reset: float = 0.0
        self.value_model: ValueCNN = torch.load(value_model_dir, map_location="cpu", weights_only=False)
        self.value_model.eval()

    def __call__(self, observation: Dict[str, Any]) -> Dict[str, Any]:

        start_time = time.perf_counter()
        deadline = start_time + 0.85  # safety margin for 1.0s limit
    
        # Detect new match and reset history
        turn = observation.get("turn", 0)
        now = time.time()
        if turn <= 1 or (now - _last_match_reset > 10.0):
            _recent_positions = []
            _last_match_reset = now

        my_pos = tuple(observation["you"]["position"])
        opp_pos = tuple(observation["opponent"]["position"])
        my_goal = observation["you"]["goal_row"]
        opp_goal = observation["opponent"]["goal_row"]
        my_walls = observation["you"]["walls_remaining"]
        opp_walls = observation["opponent"]["walls_remaining"]
    
        h_walls = {tuple(w) for w in observation["walls"].get("horizontal", [])}
        v_walls = {tuple(w) for w in observation["walls"].get("vertical", [])}
    
        legal_moves = [tuple(m) for m in observation.get("legal_moves", [])]
        if not legal_moves and not observation.get("legal_walls"):
            return {"type": "pass"}

        # Immediate win check
        for m in legal_moves:
            if m[0] == my_goal:
                self._recent_positions.append(m)
                return {"type": "move", "to": list(m)}

        my_dist_map = observation.get("distances")
        if not my_dist_map:
            my_dist_map = _compute_distance_map(my_goal, h_walls, v_walls)
        opp_dist_map = _compute_distance_map(opp_goal, h_walls, v_walls)

        my_dist = my_dist_map[my_pos[0]][my_pos[1]]
        opp_dist = opp_dist_map[opp_pos[0]][opp_pos[1]]

        if not _recent_positions or _recent_positions[-1] != my_pos:
            _recent_positions.append(my_pos)
        if len(_recent_positions) > 6:
            _recent_positions = _recent_positions[-6:]

        

        