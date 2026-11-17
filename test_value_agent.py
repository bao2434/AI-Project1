"""Test suite for Agents/value_agent.py."""
import sys
import time
from pathlib import Path

# Add workspace root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agents.value_agent import ValueAgent, choose_action

def test_single_observation():
    print("--- 1. Testing Single Observation ---")
    observation = {
        "board_size": 9,
        "player_id": 0,
        "current_player": 0,
        "turn": 10,
        "max_turns": 200,
        "done": False,
        "winner": None,
        "you": {
            "position": [4, 4],
            "goal_row": 8,
            "walls_remaining": 10,
        },
        "opponent": {
            "position": [2, 4],
            "goal_row": 0,
            "walls_remaining": 10,
        },
        "walls": {
            "horizontal": [],
            "vertical": [],
        },
        "distances": [
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [6, 6, 6, 6, 6, 6, 6, 6, 6],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "legal_moves": [
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
        ],
        "legal_walls": [],
    }

    agent = ValueAgent()
    print("ValueAgent loaded successfully.")
    print("Backend model:", type(agent.model).__name__)
    print("Checkpoint path:", agent.checkpoint_path)

    t0 = time.perf_counter()
    action = agent(observation)
    t1 = time.perf_counter()

    print(f"Chosen Action: {action}")
    print(f"Time Taken: {(t1 - t0)*1000:.2f} ms")
    assert "type" in action, "Action must contain 'type'"
    if action["type"] == "move":
        assert "to" in action
    elif action["type"] == "wall":
        assert "orientation" in action and "row" in action and "col" in action
    print("Observation test passed successfully!")

def test_immediate_win():
    print("\n--- 2. Testing Immediate Win Condition ---")
    observation = {
        "board_size": 9,
        "player_id": 0,
        "current_player": 0,
        "turn": 30,
        "max_turns": 200,
        "done": False,
        "winner": None,
        "you": {
            "position": [7, 4],
            "goal_row": 8,
            "walls_remaining": 3,
        },
        "opponent": {
            "position": [3, 4],
            "goal_row": 0,
            "walls_remaining": 2,
        },
        "walls": {"horizontal": [], "vertical": []},
        "distances": [[abs(r - 8) for _ in range(9)] for r in range(9)],
        "legal_moves": [[8, 4], [7, 3], [7, 5], [6, 4]],
        "legal_walls": [],
    }
    agent = ValueAgent()
    action = agent(observation)
    print(f"Chosen Action: {action}")
    assert action == {"type": "move", "to": [8, 4]}, "Should pick immediate winning move [8, 4]"
    print("Immediate win test passed!")

def test_module_choose_action():
    print("\n--- 3. Testing Module choose_action() Entry Point ---")
    observation = {
        "board_size": 9,
        "player_id": 0,
        "current_player": 0,
        "turn": 1,
        "max_turns": 200,
        "done": False,
        "winner": None,
        "you": {
            "position": [0, 4],
            "goal_row": 8,
            "walls_remaining": 10,
        },
        "opponent": {
            "position": [8, 4],
            "goal_row": 0,
            "walls_remaining": 10,
        },
        "walls": {"horizontal": [], "vertical": []},
        "distances": [[abs(r - 8) for _ in range(9)] for r in range(9)],
        "legal_moves": [[1, 4], [0, 3], [0, 5]],
        "legal_walls": [],
    }
    act = choose_action(observation)
    print("Action from module choose_action():", act)
    assert act is not None
    print("Module choose_action() test passed!")

if __name__ == "__main__":
    test_single_observation()
    test_immediate_win()
    test_module_choose_action()
    print("\nALL VALUE AGENT TESTS PASSED!")
