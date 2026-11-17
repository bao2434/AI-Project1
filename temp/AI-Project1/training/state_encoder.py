"""Encode Wrongway pre-action states for a small MLP or CNN.

Place at: AI-Project1/training/state_encoder.py
No third-party dependencies: outputs Python lists, readily converted to tensors.

Eight 9x9 planes, ALWAYS from the viewpoint of the acting player:
  0: acting player's pawn (one-hot)
  1: opposing pawn (one-hot)
  2: horizontal wall anchors (8x8 anchors in top-left)
  3: vertical wall anchors (8x8 anchors in top-left)
  4: acting player's remaining walls / 10 (constant plane)
  5: opponent's remaining walls / 10 (constant plane)
  6: acting player's goal row (one-hot row)
  7: elapsed-turn fraction (constant plane)

All coordinates remain ABSOLUTE, just like Arena's observations.
"""

SIZE = 9
CHANNELS = 8


def encode_cnn(state):
    """Return a [8][9][9] nested list; accepts a compact state or observation.

    Expected fields: you, opponent, walls, turn, max_turns.
    """
    you = state["you"]
    opponent = state["opponent"]
    walls = state["walls"]
    turn_fraction = min(1.0, max(0.0, state["turn"] / max(1, state.get("max_turns", 500))))

    def zeros():
        return [[0.0] * SIZE for _ in range(SIZE)]

    planes = [zeros() for _ in range(CHANNELS)]

    for plane_id, person in ((0, you), (1, opponent)):
        r, c = person["position"]
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError("Pawn position outside board")
        planes[plane_id][r][c] = 1.0

    for plane_id, orientation in ((2, "horizontal"), (3, "vertical")):
        for r, c in walls[orientation]:
            if not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1):
                raise ValueError("Invalid wall anchor")
            planes[plane_id][r][c] = 1.0

    goal_row = you["goal_row"]
    if goal_row not in (0, SIZE - 1):
        raise ValueError("Unexpected goal row")

    values = {
        4: you["walls_remaining"] / 10.0,
        5: opponent["walls_remaining"] / 10.0,
        7: turn_fraction,
    }
    for idx, value in values.items():
        for row in planes[idx]:
            row[:] = [float(value)] * SIZE

    planes[6][goal_row][:] = [1.0] * SIZE
    return planes


def encode_mlp(state):
    """Flatten the 8x9x9 CNN input to a 648-dimensional feature vector."""
    return [value for plane in encode_cnn(state) for row in plane for value in row]


if __name__ == "__main__":
    sample = {
        "you": {"position": [8, 4], "goal_row": 0, "walls_remaining": 10},
        "opponent": {"position": [0, 4], "goal_row": 8, "walls_remaining": 10},
        "walls": {"horizontal": [], "vertical": []},
        "turn": 0, "max_turns": 500,
    }
    cnn = encode_cnn(sample)
    assert len(cnn) == 8 and all(len(p) == 9 and all(len(row) == 9 for row in p) for p in cnn)
    assert sum(map(sum, cnn[0])) == 1 and sum(map(sum, cnn[1])) == 1
    assert len(encode_mlp(sample)) == 648
    print("Encoder OK: CNN=(8,9,9), MLP=648")
