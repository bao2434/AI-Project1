"""Export trained Wrongway CNN into a ONE-FILE standard-library Minimax agent.

Place: training/export_minimax_cnn.py
Example from repo root (Kaggle working copy):
 python training/export_minimax_cnn.py \
   --checkpoint /kaggle/working/models/cnn_v2/best_model.pt \
   --base Agents/minimax_agent_summary.py \
   --out Agents/minimax_cnn.py

Requires PyTorch ONLY for export. Generated agent has NO torch/numpy dependency.
This is an experimental depth-1 search with a CNN leaf evaluator, NOT a
promise that it outperforms the original depth-2/3 heuristic Minimax.
"""

import argparse
import base64
from pathlib import Path
import struct
import textwrap
import zlib


# Exact layer shapes from training/value_network.py (ValueCNN):
LAYERS = (
    ("network.0.weight", (16, 8, 3, 3)),
    ("network.0.bias", (16,)),
    ("network.2.weight", (32, 16, 3, 3)),
    ("network.2.bias", (32,)),
    ("network.4.weight", (32, 32, 3, 3)),
    ("network.4.bias", (32,)),
    ("network.7.weight", (64, 32 * 9 * 9)),
    ("network.7.bias", (64,)),
    ("network.9.weight", (1, 64)),
    ("network.9.bias", (1,)),
)

RUNTIME = r'''
# --------------------------------------------------------------------------
# CNN inference. Generated weights are embedded as compressed float32 bytes.
# The inference uses only the standard library. Trained state_encoder.py
# order is preserved exactly; the leaf is encoded from the player-to-move POV.
# --------------------------------------------------------------------------
import base64 as _base64
import math as _math
import struct as _struct
import zlib as _zlib

_CNN_B85 = """__PACKED__"""
_CNN_FLOAT_COUNT = __FLOAT_COUNT__


def _load_cnn_weights():
    raw = _zlib.decompress(_base64.b85decode(_CNN_B85.replace("\n", "")))
    if len(raw) != _CNN_FLOAT_COUNT * 4:
        raise ValueError("CNN weight payload has unexpected size")
    f = _struct.unpack("<" + str(_CNN_FLOAT_COUNT) + "f", raw)
    specs = (1152, 16, 4608, 32, 9216, 32, 165888, 64, 64, 1)
    parts, offset = [], 0
    for n in specs:
        parts.append(f[offset:offset+n])
        offset += n
    return parts

_CNN_PARTS = _load_cnn_weights()
# For every output cell, map the valid input cells to the 3x3 kernel indices.
_CNN_NEIGHBORS = []
for _p in range(81):
    _r, _c = divmod(_p, 9)
    _CNN_NEIGHBORS.append(tuple(
        ((_r + _kr - 1) * 9 + (_c + _kc - 1), _kr * 3 + _kc)
        for _kr in range(3) for _kc in range(3)
        if 0 <= _r + _kr - 1 < 9 and 0 <= _c + _kc - 1 < 9
    ))


def _cnn_conv_relu(x, in_ch, out_ch, weights, biases):
    """PyTorch Conv2d(kernel=3,padding=1) + ReLU on flat CHW arrays."""
    y = [0.0] * (out_ch * 81)
    for oc in range(out_ch):
        ob = oc * 81
        for p, neighbors in enumerate(_CNN_NEIGHBORS):
            s = biases[oc]
            for ic in range(in_ch):
                xb = ic * 81
                wb = (oc * in_ch + ic) * 9
                for ip, ki in neighbors:
                    xv = x[xb + ip]
                    if xv:
                        s += xv * weights[wb + ki]
            y[ob + p] = s if s > 0.0 else 0.0
    return y


def _cnn_predict(x):
    """Raw ValueCNN output in [-1,+1], one forward pass, no PyTorch."""
    w0, b0, w1, b1, w2, b2, w3, b3, w4, b4 = _CNN_PARTS
    a = _cnn_conv_relu(x, 8, 16, w0, b0)
    a = _cnn_conv_relu(a, 16, 32, w1, b1)
    a = _cnn_conv_relu(a, 32, 32, w2, b2)
    z = [0.0] * 64
    for i in range(64):
        offset = i * 2592
        s = b3[i]
        for j, val in enumerate(a):
            if val:
                s += val * w3[offset + j]
        z[i] = s if s > 0.0 else 0.0
    out = b4[0]
    for i in range(64):
        out += z[i] * w4[i]
    return _math.tanh(out)

_ROOT_TURN = 0
_ROOT_MAX_TURNS = 500


def _encode_leaf(state, next_player, ply):
    """Replicates training/state_encoder.py's exact channel ordering."""
    x = [0.0] * 648
    own = state.pos[next_player]
    opp = state.pos[1 - next_player]
    x[own[0] * 9 + own[1]] = 1.0
    x[81 + opp[0] * 9 + opp[1]] = 1.0
    for r, c in state.h_walls:
        x[162 + r * 9 + c] = 1.0
    for r, c in state.v_walls:
        x[243 + r * 9 + c] = 1.0
    my_w = state.walls_rem[next_player] / 10.0
    their_w = state.walls_rem[1 - next_player] / 10.0
    frac = min(1.0, max(0.0, (_ROOT_TURN + ply) / max(1, _ROOT_MAX_TURNS)))
    for p in range(81):
        x[324 + p] = my_w
        x[405 + p] = their_w
        x[567 + p] = frac
    goal_base = 486 + state.goals[next_player] * 9
    for c in range(9):
        x[goal_base + c] = 1.0
    return x


def nn_leaf_score(state, player, ply):
    """Return score from ROOT/MAX's POV; player is next to move at leaf."""
    value = _cnn_predict(_encode_leaf(state, player, ply))
    return 100.0 * (value if player == 0 else -value)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="cnn_v2/best_model.pt")
    p.add_argument("--base", default="Agents/minimax_agent_summary.py")
    p.add_argument("--out", default="Agents/minimax_cnn.py")
    p.add_argument("--max-depth", type=int, default=1,
                   help="Depth 1 recommended until CPU inference benchmarked")
    p.add_argument("--wall-branches", type=int, default=2)
    p.add_argument("--time-limit", type=float, default=0.70)
    args = p.parse_args()
    if not (0 < args.time_limit < 1) or not (1 <= args.max_depth <= 50) or not (0 <= args.wall_branches <= 18):
        p.error("Expected 0 < time-limit < 1, 1 <= max-depth <= 50, 0 <= wall-branches <= 18")
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "cnn" or checkpoint.get("input_shape") != [8, 9, 9]:
        p.error("Checkpoint must be ValueCNN trained with 8x9x9 input")
    state = checkpoint["model_state_dict"]
    floats = []
    for key, shape in LAYERS:
        if key not in state or tuple(state[key].shape) != shape:
            p.error(f"Weight mismatch {key}: expected shape {shape}")
        floats.extend(state[key].detach().float().contiguous().view(-1).tolist())
    raw = struct.pack("<" + str(len(floats)) + "f", *floats)
    encoded = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
    lines = "\n".join(textwrap.wrap(encoded, 110))

    source = Path(args.base).read_text(encoding="utf-8")
    import re
    def change_line(pattern, new):
        nonlocal source
        source, count = re.subn(pattern, new, source, count=1, flags=re.M)
        if count != 1:
            p.error("Cannot patch original Minimax source: " + pattern)

    change_line(r'^TIME_LIMIT\s*=.*$', f'TIME_LIMIT = {args.time_limit}')
    change_line(r'^MAX_DEPTH\s*=.*$', f'MAX_DEPTH = {args.max_depth}')
    change_line(r'^MAX_WALL_BRANCHES\s*=.*$', f'MAX_WALL_BRANCHES = {args.wall_branches}')
    # Only replace depth-0 leaf evaluator. Keep original heuristic for ordering
    # pawn moves and selecting the best candidate walls (cheap, fast).
    needle = '        return heuristic(state), None'
    if source.count(needle) != 1:
        p.error("Could not locate unique Minimax leaf evaluation")
    source = source.replace(needle,
        '        return nn_leaf_score(state, player, SEARCH_ROOT_DEPTH - depth), None')
    needle = '    start_time = time.monotonic()'
    if source.count(needle) != 1:
        p.error("Could not locate unique choose_action clock initialization")
    source = source.replace(needle, '''    global _ROOT_TURN, _ROOT_MAX_TURNS
    _ROOT_TURN = int(observation.get("turn", 0))
    _ROOT_MAX_TURNS = int(observation.get("max_turns", 500))
    start_time = time.monotonic()''')
    extras = RUNTIME.replace("__PACKED__", lines).replace("__FLOAT_COUNT__", str(len(floats)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source + "\n" + extras + "\n", encoding="utf-8")
    print(f"Created: {out} ({out.stat().st_size:,} bytes; {len(floats):,} weights)")
    print(f"Search: depth={args.max_depth}, wall_branches={args.wall_branches}, time_limit={args.time_limit}")
    print("No torch/numpy imports in output. Check runtime and accuracy before submission.")


if __name__ == "__main__":
    main()
