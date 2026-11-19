"""Train and evaluate the lightweight custom Value MLP for G5 (Direction D5).

Uses only Python standard library (or math/random/json) so it is 100% portable.
Splits the 126 games from `temp/AI-Project1/training/datasets/v2/states.jsonl`
and `search_v3_fixed/states_search.jsonl` strictly by `game_id` into:
  - Train: 70% of games (88 games)
  - Val:   15% of games (19 games)
  - Test:  15% of games (19 games)

Architecture: Residual Tactical MLP (12 strategic features -> 16 ReLU hidden units -> 1 Sigmoid output + linear skip connection).
"""
import json
from math import exp, log, sqrt
from pathlib import Path
import random
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Gedix.g4_combined import (
    BIT,
    distance,
    from_obs,
    route_threat_profile,
)
from Gedix.g0_baseline import value as g0_value
from Gedix.g1_threat_eval import value as g1_value


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    ez = exp(z)
    return ez / (1.0 + ez)


def extract_features(st):
    """12-dimensional strategic feature vector (pure standard library)."""
    walls, me = st[4], st[5]
    op = 1 - me
    sq_me, sq_op = st[me], st[op]
    dm = distance(walls, me, sq_me)
    do = distance(walls, op, sq_op)
    wm = st[2 + me]
    wo = st[2 + op]

    t = do - dm
    um = min(wm, do)
    uo = min(wo, dm)
    phase = (wm + wo) / 20.0

    tm = route_threat_profile(walls, me, sq_me, sq_op)[0] if wo > 0 else 0
    to = route_threat_profile(walls, op, sq_op, sq_me)[0] if wm > 0 else 0
    tm_cap = min(tm, 10)
    to_cap = min(to, 10)

    opt = max(-6, min(6, t + um))
    pes = max(-6, min(6, t - uo))
    race = 1.0 if (wo == 0 and t >= 0) else (-1.0 if (wm == 0 and t < 0) else 0.0)

    return (
        1.0,
        t / 4.0,
        (wm - wo) / 4.0,
        (um - uo) / 4.0,
        phase,
        (t / 4.0) * phase,
        to_cap / 6.0,
        tm_cap / 6.0,
        (dm + do) / 16.0,
        opt / 6.0,
        pes / 6.0,
        race,
    )


def forward_mlp(x, w_lin, w1, b1, w2, b2):
    z = b2 + sum(xi * wi for xi, wi in zip(x, w_lin))
    for j in range(len(b1)):
        h = b1[j] + sum(x[i] * w1[i][j] for i in range(len(x)))
        if h > 0.0:
            z += h * w2[j]
    return sigmoid(z)


def evaluate_predictions(samples, pred_fn):
    n = len(samples)
    correct = 0
    brier_sum = 0.0
    bce_sum = 0.0
    for st, x, y_outcome, y_target in samples:
        p = pred_fn(st, x)
        p_clip = min(max(p, 1e-6), 1.0 - 1e-6)
        pred_label = 1 if p >= 0.5 else 0
        if pred_label == y_outcome:
            correct += 1
        brier_sum += (p - y_outcome) ** 2
        bce_sum += -(y_outcome * log(p_clip) + (1 - y_outcome) * log(1.0 - p_clip))
    return {
        "count": n,
        "accuracy": round(correct / max(1, n), 4),
        "brier_score": round(brier_sum / max(1, n), 4),
        "log_loss": round(bce_sum / max(1, n), 4),
    }


def main():
    v2_path = ROOT / "temp/AI-Project1/training/datasets/v2/states.jsonl"
    search_path = ROOT / "temp/AI-Project1/training/datasets/search_v3_fixed/states_search.jsonl"

    teacher_map = {}
    if search_path.is_file():
        with open(search_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = (row["game_id"], row["state"]["turn"])
                # teacher_value is in [-1, 1] -> convert to [0, 1] probability target
                teacher_map[key] = 0.5 * (row["teacher_value"] + 1.0)

    by_game = {}
    with open(v2_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gid = row["game_id"]
            st_dict = row["state"]
            # Skip random opening plies (turns < 8) so we train on tactical mid/endgame states
            if st_dict.get("turn", 0) < 8:
                continue
            st = from_obs(st_dict)
            x = extract_features(st)
            y_outcome = 1 if row["outcome"] == 1 else 0
            key = (gid, st_dict.get("turn", 0))
            if key in teacher_map:
                # Blended target: 60% game outcome + 40% search teacher value
                y_target = 0.60 * y_outcome + 0.40 * teacher_map[key]
            else:
                # Blended target with soft G1 teacher regularization to prevent noisy outcome overfitting
                y_target = 0.75 * y_outcome + 0.25 * g1_value(st, "G1b")
            by_game.setdefault(gid, []).append((st, x, y_outcome, y_target))

    game_ids = sorted(by_game.keys())
    rng = random.Random(20261119)
    rng.shuffle(game_ids)

    n_games = len(game_ids)
    n_train = int(0.70 * n_games)
    n_val = int(0.15 * n_games)
    train_gids = game_ids[:n_train]
    val_gids = game_ids[n_train:n_train + n_val]
    test_gids = game_ids[n_train + n_val:]

    train_samples = [s for gid in train_gids for s in by_game[gid]]
    val_samples = [s for gid in val_gids for s in by_game[gid]]
    test_samples = [s for gid in test_gids for s in by_game[gid]]

    print(f"Games split: train={len(train_gids)}, val={len(val_gids)}, test={len(test_gids)}")
    print(f"Samples split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Initialize 12 -> 16 -> 1 residual tactical MLP
    in_dim = 12
    hid_dim = 16
    # Warm-start linear skip connection near G1b tactical weights
    w_lin = [0.0, 3.28, 0.40, 0.56, 0.0, 0.0, 1.59, -1.76, 0.0, 0.25, 0.25, 0.35]
    w1 = [[rng.uniform(-0.20, 0.20) for _ in range(hid_dim)] for _ in range(in_dim)]
    b1 = [rng.uniform(-0.05, 0.05) for _ in range(hid_dim)]
    w2 = [rng.uniform(-0.20, 0.20) for _ in range(hid_dim)]
    b2 = 0.0

    best_val_brier = 999.0
    best_weights = None

    lr_init = 0.015
    l2 = 1e-4
    epochs = 25

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_samples)
        lr = lr_init * (0.92 ** (epoch - 1))
        for _, x, _, y_target in train_samples:
            # Forward
            h_pre = [b1[j] + sum(x[i] * w1[i][j] for i in range(in_dim)) for j in range(hid_dim)]
            h_act = [v if v > 0.0 else 0.0 for v in h_pre]
            z = b2 + sum(x[i] * w_lin[i] for i in range(in_dim)) + sum(h_act[j] * w2[j] for j in range(hid_dim))
            p = sigmoid(z)

            # Gradient of BCE wrt z: dz = p - y_target
            dz = p - y_target

            # Update output & linear skip
            b2 -= lr * dz
            for i in range(in_dim):
                w_lin[i] -= lr * (dz * x[i] + l2 * w_lin[i])

            # Backprop to hidden layer
            for j in range(hid_dim):
                grad_w2 = dz * h_act[j] + l2 * w2[j]
                dh = (dz * w2[j]) if h_pre[j] > 0.0 else 0.0
                w2[j] -= lr * grad_w2
                if dh != 0.0:
                    b1[j] -= lr * dh
                    for i in range(in_dim):
                        w1[i][j] -= lr * (dh * x[i] + l2 * w1[i][j])

        val_metrics = evaluate_predictions(val_samples, lambda st, x: forward_mlp(x, w_lin, w1, b1, w2, b2))
        if val_metrics["brier_score"] < best_val_brier:
            best_val_brier = val_metrics["brier_score"]
            best_weights = (
                [round(v, 6) for v in w_lin],
                [[round(v, 6) for v in row] for row in w1],
                [round(v, 6) for v in b1],
                [round(v, 6) for v in w2],
                round(b2, 6),
            )

    w_lin, w1, b1, w2, b2 = best_weights

    # Compare G0, G1b, and G5 MLP on the held-out test games
    g0_test = evaluate_predictions(test_samples, lambda st, x: g0_value(st))
    g1_test = evaluate_predictions(test_samples, lambda st, x: g1_value(st, "G1b"))
    g5_test = evaluate_predictions(test_samples, lambda st, x: forward_mlp(x, w_lin, w1, b1, w2, b2))

    g0_val = evaluate_predictions(val_samples, lambda st, x: g0_value(st))
    g1_val = evaluate_predictions(val_samples, lambda st, x: g1_value(st, "G1b"))
    g5_val = evaluate_predictions(val_samples, lambda st, x: forward_mlp(x, w_lin, w1, b1, w2, b2))

    # Measure inference speed
    t0 = perf_counter()
    for _, x, _, _ in test_samples:
        forward_mlp(x, w_lin, w1, b1, w2, b2)
    dt = perf_counter() - t0
    us_per_eval = (dt / max(1, len(test_samples))) * 1e6

    summary = {
        "dataset": str(v2_path.relative_to(ROOT)),
        "total_games": n_games,
        "splits": {
            "train_games": len(train_gids),
            "val_games": len(val_gids),
            "test_games": len(test_gids),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "architecture": "12 -> 16 (ReLU) -> 1 (Sigmoid) + 12-dim linear residual skip (221 parameters)",
        "inference_us_per_eval": round(us_per_eval, 2),
        "validation_metrics": {
            "G0_heuristic": g0_val,
            "G1b_threat_heuristic": g1_val,
            "G5_trained_mlp": g5_val,
        },
        "holdout_test_metrics": {
            "G0_heuristic": g0_test,
            "G1b_threat_heuristic": g1_test,
            "G5_trained_mlp": g5_test,
        },
        "weights": {
            "w_lin": w_lin,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        },
    }

    out_path = ROOT / "Gedix/g5_mlp_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Holdout Test Comparison:")
    print("  G0 Heuristic:       ", g0_test)
    print("  G1b Threat Heuristic:", g1_test)
    print("  G5 Trained MLP:     ", g5_test)
    print(f"  MLP Inference Speed: {us_per_eval:.2f} us/eval")


if __name__ == "__main__":
    main()
