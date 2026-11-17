"""Train a Wrongway outcome-labeled MLP/CNN on Kaggle or locally.

Place in AI-Project1/training/train_value.py.
Usage from repository root:
 python training/train_value.py --data training/datasets/v1/states.jsonl --model mlp --out training/models/mlp_v1

Requires PyTorch. This script is OFFLINE training; the final Arena agent must
use standard-library-only inference with embedded weights.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from state_encoder import encode_cnn
from value_network import build_model


def split_by_game(rows, seed, train_fraction=0.70, val_fraction=0.15):
    """Keep every state of a single game in the same partition."""
    ids = sorted({str(row["game_id"]) for row in rows})
    if len(ids) < 3:
        raise ValueError("Need at least 3 distinct game_ids for train/val/test")
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = min(n - 2, max(1, round(n * train_fraction)))
    n_val = min(n - n_train - 1, max(1, round(n * val_fraction)))
    groups = {
        "train": ids[:n_train],
        "validation": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    assert all(groups.values())
    return groups


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "game_id" not in row or "state" not in row or "outcome" not in row:
                raise ValueError(f"Missing game_id/state/outcome on line {line_num}")
            if row["outcome"] not in (-1, 0, 1):
                raise ValueError(f"Unsupported outcome on line {line_num}")
            rows.append(row)
    if not rows:
        raise ValueError("No rows in states.jsonl")
    return rows


def tensors_for(rows, ids):
    selected = [row for row in rows if str(row["game_id"]) in ids]
    x = torch.tensor([encode_cnn(row["state"]) for row in selected], dtype=torch.float32)
    y = torch.tensor([row["outcome"] for row in selected], dtype=torch.float32)
    return x, y, selected


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    right_signs = 0
    sign_total = 0
    n = 0
    for x, y in loader:
        prediction = model(x.to(device))
        y = y.to(device)
        squared_error += torch.sum((prediction - y) ** 2).item()
        absolute_error += torch.sum(torch.abs(prediction - y)).item()
        mask = y != 0
        right_signs += ((prediction[mask] >= 0) == (y[mask] > 0)).sum().item()
        sign_total += int(mask.sum().item())
        n += y.numel()
    return {
        "mse": squared_error / n,
        "mae": absolute_error / n,
        "sign_accuracy_non_draw": right_signs / sign_total if sign_total else None,
        "states": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/datasets/v1/states.jsonl")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--out", default="training/models/mlp_v1")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.patience <= 0:
        parser.error("epochs, batch-size, lr, patience must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; try --device cpu")

    path = Path(args.data).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(path)
    groups = split_by_game(rows, seed=args.seed)
    tensors = {part: tensors_for(rows, set(ids)) for part, ids in groups.items()}
    loaders = {}
    for part, (x, y, _) in tensors.items():
        dataset = TensorDataset(x, y)
        loaders[part] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(part == "train"), num_workers=0
        )
    print("Device:", device, "| Model:", args.model)
    for part in groups:
        x, _, _ = tensors[part]
        print(f"{part:>10}: {len(groups[part])} games, {x.shape[0]} states")
    print("First batch input shape:", next(iter(loaders["train"]))[0].shape)

    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    checkpoint = out / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, loaders["train"], device)
        val_metrics = evaluate(model, loaders["validation"], device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(f"Epoch {epoch:02d}: train MSE={train_metrics['mse']:.4f}, "
              f"val MSE={val_metrics['mse']:.4f}")
        if val_metrics["mse"] < best_val - 1e-6:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "architecture": args.model,
                "input_shape": [8, 9, 9],
                "epoch": best_epoch,
                "seed": args.seed,
                "model_state_dict": model.state_dict(),
            }, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: no validation improvement")
                break

    # Load best validation checkpoint, then evaluate test data only once.
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    final = {
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "counts": {part: {"games": len(groups[part]), "states": len(tensors[part][2])}
                   for part in groups},
        "train": evaluate(model, loaders["train"], device),
        "validation": evaluate(model, loaders["validation"], device),
        "test": evaluate(model, loaders["test"], device),
    }
    (out / "splits.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("\nFinal metrics (best validation checkpoint):")
    print(json.dumps(final, indent=2))
    print("Saved to:", out)
    print("NOTE: .pt is a PyTorch checkpoint for research, not a valid single-file Arena submission.")


if __name__ == "__main__":
    main()
