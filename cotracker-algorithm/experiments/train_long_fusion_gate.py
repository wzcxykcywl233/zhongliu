"""Train MLP and full-sequence Mamba fusion gates from cached real labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))

from resources.long_fusion import make_long_fusion_gate


def load_records(directory: Path) -> list[dict[str, object]]:
    records = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(directory.glob("*.pt"))
    ]
    if not records:
        raise FileNotFoundError(f"no cached cases in {directory}")
    return records


def atomic_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluate(model, records, mean, std, device) -> dict[str, float]:
    losses = []
    correct = 0
    total = 0
    global_weight = []
    with torch.no_grad():
        for record in records:
            features = record["features"].to(device)
            target = record["target_global"].to(device)
            logits = model(((features - mean) / std)[None])[0]
            mask = torch.arange(target.shape[0], device=device) > 0
            losses.append(float(F.binary_cross_entropy_with_logits(logits[mask], target[mask])))
            correct += int(((logits[mask] > 0) == (target[mask] > 0.5)).sum())
            total += int(mask.sum())
            global_weight.append(float(torch.sigmoid(logits[mask]).mean()))
    return {
        "loss": float(np.mean(losses)),
        "branch_accuracy": correct / max(total, 1),
        "mean_global_weight": float(np.mean(global_weight)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["mlp", "mamba"], required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = load_records(args.train_cache)
    validation_records = load_records(args.validation_cache)
    all_features = torch.cat([record["features"] for record in train_records], dim=0)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0).clamp_min(1e-6)
    all_targets = torch.cat([record["target_global"][1:] for record in train_records])
    positives = float(all_targets.sum())
    negatives = float(all_targets.numel() - positives)
    positive_weight = max(0.25, min(4.0, negatives / max(positives, 1.0)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "gate_final.pth"
    if final_path.is_file():
        completed = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            completed.get("kind") != args.kind
            or completed.get("steps") != args.steps
            or completed.get("seed") != args.seed
        ):
            raise ValueError(
                "existing final checkpoint does not match requested "
                f"kind/steps/seed: {final_path}"
            )
        print(f"FINAL CHECKPOINT HIT {final_path}", flush=True)
        return 0
    model = make_long_fusion_gate(args.kind).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    start_step = 0
    checkpoints = sorted(args.output_dir.glob("gate_step_*.pth"))
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"RESUME {checkpoints[-1].name} step={start_step}", flush=True)

    mean = feature_mean.to(device)
    std = feature_std.to(device)
    pos_weight = torch.tensor(positive_weight, device=device)
    last_loss = float("nan")
    for step in range(start_step, args.steps):
        record = train_records[step % len(train_records)]
        features = record["features"].to(device)
        target = record["target_global"].to(device)
        difference = (
            record["global_dice"] - record["hierarchical_dice"]
        ).abs().to(device)
        logits = model(((features - mean) / std)[None])[0]
        mask = torch.arange(target.shape[0], device=device) > 0
        quality_weight = 1.0 + (difference * 50.0).clamp(max=4.0)
        loss = F.binary_cross_entropy_with_logits(
            logits[mask],
            target[mask],
            reduction="none",
            pos_weight=pos_weight,
        )
        loss = (loss * quality_weight[mask]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.detach())
        completed_step = step + 1
        if completed_step == 1 or completed_step % 10 == 0:
            print(
                f"kind={args.kind} step={completed_step}/{args.steps} "
                f"case={record['case_id']} loss={last_loss:.6f}",
                flush=True,
            )
        if completed_step % args.save_every == 0:
            checkpoint_path = args.output_dir / f"gate_step_{completed_step:08d}.pth"
            atomic_save(
                {
                    "kind": args.kind,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": completed_step,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "seed": args.seed,
                    "rng_state": rng_state(),
                },
                checkpoint_path,
            )
            for obsolete in sorted(args.output_dir.glob("gate_step_*.pth"))[:-3]:
                obsolete.unlink()

    validation = evaluate(model, validation_records, mean, std, device)
    final = {
        "kind": args.kind,
        "model": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "steps": args.steps,
        "seed": args.seed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "positive_weight": positive_weight,
        "last_training_loss": last_loss,
        "validation": validation,
    }
    atomic_save(final, final_path)
    summary_path = args.output_dir / "training-summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({key: value for key, value in final.items() if key != "model" and not torch.is_tensor(value)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    print(json.dumps(validation, sort_keys=True), flush=True)
    print(f"COMMITTED {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
