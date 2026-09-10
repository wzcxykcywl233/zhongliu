"""Prepare supervised full-sequence fusion tokens from labeled TrackRAD cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk
import torch


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))

import resources


MODEL_SHAPE = (384, 512)
CACHE_SCHEMA = 1
CACHE_PROFILE = "hierarchical_full_grid0_iterations2"


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def dice_per_frame(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    prediction = prediction.bool().flatten(1)
    truth = truth.bool().flatten(1)
    intersection = (prediction & truth).sum(dim=1).float()
    denominator = prediction.sum(dim=1).float() + truth.sum(dim=1).float()
    return torch.where(
        denominator > 0,
        2.0 * intersection / denominator,
        torch.ones_like(denominator),
    )


def tracking_masks(tracking, original_shape: tuple[int, int]) -> torch.Tensor:
    masks = resources.convert_point_trajectory_to_mask_sequence(
        tracking.trajectories,
        video_shape=MODEL_SHAPE,
    )
    masks = resources.reshape_video(masks, target_shape=original_shape)
    return masks[0].cpu()


def process_case(model, case_dir: Path) -> dict[str, object]:
    case_id = case_dir.name
    frames_path = case_dir / "images" / f"{case_id}_frames.mha"
    first_label_path = case_dir / "targets" / f"{case_id}_first_label.mha"
    labels_path = case_dir / "targets" / f"{case_id}_labels.mha"
    for required in (frames_path, first_label_path, labels_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    frames = read_array(frames_path)
    first_label = read_array(first_label_path)
    labels = read_array(labels_path)
    if frames.ndim != 3 or labels.shape != frames.shape:
        raise ValueError(
            f"{case_id}: frames {frames.shape} and labels {labels.shape} do not match"
        )
    width, height, frame_count = frames.shape
    if first_label.shape != (width, height, 1):
        raise ValueError(f"{case_id}: unexpected first label {first_label.shape}")

    video = torch.from_numpy(frames).float().permute(2, 1, 0)[None, :, None]
    query = torch.from_numpy(first_label).float().permute(2, 1, 0)[None]
    video = resources.reshape_video(video, target_shape=MODEL_SHAPE)
    query = resources.reshape_video(query, target_shape=MODEL_SHAPE)
    queries = resources.convert_mask_to_points(query, n_border_points=1000)

    with torch.no_grad():
        context = resources.hierarchical_forward_pass(
            model=model,
            video=video,
            queries=queries,
            span=10,
            support_grid_size=0,
            n_iterations=2,
            original_feature_weight=0.5,
            occlusion_merge=True,
            dual_anchor_weight=0.5,
            return_long_fusion_context=True,
        )
        features = resources.build_long_fusion_features(
            context.hierarchical,
            context.global_result,
            context.hierarchical_similarity,
            context.global_similarity,
            MODEL_SHAPE,
        )[0]
        hierarchical_masks = tracking_masks(context.hierarchical, (height, width))
        global_masks = tracking_masks(context.global_result, (height, width))

    truth = torch.from_numpy(labels).permute(2, 1, 0) > 0
    hierarchical_dice = dice_per_frame(hierarchical_masks, truth)
    global_dice = dice_per_frame(global_masks, truth)
    target_global = (global_dice > hierarchical_dice).float()
    target_global[0] = 0.0
    return {
        "schema": CACHE_SCHEMA,
        "profile": CACHE_PROFILE,
        "case_id": case_id,
        "features": features.cpu().float(),
        "target_global": target_global.cpu(),
        "hierarchical_dice": hierarchical_dice.cpu(),
        "global_dice": global_dice.cpu(),
        "frames": frame_count,
    }


def valid_cache_record(record: object, case_id: str) -> bool:
    if not isinstance(record, dict):
        return False
    features = record.get("features")
    return (
        record.get("schema") == CACHE_SCHEMA
        and record.get("profile") == CACHE_PROFILE
        and record.get("case_id") == case_id
        and torch.is_tensor(features)
        and features.ndim == 2
        and features.shape[-1] == resources.LONG_FUSION_FEATURE_DIM
    )


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.dataset_dir.iterdir() if path.is_dir())
    if len(case_dirs) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} cases, found {len(case_dirs)} in {args.dataset_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = resources.setup_model()
    rows = []
    for index, case_dir in enumerate(case_dirs, start=1):
        output = args.output_dir / f"{case_dir.name}.pt"
        record = None
        if output.is_file() and output.stat().st_size > 0:
            candidate = torch.load(output, map_location="cpu", weights_only=False)
            if valid_cache_record(candidate, case_dir.name):
                record = candidate
                print(
                    f"CACHE HIT {index}/{len(case_dirs)} {case_dir.name}",
                    flush=True,
                )
            else:
                print(
                    f"CACHE INVALID {index}/{len(case_dirs)} {case_dir.name}; regenerating",
                    flush=True,
                )
        if record is None:
            print(f"PREPARE {index}/{len(case_dirs)} {case_dir.name}", flush=True)
            record = process_case(model, case_dir)
            atomic_torch_save(record, output)
            print(f"COMMITTED {case_dir.name}", flush=True)
        hierarchical = record["hierarchical_dice"][1:]
        global_dice = record["global_dice"][1:]
        oracle = torch.maximum(hierarchical, global_dice)
        rows.append(
            {
                "Case": record["case_id"],
                "Frames": int(record["frames"]),
                "HierarchicalFrameDSC": float(hierarchical.mean()),
                "GlobalFrameDSC": float(global_dice.mean()),
                "OracleFrameDSC": float(oracle.mean()),
                "GlobalWins": int((global_dice > hierarchical).sum()),
                "HierarchicalWinsOrTies": int((global_dice <= hierarchical).sum()),
            }
        )

    csv_path = args.output_dir / "oracle-audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": 1,
        "cases": len(rows),
        "frames_excluding_query": sum(row["Frames"] - 1 for row in rows),
        "hierarchical_frame_dsc": float(np.mean([row["HierarchicalFrameDSC"] for row in rows])),
        "global_frame_dsc": float(np.mean([row["GlobalFrameDSC"] for row in rows])),
        "oracle_frame_dsc": float(np.mean([row["OracleFrameDSC"] for row in rows])),
        "global_wins": sum(row["GlobalWins"] for row in rows),
        "hierarchical_wins_or_ties": sum(row["HierarchicalWinsOrTies"] for row in rows),
        "complete": True,
    }
    atomic_json(summary, args.output_dir / "oracle-summary.json")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
