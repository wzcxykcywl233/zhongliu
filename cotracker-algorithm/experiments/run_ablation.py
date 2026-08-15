#!/usr/bin/env python3
"""Run named CoTracker profiles through the repository's unified evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ALGORITHM_DIR.parent
sys.path.insert(0, str(ALGORITHM_DIR))

from experiments import EXPERIMENTS  # noqa: E402


def extract_metrics(output: str) -> dict:
    marker = "metrics.json:"
    if marker not in output:
        raise RuntimeError("test-algorithm.sh did not print metrics.json")
    payload = output.rsplit(marker, 1)[1].lstrip()
    metrics, _ = json.JSONDecoder().raw_decode(payload)
    return metrics


def run_profile(profile: str, dataset_dir: Path, output_dir: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "COTRACKER_EXPERIMENT": profile,
            "ALGORITHM_DIR_OVERRIDE": str(ALGORITHM_DIR),
            "DATASET_DIR_OVERRIDE": str(dataset_dir),
        }
    )
    process = subprocess.Popen(
        ["bash", str(REPOSITORY_ROOT / "test-algorithm.sh")],
        cwd=REPOSITORY_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return_code = process.wait()
    output = "".join(lines)
    (output_dir / f"{profile}.log").write_text(output, encoding="utf-8")
    if return_code != 0:
        raise RuntimeError(f"{profile} failed with exit code {return_code}")
    return extract_metrics(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("ablation-results"))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Profiles:", ", ".join(args.profiles))
    print("Dataset:", dataset_dir)
    print("Output:", output_dir)
    if args.dry_run:
        return 0

    results: dict[str, dict] = {}
    for profile in args.profiles:
        print(f"\n===== {profile} =====")
        results[profile] = run_profile(profile, dataset_dir, output_dir)
        (output_dir / "summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
