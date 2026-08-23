#!/usr/bin/env python3
"""Run named CoTracker profiles through the repository's unified evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

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


def atomic_write_json(path: Path, payload: object) -> None:
    """Replace a JSON checkpoint only after its complete contents reach disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid checkpoint file: {path}") from exc


def run_profile(profile: str, dataset_dir: Path, output_dir: Path) -> dict:
    profile_dir = output_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = profile_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"===== SKIP {profile}: completed checkpoint found =====", flush=True)
        return load_json(metrics_path, {})

    env = os.environ.copy()
    env.update(
        {
            "COTRACKER_EXPERIMENT": profile,
            "ALGORITHM_DIR_OVERRIDE": str(ALGORITHM_DIR),
            "DATASET_DIR_OVERRIDE": str(dataset_dir),
            "TRACKRAD_RESUME_DIR": str(profile_dir / "checkpoint"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = profile_dir / "console.log"
    lines: list[str] = []
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        started = datetime.now(timezone.utc).astimezone().isoformat()
        banner = f"\n===== RESUME {profile} at {started} =====\n"
        print(banner, end="", flush=True)
        log_handle.write(banner)
        process = subprocess.Popen(
            ["bash", str(REPOSITORY_ROOT / "test-algorithm-resumable.sh")],
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
                lines.append(line)
            return_code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise

    output = "".join(lines)
    if return_code != 0:
        raise RuntimeError(f"{profile} failed with exit code {return_code}")
    metrics = extract_metrics(output)
    atomic_write_json(metrics_path, metrics)
    return metrics


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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep running later profiles while retaining failed-profile checkpoints",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "run-state.json"
    state = load_json(state_path, {})
    resolved_dataset = str(dataset_dir)
    if state and state.get("dataset") != resolved_dataset:
        parser.error(
            "output directory belongs to a different dataset: "
            f"{state.get('dataset')}"
        )
    if not state:
        atomic_write_json(
            state_path,
            {
                "dataset": resolved_dataset,
                "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            },
        )

    print("Profiles:", ", ".join(args.profiles))
    print("Dataset:", dataset_dir)
    print("Output:", output_dir)
    if args.dry_run:
        return 0

    summary_path = output_dir / "summary.json"
    results: dict[str, dict] = load_json(summary_path, {})
    failures: list[str] = []
    for profile in args.profiles:
        print(f"\n===== {profile} =====")
        try:
            results[profile] = run_profile(profile, dataset_dir, output_dir)
            atomic_write_json(summary_path, results)
        except Exception as exc:
            failures.append(profile)
            print(f"===== FAILED {profile}: {exc} =====", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
    if failures:
        print("Failed profiles:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
