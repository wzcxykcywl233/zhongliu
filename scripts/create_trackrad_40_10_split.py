"""Create a deterministic, patient-level 40/10 split of TrackRAD's 50 cases.

The source dataset is never modified.  Output trees use hard links so the MHA
payloads do not consume a second copy of disk space on the same NTFS volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


COHORT_VALIDATION_QUOTAS = {"A": 5, "B": 3, "C": 2}


def read_scalar_json(case_dir: Path, *names: str):
    for name in names:
        path = case_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing metadata {names!r} in {case_dir}")


def stable_rank(seed: str, case_name: str) -> str:
    return hashlib.sha256(f"{seed}:{case_name}".encode("utf-8")).hexdigest()


def allocate_validation_counts(
    region_counts: dict[str, int], validation_count: int
) -> dict[str, int]:
    """Largest-remainder allocation of a cohort quota across body regions."""

    total = sum(region_counts.values())
    if validation_count < 0 or validation_count > total:
        raise ValueError("Invalid validation quota")
    if total == 0:
        return {}

    raw = {
        region: count * validation_count / total
        for region, count in region_counts.items()
    }
    allocation = {region: int(value) for region, value in raw.items()}
    remaining = validation_count - sum(allocation.values())
    order = sorted(
        region_counts,
        key=lambda region: (-(raw[region] - allocation[region]), region),
    )
    for region in order:
        if remaining == 0:
            break
        if allocation[region] < region_counts[region]:
            allocation[region] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate validation quota")
    return allocation


def select_split(cases: Iterable[dict], seed: str) -> tuple[list[dict], list[dict]]:
    cases = list(cases)
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_cohort[case["cohort"]].append(case)

    actual_cohorts = {cohort: len(items) for cohort, items in by_cohort.items()}
    expected_cohorts = {"A": 25, "B": 15, "C": 10}
    if actual_cohorts != expected_cohorts:
        raise ValueError(
            f"Expected cohort counts {expected_cohorts}, found {actual_cohorts}"
        )

    validation_names: set[str] = set()
    for cohort, quota in COHORT_VALIDATION_QUOTAS.items():
        cohort_cases = by_cohort[cohort]
        region_counts = Counter(case["scanned_region"] for case in cohort_cases)
        region_quotas = allocate_validation_counts(dict(region_counts), quota)
        by_region: dict[str, list[dict]] = defaultdict(list)
        for case in cohort_cases:
            by_region[case["scanned_region"]].append(case)
        for region, region_quota in region_quotas.items():
            ranked = sorted(
                by_region[region], key=lambda case: stable_rank(seed, case["case"])
            )
            validation_names.update(case["case"] for case in ranked[:region_quota])

    training = sorted(
        (case for case in cases if case["case"] not in validation_names),
        key=lambda case: case["case"],
    )
    validation = sorted(
        (case for case in cases if case["case"] in validation_names),
        key=lambda case: case["case"],
    )
    if len(training) != 40 or len(validation) != 10:
        raise RuntimeError(
            f"Split has {len(training)} training and {len(validation)} validation cases"
        )
    return training, validation


def link_case(source_case: Path, destination_root: Path) -> None:
    destination_case = destination_root / source_case.name
    destination_case.mkdir(parents=True, exist_ok=True)
    for source in source_case.rglob("*"):
        relative = source.relative_to(source_case)
        target = destination_case / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or not os.path.samefile(source, target):
                raise FileExistsError(
                    f"Refusing to replace a non-matching split file: {target}"
                )
            continue
        os.link(source, target)


def summarize(cases: list[dict]) -> dict:
    return {
        "cases": len(cases),
        "cohorts": dict(sorted(Counter(case["cohort"] for case in cases).items())),
        "field_strengths": dict(
            sorted(Counter(str(case["field_strength_t"]) for case in cases).items())
        ),
        "scanned_regions": dict(
            sorted(Counter(case["scanned_region"] for case in cases).items())
        ),
    }


def write_atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="trackrad-40-10-v1")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")

    case_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if len(case_dirs) != 50:
        raise ValueError(f"Expected 50 source cases, found {len(case_dirs)}")

    cases = []
    for case_dir in case_dirs:
        cohort = case_dir.name.split("_", 1)[0]
        cases.append(
            {
                "case": case_dir.name,
                "cohort": cohort,
                "field_strength_t": read_scalar_json(
                    case_dir, "b-field-strength.json", "field-strength.json"
                ),
                "frame_rate_hz": read_scalar_json(case_dir, "frame-rate.json"),
                "scanned_region": read_scalar_json(case_dir, "scanned-region.json"),
                "source": str(case_dir),
            }
        )

    training, validation = select_split(cases, args.seed)
    training_root = output_root / "trackrad2025_labeled_train_40"
    validation_root = output_root / "trackrad2025_labeled_validation_10"
    training_root.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    expected_training = {case["case"] for case in training}
    expected_validation = {case["case"] for case in validation}
    existing_training = {path.name for path in training_root.iterdir() if path.is_dir()}
    existing_validation = {
        path.name for path in validation_root.iterdir() if path.is_dir()
    }
    if existing_training - expected_training:
        raise RuntimeError(
            f"Unexpected cases already exist in {training_root}: "
            f"{sorted(existing_training - expected_training)}"
        )
    if existing_validation - expected_validation:
        raise RuntimeError(
            f"Unexpected cases already exist in {validation_root}: "
            f"{sorted(existing_validation - expected_validation)}"
        )

    for case in training:
        link_case(Path(case["source"]), training_root)
    for case in validation:
        link_case(Path(case["source"]), validation_root)

    manifest = args.manifest or output_root / "trackrad-40-10-split.json"
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "method": "patient-level cohort quota plus within-cohort region largest-remainder stratification; SHA-256 seeded ranking",
        "source": str(source),
        "training_root": str(training_root),
        "validation_root": str(validation_root),
        "training": training,
        "validation": validation,
        "training_summary": summarize(training),
        "validation_summary": summarize(validation),
    }
    write_atomic_json(manifest, payload)

    print(f"Training cases:   {len(training)} -> {training_root}")
    print(f"Validation cases: {len(validation)} -> {validation_root}")
    print(f"Manifest:         {manifest}")
    print("Validation IDs:   " + ", ".join(case["case"] for case in validation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

