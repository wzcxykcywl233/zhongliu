"""Read-only paired diagnosis of the retune point-count observation.

This is exploratory analysis of an already inspected public test split, not a
new blind test or a training procedure. Only official per-case metrics are used.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

FIELDS = {
    "DSC": "dice_similarity_coefficient",
    "HD95": "hausdorff_distance_95",
    "MASD": "surface_distance_average",
    "CD": "center_distance",
    "D98": "relative_d98_dose",
}
HIGHER = {"DSC", "D98"}
VALIDATION = {
    "rt_points_0": 250,
    "rt_points_1": 500,
    "rt_points_2": 750,
    "rt_control": 1000,
    "rt_points_4": 1250,
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics(path, expected_cases):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    records = data["results"]
    if len(records) != expected_cases:
        raise ValueError(f"Expected {expected_cases} results: {path}")
    cases = {}
    for record in records:
        case = record["case_id"]
        if case in cases:
            raise ValueError(f"Duplicate case {case}: {path}")
        values = {key: float(record[field]) for key, field in FIELDS.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite metric {case}: {path}")
        cases[case] = values
    return cases


def paired_rows(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("Case IDs differ between paired profiles")
    rows = []
    for case in sorted(reference):
        row = {"Case": case, "Cohort": case.split("_", 1)[0]}
        for key in FIELDS:
            a, b = reference[case][key], candidate[case][key]
            row[f"Control_{key}"] = a
            row[f"Points250_{key}"] = b
            row[f"Delta_{key}"] = b - a
        rows.append(row)
    return rows


def mean(values):
    return sum(values) / len(values)


def interval(values, rng, draws=4000):
    means = []
    n = len(values)
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def summarize(rows, scope, rng):
    summary = []
    for key in FIELDS:
        values = [row[f"Delta_{key}"] for row in rows]
        oriented = [value if key in HIGHER else -value for value in values]
        low, high = interval(values, rng)
        summary.append({
            "Scope": scope,
            "Metric": key,
            "Cases": len(rows),
            "MeanDelta": mean(values),
            "MedianDelta": sorted(values)[len(values) // 2] if len(values) % 2 else
                mean(sorted(values)[len(values) // 2 - 1:len(values) // 2 + 1]),
            "BootstrapLow95": low,
            "BootstrapHigh95": high,
            "ImprovedCases": sum(value > 1e-8 for value in oriented),
            "WorsenedCases": sum(value < -1e-8 for value in oriented),
            "TiedCases": sum(abs(value) <= 1e-8 for value in oriented),
            "LeaveOneOutMin": min((sum(values) - value) / (len(values) - 1) for value in values),
            "LeaveOneOutMax": max((sum(values) - value) / (len(values) - 1) for value in values),
        })
    return summary


def write_csv(path, rows):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def diagnose(root, output):
    test = root / "final" / "test-38"
    validation = root / "single" / "validation-10"
    paths = {
        "test_control": test / "rt_control" / "metrics.json",
        "test_points250": test / "rt_points_0" / "metrics.json",
    }
    paths.update({f"validation_{name}": validation / name / "metrics.json" for name in VALIDATION})
    loaded = {key: load_metrics(path, 38 if key.startswith("test_") else 10)
              for key, path in paths.items()}
    rows = paired_rows(loaded["test_control"], loaded["test_points250"])
    rng = random.Random(20260924)
    summary = summarize(rows, "all", rng)
    for cohort in sorted({row["Cohort"] for row in rows}):
        subset = [row for row in rows if row["Cohort"] == cohort]
        if len(subset) > 1:
            summary.extend(summarize(subset, cohort, rng))
    validation_ids = set(loaded["validation_rt_control"])
    if any(set(loaded[f"validation_{name}"]) != validation_ids for name in VALIDATION):
        raise ValueError("Validation case IDs differ across point-count profiles")
    validation_rows = []
    for name, count in VALIDATION.items():
        for case, values in sorted(loaded[f"validation_{name}"].items()):
            validation_rows.append({"Profile": name, "Points": count, "Case": case, **values})
    simultaneous = {
        "DSC_and_D98_better": sum(row["Delta_DSC"] > 1e-8 and row["Delta_D98"] > 1e-8 for row in rows),
        "DSC_better_D98_worse": sum(row["Delta_DSC"] > 1e-8 and row["Delta_D98"] < -1e-8 for row in rows),
        "DSC_worse_D98_better": sum(row["Delta_DSC"] < -1e-8 and row["Delta_D98"] > 1e-8 for row in rows),
        "HD95_and_MASD_better": sum(row["Delta_HD95"] < -1e-8 and row["Delta_MASD"] < -1e-8 for row in rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test-38-paired-cases.csv", rows)
    write_csv(output / "test-38-paired-summary.csv", summary)
    write_csv(output / "validation-10-point-counts.csv", validation_rows)
    result = {
        "complete": True,
        "test_cases": 38,
        "validation_cases": 10,
        "test_profiles": ["rt_control", "rt_points_0"],
        "validation_point_counts": VALIDATION,
        "source_metrics_sha256": {str(path.relative_to(root)): digest(path) for path in paths.values()},
        "test_tradeoffs": simultaneous,
        "limitations": "Public test-38 has already been inspected. Case bootstrap is exploratory; repeated frames/patients need not be independent. No p-values or blind-test claim. This diagnosis does not isolate tracking from contour reconstruction.",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
