"""Run inside the inference container. No training, tuning, or test-label access."""
import argparse
import contextlib
import csv
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from chain_diagnostics import ChainTrace, compare_traces, file_sha256, overview_rows

PROFILES = ("memory_control", "memory_control_repeat", "memory_pointwise_fusion", "memory_current_retrieval")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.flush()
    def flush(self):
        self.stream.flush()
        self.log.flush()


def valid_checkpoint(folder):
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        return all((folder / name).is_file() and file_sha256(folder / name) == sha
                   for name, sha in record["files"].items()) and set(record["files"]) == {"trace.npz", "trace-meta.json", "run.json"}
    except (OSError, ValueError, KeyError):
        return False


def run_case(case, profile, output, previous, sample_points):
    import SimpleITK as sitk
    from model import run_algorithm

    destination = output / "jobs" / case.name / profile
    if valid_checkpoint(destination):
        print(f"CHECKPOINT HIT {profile}/{case.name}", flush=True)
        return destination
    attempts = output / ".attempts"
    attempts.mkdir(exist_ok=True)
    if destination.exists():
        destination.rename(attempts / f"recovered-{case.name}-{profile}-{uuid.uuid4().hex[:8]}")
    attempt = attempts / f"{case.name}-{profile}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir()
    print(f"RUNNING {profile}/{case.name}; attempt={attempt}", flush=True)
    actual_profile = "memory_control" if profile == "memory_control_repeat" else profile
    os.environ["COTRACKER_EXPERIMENT"] = actual_profile
    os.environ["TRACKRAD_DIAGNOSTICS"] = "1"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    trace = ChainTrace(sample_points)
    frames = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "images" / f"{case.name}_frames.mha")))
    target = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "targets" / f"{case.name}_first_label.mha")))
    def metadata(name):
        return json.loads((case / name).read_text(encoding="utf-8-sig"))
    started = time.perf_counter()
    with (attempt / "case.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        prediction = run_algorithm(frames, target, metadata("frame-rate.json"),
                                   metadata("b-field-strength.json"), metadata("scanned-region.json"),
                                   _chain_trace=trace)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    import hashlib
    prediction_hash = hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
    previous_match = None
    prior_file = previous / actual_profile / "checkpoint" / "jobs" / case.name / "diagnostics.json" if previous else None
    if prior_file and prior_file.is_file():
        prior = json.loads(prior_file.read_text(encoding="utf-8-sig"))
        previous_match = prediction_hash == prior["prediction"]["array_sha256"]
    trace.save(attempt)
    atomic_json(attempt / "run.json", {"profile": profile, "case": case.name,
                "array_sha256": prediction_hash, "previous_output_matches": previous_match,
                "seconds_with_observation": elapsed, "note": "Diagnostic timing includes observation overhead; not a speed benchmark."})
    atomic_json(attempt / "complete.json", {"files": {name: file_sha256(attempt / name)
                for name in ("trace.npz", "trace-meta.json", "run.json")}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt.rename(destination)
    del trace, prediction, frames, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"COMMITTED {profile}/{case.name}; prior_output_matches={previous_match}", flush=True)
    return destination


def summarize(output, cases, split):
    rows, case_rows, coverage = [], [], []
    for case in cases:
        ref = output / "jobs" / case.name / "memory_control"
        repeat = compare_traces(ref, output / "jobs" / case.name / "memory_control_repeat")
        for profile in PROFILES[1:]:
            candidate = output / "jobs" / case.name / profile
            report = repeat if profile == "memory_control_repeat" else compare_traces(ref, candidate)
            atomic_json(candidate / "comparison.json", report)
            run = json.loads((candidate / "run.json").read_text())
            control_run = json.loads((ref / "run.json").read_text())
            paired_valid = repeat["all_saved_arrays_exact"] and run["previous_output_matches"] is not False and control_run["previous_output_matches"] is not False
            case_rows.append({"Split": split, "Case": case.name, "Profile": profile,
                              "ControlRepeatExact": repeat["all_saved_arrays_exact"],
                              "ControlPriorMatch": control_run["previous_output_matches"],
                              "CandidatePriorMatch": run["previous_output_matches"],
                              "PairChecksPassed": paired_valid,
                              "MatchedSegments": report["matched_segments"],
                              "ReferenceSegments": report["reference_segments"],
                              "CandidateSegments": report["candidate_segments"],
                              "MissingFeaturePairs": len(report["missing_feature_pairs"])})
            for kind in ("unmatched_reference", "unmatched_candidate"):
                for segment in report[kind]:
                    coverage.append({"Split": split, "Case": case.name, "Profile": profile, "Side": kind, "Segment": segment})
            for row in report["rows"]:
                flat = {"Split": split, "Case": case.name, "Profile": profile, "PairChecksPassed": paired_valid}
                for key, value in row.items():
                    if isinstance(value, dict):
                        flat.update({key + "_" + subkey: v for subkey, v in value.items()})
                    else:
                        flat[key] = value
                rows.append(flat)
    for name, values in (("chain-overview.csv", overview_rows(rows)), ("chain-stages.csv", rows), ("chain-cases.csv", case_rows), ("unmatched-segments.csv", coverage)):
        keys = list(dict.fromkeys(key for row in values for key in row)) or ["Split", "Case", "Profile", "Side", "Segment"]
        path = output / name
        with path.with_suffix(".tmp").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(values)
        path.with_suffix(".tmp").replace(path)
    summary = {"split": split, "cases": len(cases), "profiles": list(PROFILES),
               "pair_checks_passed": all(row["PairChecksPassed"] for row in case_rows),
               "prior_output_checks_available": sum(row["CandidatePriorMatch"] is not None for row in case_rows),
               "prior_output_checks_expected": len(case_rows),
               "missing_feature_pairs": sum(row["MissingFeaturePairs"] for row in case_rows),
               "unmatched_segments": len(coverage), "interpretation": "Feature comparisons are sampled and may include changed segment queries. No causal attenuation ratio is assumed."}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["pair_checks_passed"]:
        raise RuntimeError("Repeat/prior-output checks failed; inspect chain-cases.csv before interpreting differences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path)
    parser.add_argument("--split", choices=("test-38", "validation-10"), default="test-38")
    parser.add_argument("--sample-points", type=int, default=64)
    args = parser.parse_args()
    cases = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    expected = 38 if args.split == "test-38" else 10
    if len(cases) != expected or args.sample_points < 1:
        raise ValueError("Unexpected case count or invalid feature sampling")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in diagnostic container; check Docker GPU access before resuming")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "runtime-info.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "split": args.split,
        "case_ids": [case.name for case in cases]})
    for case in cases:
        for profile in PROFILES:
            run_case(case, profile, args.output, args.previous_results, args.sample_points)
    summarize(args.output, cases, args.split)


if __name__ == "__main__":
    main()
