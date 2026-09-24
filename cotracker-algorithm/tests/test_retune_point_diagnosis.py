import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from diagnose_retune_points import VALIDATION, diagnose, paired_rows


def test_paired_cases_reject_different_ids():
    try:
        paired_rows({"A_001": {}}, {"A_002": {}})
    except ValueError as exc:
        assert "Case IDs differ" in str(exc)
    else:
        raise AssertionError("mismatched case IDs accepted")


def test_diagnosis_reads_frozen_splits_without_changing_inputs(tmp_path):
    root = tmp_path / "results"
    fields = {
        "dice_similarity_coefficient": 0.8,
        "hausdorff_distance_95": 4.0,
        "surface_distance_average": 2.0,
        "center_distance": 3.0,
        "relative_d98_dose": 0.9,
    }
    paths = []
    for profile, count in [("rt_control", 1000), ("rt_points_0", 250)]:
        path = root / "final" / "test-38" / profile / "metrics.json"
        path.parent.mkdir(parents=True)
        rows = []
        for i in range(38):
            values = dict(fields)
            if count == 250:
                values["dice_similarity_coefficient"] += 0.01
                values["relative_d98_dose"] -= 0.01
            rows.append({"case_id": f"A_{i:03d}", **values})
        path.write_text(json.dumps({"results": rows}), encoding="utf-8")
        paths.append(path)
    for profile in VALIDATION:
        path = root / "single" / "validation-10" / profile / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"results": [
            {"case_id": f"B_{i:03d}", **fields} for i in range(10)
        ]}), encoding="utf-8")
        paths.append(path)
    before = [path.read_bytes() for path in paths]
    output = tmp_path / "output"
    report = diagnose(root, output)
    assert report["test_tradeoffs"]["DSC_better_D98_worse"] == 38
    assert (output / "test-38-paired-cases.csv").is_file()
    assert (output / "validation-10-point-counts.csv").is_file()
    assert before == [path.read_bytes() for path in paths]
