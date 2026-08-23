#!/usr/bin/env bash

# Resumable TrackRAD evaluation runner. Completed cases are committed to a
# persistent host directory atomically, so an interrupted run only repeats the
# case that was active at the time of interruption.
set -Eeuo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$REPOSITORY_ROOT"

uuidgen_fallback() {
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  else
    cat /proc/sys/kernel/random/uuid
  fi
}

ALGORITHM_DIR="${ALGORITHM_DIR_OVERRIDE:-./cotracker-algorithm}"
DATASET_DIR="${DATASET_DIR_OVERRIDE:-./dataset/labeled}"
RESUME_DIR="${TRACKRAD_RESUME_DIR:?TRACKRAD_RESUME_DIR is required}"
GROUND_TRUTH_PATH="$DATASET_DIR"
PROFILE="${COTRACKER_EXPERIMENT:-baseline}"

mkdir -p "$RESUME_DIR" "$RESUME_DIR/.attempts" "$RESUME_DIR/jobs" "$RESUME_DIR/evaluation"
RESUME_DIR=$(cd "$RESUME_DIR" && pwd -P)
DATASET_DIR=$(cd "$DATASET_DIR" && pwd -P)
ALGORITHM_DIR=$(cd "$ALGORITHM_DIR" && pwd -P)
GROUND_TRUTH_PATH="$DATASET_DIR"

exec 9>"$RESUME_DIR/.run.lock"
if ! flock -n 9; then
  echo "Another resumable evaluation is already using $RESUME_DIR" >&2
  exit 75
fi

echo "=+= Build Algorithm and evaluation containers"
docker build "$ALGORITHM_DIR" \
  --platform=linux/amd64 \
  --tag "trackrad-algorithm-$(basename "$ALGORITHM_DIR")" 2>&1

docker build "./evaluation" \
  --platform=linux/amd64 \
  --tag "trackrad-evaluation" 2>&1

echo "=+= Running Algorithm"
echo "CoTracker experiment: $PROFILE"
echo "Persistent checkpoint: $RESUME_DIR"

shopt -s nullglob
case_folders=("$DATASET_DIR"/*)
if [ "${#case_folders[@]}" -eq 0 ]; then
  echo "No cases found in $DATASET_DIR" >&2
  exit 2
fi

for case_folder in "${case_folders[@]}"; do
  [ -d "$case_folder" ] || continue
  case_id=$(basename "$case_folder")
  completed_dir="$RESUME_DIR/jobs/$case_id"
  completed_output="$completed_dir/output/images/mri-linac-series-targets/output.mha"

  if [ -f "$completed_dir/.complete" ] \
    && [ -f "$completed_dir/prediction.json" ] \
    && [ -s "$completed_output" ]; then
    echo "Checkpoint hit, skipping completed case: $case_id"
    continue
  fi

  if [ -e "$completed_dir" ]; then
    recovered="$RESUME_DIR/.attempts/${case_id}-incomplete-$(uuidgen_fallback)"
    echo "Moving incomplete checkpoint aside: $completed_dir -> $recovered"
    mv "$completed_dir" "$recovered"
  fi

  attempt_id="$(uuidgen_fallback)"
  attempt_dir="$RESUME_DIR/.attempts/${case_id}-${attempt_id}"
  mkdir -p "$attempt_dir/output"
  echo "Running algorithm for case: $case_id"
  echo "Attempt directory: $attempt_dir"

  start_time=$(date +"%Y-%m-%dT%H:%M:%S.%6NZ")
  set +e
  docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --env COTRACKER_EXPERIMENT="$PROFILE" \
    --volume "$case_folder/frame-rate.json":/input/frame-rate.json:ro \
    --volume "$case_folder/b-field-strength.json":/input/b-field-strength.json:ro \
    --volume "$case_folder/scanned-region.json":/input/scanned-region.json:ro \
    --volume "$case_folder/images/${case_id}_frames.mha":/input/images/mri-linacs/${case_id}_frames.mha:ro \
    --volume "$case_folder/targets/${case_id}_first_label.mha":/input/images/mri-linac-target/target.mha:ro \
    --volume "$attempt_dir/output":/output \
    "trackrad-algorithm-$(basename "$ALGORITHM_DIR")" 2>&1 \
    | tee -a "$attempt_dir/case.log"
  algorithm_status=${PIPESTATUS[0]}
  set -e
  if [ "$algorithm_status" -ne 0 ]; then
    echo "Case $case_id failed with exit code $algorithm_status; checkpoint not committed" >&2
    exit "$algorithm_status"
  fi
  end_time=$(date +"%Y-%m-%dT%H:%M:%S.%6NZ")

  if [ ! -s "$attempt_dir/output/images/mri-linac-series-targets/output.mha" ]; then
    echo "Case $case_id did not produce output.mha; checkpoint not committed" >&2
    exit 3
  fi

  python3 - "$attempt_dir/prediction.json" "$case_id" "$start_time" "$end_time" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
case_id, start_time, end_time = sys.argv[2:]
payload = {
    "pk": f"jobs/{case_id}",
    "inputs": [
        {"value": 8, "interface": {"slug": "frame-rate"}},
        {"value": 1.5, "interface": {"slug": "magnetic-field-strength"}},
        {"value": "abdomen", "interface": {"slug": "scanned-region"}},
        {
            "image": {"name": "mri-linac-target.mha"},
            "interface": {
                "slug": "mri-linac-target",
                "relative_path": "images/mri-linac-target",
            },
        },
        {
            "image": {"name": f"{case_id}.mha"},
            "interface": {
                "slug": "mri-linac-series",
                "relative_path": "images/mri-linacs",
            },
        },
    ],
    "status": "Succeeded",
    "outputs": [
        {
            "image": {"name": "output.mha"},
            "interface": {
                "slug": "mri-linac-series-targets",
                "relative_path": "images/mri-linac-series-targets",
            },
        }
    ],
    "started_at": start_time,
    "completed_at": end_time,
}
temporary = path.with_suffix(".json.tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY

  printf '%s\n' "$end_time" > "$attempt_dir/.complete"
  sync \
    "$attempt_dir/output/images/mri-linac-series-targets/output.mha" \
    "$attempt_dir/prediction.json" \
    "$attempt_dir/.complete" || true
  if [ -e "$completed_dir" ]; then
    echo "Refusing to overwrite unexpected checkpoint directory: $completed_dir" >&2
    exit 4
  fi
  mv "$attempt_dir" "$completed_dir"
  echo "Checkpoint committed: $case_id"
done

python3 - "$RESUME_DIR" <<'PY'
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
predictions = []
for path in sorted((root / "jobs").glob("*/prediction.json")):
    predictions.append(json.loads(path.read_text(encoding="utf-8")))
temporary = root / "predictions.json.tmp"
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(predictions, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, root / "predictions.json")
print(f"Prepared {len(predictions)} completed predictions")
PY

echo "=+= Running Evaluation"
docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --gpus all \
  --volume "$RESUME_DIR":/input:ro \
  --volume "$RESUME_DIR/evaluation":/output \
  --volume "$GROUND_TRUTH_PATH":/opt/app/ground_truth:ro \
  "trackrad-evaluation"

echo "metrics.json:"
cat "$RESUME_DIR/evaluation/metrics.json"
echo ""
