#!/usr/bin/env bash

# Launch the complete hierarchical re-anchoring matrix. Re-running this command
# with the same output directory resumes completed cases and profiles.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
DATASET_DIR="${1:?Usage: $0 DATASET_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$REPOSITORY_ROOT/hierarchical-results}"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
DATASET_DIR=$(cd "$DATASET_DIR" && pwd -P)

exec 8>"$OUTPUT_DIR/.supervisor.lock"
if ! flock -n 8; then
  echo "Another hierarchical experiment supervisor is using $OUTPUT_DIR" >&2
  exit 75
fi

exec > >(tee -a "$OUTPUT_DIR/supervisor.log") 2>&1
printf '%s\n' "$$" > "$OUTPUT_DIR/supervisor.pid"
echo "===== hierarchical experiments started $(date --iso-8601=seconds) ====="
echo "Repository: $REPOSITORY_ROOT"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"

cd "$REPOSITORY_ROOT"
set +e
python3 -u cotracker-algorithm/experiments/run_ablation.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --continue-on-error \
  --profiles \
    hierarchical_d10 \
    hierarchical_d10_original_feat_05 \
    hierarchical_d10_occlusion_merge \
    hierarchical_d10_dual_anchor \
    hierarchical_full
status=$?
set -e
echo "===== hierarchical experiments finished $(date --iso-8601=seconds), exit=$status ====="
exit "$status"
