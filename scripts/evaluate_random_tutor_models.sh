#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="${1:?repository root is required}"
dataset_dir="${2:?TrackRAD labeled dataset is required}"
training_root="${3:?training results root is required}"
evaluation_root="${4:?evaluation results root is required}"
shift 4

profiles=("$@")
if [ "${#profiles[@]}" -eq 0 ]; then
  profiles=(
    baseline_single_teacher
    random_tutor_w0025
    random_tutor_w005
    random_tutor_w010
    random_tutor_w020
    same_teacher_control_w010
  )
fi

mkdir -p "$evaluation_root"
for profile in "${profiles[@]}"; do
  checkpoint="$training_root/$profile/cotracker_three_final.pth"
  if [ ! -s "$checkpoint" ]; then
    echo "Missing completed model: $checkpoint" >&2
    exit 2
  fi
  echo "===== EVALUATE $profile ====="
  (
    cd "$repo_root"
    COTRACKER_EXPERIMENT=baseline \
    COTRACKER_CHECKPOINT_OVERRIDE="$checkpoint" \
    DATASET_DIR_OVERRIDE="$dataset_dir" \
    ALGORITHM_DIR_OVERRIDE="$repo_root/cotracker-algorithm" \
    TRACKRAD_RESUME_DIR="$evaluation_root/$profile" \
      bash ./test-algorithm-resumable.sh
  ) 2>&1 | tee -a "$evaluation_root/$profile.log"
done
