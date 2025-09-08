#!/usr/bin/env bash

# replacement for unavailable uuidgen
uuidgen() {
  echo "$(date +%s%N)$$$RANDOM" \
    | md5sum \
    | awk '{print substr($1,1,8)"-"substr($1,9,4)"-"substr($1,13,4)"-"substr($1,17,4)"-"substr($1,21)}'
}

# This script replicates the grand-challenge evaluation process

# If you want to test the algorithm and evaluation method locally, you can use this script
# to run the algorithm and evaluation inside a docker container.

# Stop at first error
set -e

# The path to the algorithm to test.
ALGORITHM_DIR="./cotracker-algorithm"

# The folder to the cases to test the algorithm on
# For initial local testing the provided minimal example dataset can be used
# For testing prior to submission we recommend to test with the full labeled dataset
DATASET_DIR="./dataset/labeled/" # minimal example dataset

# override variables if os environment variables are set
if [ -n "$ALGORITHM_DIR_OVERRIDE" ]; then
  ALGORITHM_DIR=$ALGORITHM_DIR_OVERRIDE
fi
if [ -n "$DATASET_DIR_OVERRIDE" ]; then
  DATASET_DIR=$DATASET_DIR_OVERRIDE
fi

# The path to the ground truth for the evaluation 
# For local testing the full dataset can be used, as the folder structure is the same
GROUND_TRUTH_PATH=$DATASET_DIR

# Docker volume for temporary storage of predictions and metrics
# Use uuidgen to create a unique volume name to avoid conflicts
TRACKRAD_VOLUME=trackrad-volume-$(uuidgen)

# Create a volume for temporary storage of predictions and metrics
# Remove the volume if it exists
docker volume rm -f $TRACKRAD_VOLUME > /dev/null
docker volume create $TRACKRAD_VOLUME > /dev/null

echo "=+= Build Algorithm and evaluation containers" 

docker build "$ALGORITHM_DIR" \
  --platform=linux/amd64 \
  --tag "trackrad-algorithm-$(basename $ALGORITHM_DIR)" 2>&1

docker build "./evaluation" \
  --quiet \
  --platform=linux/amd64 \
  --tag "trackrad-evaluation" 2>&1

echo "=+= Running Algorithm"

# Iterate over all cases/subfolders in the dataset
for case_folder in $DATASET_DIR/*; do

job_id=$(uuidgen)
case_id=$(basename $case_folder)

# Absolute path to the case folder for mounting
case_path=$(realpath $case_folder)
echo "Running algorithm for case: $case_id"

docker run --rm \
    --quiet \
    --env job_id=$job_id \
    --volume $TRACKRAD_VOLUME:/output \
    alpine:latest \
    /bin/sh -c 'mkdir -m 777 -p /output/${job_id}/output'

start_time=$(date +"%Y-%m-%dT%H:%M:%S.%6NZ")

## Note the extra arguments that are passed here:
# '--network none'
#    entails there is no internet connection
# 'gpus all'
#    enables access to any GPUs present
# '--volume <NAME>:/tmp'
#   is added because on Grand Challenge this directory cannot be used to store permanent files
# The input files are mounted as read-only, following the grand-challenge interfaces
docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --gpus all \
  --volume "$case_path/frame-rate.json":/input/frame-rate.json:ro \
  --volume "$case_path/b-field-strength.json":/input/b-field-strength.json:ro \
  --volume "$case_path/scanned-region.json":/input/scanned-region.json:ro \
  --volume "$case_path/images/${case_id}_frames.mha":/input/images/mri-linacs/${case_id}_frames.mha:ro \
  --volume "$case_path/targets/${case_id}_first_label.mha":/input/images/mri-linac-target/target.mha:ro \
  --mount type=volume,src=$TRACKRAD_VOLUME,dst=/output,volume-subpath=$job_id/output \
  "trackrad-algorithm-$(basename $ALGORITHM_DIR)"

end_time=$(date +"%Y-%m-%dT%H:%M:%S.%6NZ")

# Create a minimal prediction.json file for this case
docker run --rm \
    --quiet \
    --env case_id=$case_id \
    --env job_id=$job_id \
    --env end_time=$end_time \
    --env start_time=$start_time \
    --mount type=volume,src=$TRACKRAD_VOLUME,dst=/output,volume-subpath=$job_id \
    alpine:latest \
    /bin/sh -c 'cat > /output/prediction.json << EOF 
{
  "pk": "$job_id",
  "inputs": [
    {
      "value": 8,
      "interface": {
        "slug": "frame-rate"
      }
    },
    {
      "value": 1.5,
      "interface": {
        "slug": "magnetic-field-strength"
      }
    },
    {
      "value": "abdomen",
      "interface": {
        "slug": "scanned-region"
      }
    },
    {
      "image": {
        "name": "mri-linac-target.mha"
      },
      "interface": {
        "slug": "mri-linac-target",
        "relative_path": "images/mri-linac-target"
      }
    },
    {
      "image": {
        "name": "$case_id.mha"
      },
      "interface": {
        "slug": "mri-linac-series",
        "relative_path": "images/mri-linacs"
      }
    }
  ],
  "status": "Succeeded",
  "outputs": [
    {
      "image": {
        "name": "output.mha"
      },
      "interface": {
        "slug": "mri-linac-series-targets",
        "relative_path": "images/mri-linac-series-targets"
      }
    }
  ],
  "started_at": "$start_time",
  "completed_at": "$end_time"
}'

done

# Combine all predictions into a single file
# Use bash to merge /output/*/prediction.json files into /output/predictions.json separeted by \n,\n
# Does not use jq because it is not available in the evaluation container
docker run --rm \
    --mount type=volume,src=$TRACKRAD_VOLUME,dst=/output \
    alpine:latest \
    /bin/sh -c "echo '[' > /output/predictions.json && \
      find /output/*/prediction.json -exec sh -c 'cat {} && echo \",\"' \; | sed '$ s/,$//' >> /output/predictions.json &&\
      echo ']' >> /output/predictions.json"

# Maybe view the current content of the volume for debugging
#docker run --rm \
#    --mount type=volume,src=trackrad-volume,dst=/output \
#    alpine:latest \
#    /bin/sh -c "ls -la /output/*"

echo "=+= Running Evaluation"

# Prepare output folder for the evaluations
docker run --rm \
    --quiet \
    --env job_id=$job_id \
    --volume $TRACKRAD_VOLUME:/output \
    alpine:latest \
    /bin/sh -c 'mkdir -m 777 -p /output/evaluation'

# Perform the evaluation of the preditions
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --mount type=volume,src=$TRACKRAD_VOLUME,dst=/input \
    --mount type=volume,src=$TRACKRAD_VOLUME,dst=/output,volume-subpath=evaluation \
    --volume "$GROUND_TRUTH_PATH":/opt/app/ground_truth \
    "trackrad-evaluation"

# Finally print the metrics from the volume to the local folder
echo "metrics.json:"
docker run --rm \
    --quiet \
    --mount type=volume,src=$TRACKRAD_VOLUME,dst=/output \
    alpine:latest /bin/sh -c 'cat /output/evaluation/metrics.json'
echo ""

# Delete the volume since it is not needed anymore
docker volume rm -f $TRACKRAD_VOLUME > /dev/null
