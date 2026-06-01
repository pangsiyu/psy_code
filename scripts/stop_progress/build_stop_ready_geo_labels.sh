#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-/data3/psy_code/conda_envs/janusvln/bin/python}"
JANUS_JSON="${JANUS_JSON:-data/train_r2r_only.json}"
R2R_JSON_GZ="${R2R_JSON_GZ:-data/datasets/r2r/train/train.json.gz}"
R2R_GT_JSON_GZ="${R2R_GT_JSON_GZ:-data/datasets/r2r/train/train_gt.json.gz}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/train_r2r_only_stop_ready_geo.jsonl}"
SUCCESS_RADIUS="${SUCCESS_RADIUS:-3.0}"
DISTANCE_TYPE="${DISTANCE_TYPE:-geodesic}"
MAX_RECORDS="${MAX_RECORDS:--1}"

"${PYTHON}" tools/build_r2r_distance_stop_labels.py \
  --janus_json "${JANUS_JSON}" \
  --r2r_json_gz "${R2R_JSON_GZ}" \
  --r2r_gt_json_gz "${R2R_GT_JSON_GZ}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --success_radius "${SUCCESS_RADIUS}" \
  --distance_type "${DISTANCE_TYPE}" \
  --max_records "${MAX_RECORDS}"
