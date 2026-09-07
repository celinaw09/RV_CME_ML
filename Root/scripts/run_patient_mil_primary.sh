#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/allpatients_resized" >&2
  exit 2
fi

data_root="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

python "${project_root}/src/nested_cv_pipeline.py" \
  --data-root "${data_root}" \
  --output-dir "${project_root}/experiment_logs/RV_CME_PATIENT_MIL_PRIMARY" \
  --outer-folds 5 \
  --inner-folds 4 \
  --final-validation-splits 5 \
  --epochs 30 \
  --patience 10 \
  --min-delta 0.0001 \
  --batch-size 8 \
  --num-workers 0 \
  --weight-decay 0.001 \
  --step-size 10 \
  --scheduler-gamma 0.5 \
  --eta-min 0.000001 \
  --focal-gamma 2.0 \
  --seed 42 \
  --gpus mps \
  --bootstrap-iterations 5000 \
  --gradcam-samples-per-category 5
