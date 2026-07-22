#!/usr/bin/env bash
# Runs the full data-prep pipeline: redact -> generate synthetic data ->
# convert to ChatML -> split into train/validation/test.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1/4 cleaning raw Canvas exports =="
python preprocessing/clean_canvas_data.py

echo "== 2/4 generating synthetic examples =="
python preprocessing/generate_synthetic_data.py --config configs/data.yaml

echo "== 3/4 converting to ChatML =="
python preprocessing/convert_to_chatml.py "$@"

echo "== 4/4 splitting dataset =="
python preprocessing/split_dataset.py --config configs/data.yaml

echo "done. train/validation/test written under datasets/."
