#!/usr/bin/env bash
# Merges the trained LoRA adapter into the base model, then exports to GGUF.
# Requires LLAMA_CPP_DIR to point at a built llama.cpp checkout (see
# training/export_model.py's docstring).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1/2 merging LoRA adapter =="
python training/merge_lora.py --config configs/serving.yaml

echo "== 2/2 exporting to GGUF =="
python training/export_model.py --config configs/serving.yaml

echo "done. GGUF weights ready under models/gguf/."
