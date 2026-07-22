#!/usr/bin/env bash
# Runs QLoRA fine-tuning. Pass --resume to continue from the latest checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "--resume" ]]; then
  python training/resume.py --config configs/qlora.yaml
else
  python training/train.py --config configs/qlora.yaml
fi
