"""Resume QLoRA training from the most recent checkpoint under output_dir.

Reuses training/train.py's setup so the resumed run uses the exact same
model/LoRA/data configuration — only the checkpoint-selection logic differs.

Usage:
    python training/resume.py --config configs/qlora.yaml
    python training/resume.py --config configs/qlora.yaml --checkpoint outputs/qwen3-8b-canvas-ai/checkpoint-300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import load_config, run_training


def find_latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {output_dir}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="defaults to the latest checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["training"]["output_dir"])

    checkpoint = args.checkpoint or find_latest_checkpoint(output_dir)
    print(f"resuming from {checkpoint}")

    run_training(config, resume_from_checkpoint=str(checkpoint))


if __name__ == "__main__":
    main()
