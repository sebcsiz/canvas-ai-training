"""Merge the trained LoRA adapter into the base Qwen3-8B weights.

Produces a standalone full model (no PEFT dependency needed to load it),
which training/export_model.py then converts to GGUF. Paths come from
configs/serving.yaml so the merge/export/serve steps agree on locations.

Usage:
    python training/merge_lora.py --config configs/serving.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/serving.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["merge"]

    print(f"loading base model {config['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    print(f"loading adapter from {config['adapter_dir']}")
    model = PeftModel.from_pretrained(base_model, config["adapter_dir"])

    print("merging adapter into base weights")
    merged = model.merge_and_unload()

    output_dir = Path(config["merged_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"saved merged model -> {output_dir}")


if __name__ == "__main__":
    main()
