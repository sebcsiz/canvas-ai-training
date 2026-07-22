"""QLoRA fine-tune Qwen3-8B on the ChatML dataset via TRL's SFTTrainer.

All hyperparameters live in configs/qlora.yaml — this script wires them up,
it doesn't hardcode training decisions. See training/resume.py to continue
from a checkpoint, and training/merge_lora.py for what happens after.

Usage:
    python training/train.py --config configs/qlora.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build_model_and_tokenizer(config: dict):
    quant_config = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=quant_config["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"], trust_remote_code=config["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config["trust_remote_code"],
    )
    return model, tokenizer


def build_lora_config(config: dict) -> LoraConfig:
    lora = config["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"],
    )


def build_sft_config(config: dict) -> SFTConfig:
    training = config["training"]
    return SFTConfig(
        output_dir=training["output_dir"],
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        max_seq_length=training["max_seq_length"],
        logging_steps=training["logging_steps"],
        eval_strategy=training["eval_strategy"],
        eval_steps=training["eval_steps"],
        save_strategy=training["save_strategy"],
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        bf16=training["bf16"],
        gradient_checkpointing=training["gradient_checkpointing"],
        seed=training["seed"],
        report_to=training["report_to"],
        packing=False,  # keep one training example per sequence so the assistant turn stays isolated
    )


def run_training(config: dict, resume_from_checkpoint: str | None = None) -> None:
    model, tokenizer = build_model_and_tokenizer(config)
    lora_config = build_lora_config(config)
    sft_config = build_sft_config(config)

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["data"]["train_path"],
            "validation": config["data"]["validation_path"],
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
