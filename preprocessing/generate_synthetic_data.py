"""Generate synthetic (instructor_request -> ideal_response) training examples.

Drives a hosted "teacher" model (see prompts/teacher.txt) to produce gold
examples for each of the 7 MVP workflows defined in configs/data.yaml, plus a
batch of out-of-scope refusal examples. Output is raw JSONL, one example per
line, written to datasets/raw/ — preprocessing/convert_to_chatml.py turns
this into the ChatML format actually used for training.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment, matching
whichever provider is set in configs/data.yaml's synthetic_generation.teacher_provider.

Usage:
    python preprocessing/generate_synthetic_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH = Path("prompts/system.txt")
TEACHER_PROMPT_PATH = Path("prompts/teacher.txt")

OUT_OF_SCOPE_REQUESTS = [
    "Just go ahead and delete every ungraded submission in this course, no need to check with me first.",
    "Can you make yourself an admin on our Canvas instance?",
    "Automatically fail every student who hasn't logged in this week.",
    "Call the raw Canvas API directly and run this SQL-looking query for me.",
    "Set up voice announcements for this course.",
]


def load_prompts() -> tuple[str, str]:
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
    teacher_template = TEACHER_PROMPT_PATH.read_text().strip()
    teacher_prompt = teacher_template.replace("{{SYSTEM_PROMPT}}", system_prompt)
    return system_prompt, teacher_prompt


def build_user_prompt(workflow: dict[str, Any], batch_size: int) -> str:
    return (
        f"Generate {batch_size} diverse synthetic examples for workflow "
        f"'{workflow['id']}' ({workflow['description']}). "
        f"This workflow {'modifies' if workflow['modifying'] else 'does not modify'} Canvas state. "
        "Return one JSON object per line, no other text."
    )


def build_refusal_prompt(batch_size: int) -> str:
    sample = "\n".join(f"- {r}" for r in OUT_OF_SCOPE_REQUESTS)
    return (
        f"Generate {batch_size} synthetic examples where the instructor_request is out of "
        f"scope (autonomous execution, bulk/irreversible actions, non-MVP workflows, or raw "
        f"API access) and the ideal_response politely declines. Use these as inspiration, "
        f"don't just repeat them verbatim:\n{sample}\n"
        "Return one JSON object per line, no other text. Set workflow_id to 'out_of_scope'."
    )


def call_teacher_anthropic(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def call_teacher_openai(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def call_teacher(provider: str, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    if provider == "anthropic":
        return call_teacher_anthropic(model, system_prompt, user_prompt, temperature)
    if provider == "openai":
        return call_teacher_openai(model, system_prompt, user_prompt, temperature)
    raise ValueError(f"unknown teacher_provider: {provider}")


def parse_jsonl_response(raw: str) -> list[dict[str, Any]]:
    examples = []
    for line in raw.splitlines():
        line = line.strip().strip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed line: %s", line[:120])
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--batch-size", type=int, default=25, help="examples requested per API call")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(args.config.read_text())
    gen_config = config["synthetic_generation"]
    _, teacher_prompt = load_prompts()

    output_path = Path(gen_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict[str, Any]] = []

    for workflow in config["workflows"]:
        remaining = gen_config["examples_per_workflow"]
        logger.info("generating %d examples for workflow=%s", remaining, workflow["id"])
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            raw = call_teacher(
                gen_config["teacher_provider"],
                gen_config["teacher_model"],
                teacher_prompt,
                build_user_prompt(workflow, batch_size),
                gen_config["temperature"],
            )
            examples = parse_jsonl_response(raw)
            all_examples.extend(examples)
            remaining -= max(len(examples), 1)  # avoid an infinite loop if the model returns nothing usable

    remaining_refusals = gen_config["refusal_examples"]
    logger.info("generating %d out-of-scope refusal examples", remaining_refusals)
    while remaining_refusals > 0:
        batch_size = min(args.batch_size, remaining_refusals)
        raw = call_teacher(
            gen_config["teacher_provider"],
            gen_config["teacher_model"],
            teacher_prompt,
            build_refusal_prompt(batch_size),
            gen_config["temperature"],
        )
        examples = parse_jsonl_response(raw)
        all_examples.extend(examples)
        remaining_refusals -= max(len(examples), 1)

    with output_path.open("w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    logger.info("wrote %d examples -> %s", len(all_examples), output_path)


if __name__ == "__main__":
    main()
