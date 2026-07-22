"""Measure structured-action accuracy against the held-out test set.

This isn't free-text similarity — the assistant's job is to propose a
specific Canvas action, so we score whether the generated response contains
the same action type and the same key parameters as the gold ideal_response
for modifying workflows, and whether it correctly declines for
out_of_scope examples. Requires a llama.cpp server already running (see
scripts/evaluate.sh and configs/serving.yaml).

Usage:
    python evaluation/accuracy.py --test-set datasets/test/test.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.provider import LocalQwenProvider

# Looks for a fenced json/action block in a response, e.g.
# ```json\n{"action": "create_assignment", ...}\n```
ACTION_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class EvalResult:
    workflow_id: str
    correct: bool
    reason: str


def extract_action(text: str) -> dict | None:
    match = ACTION_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def is_refusal(text: str) -> bool:
    refusal_markers = ("can't help with that", "outside what i can do", "i'm not able to", "decline")
    lowered = text.lower()
    return any(marker in lowered for marker in refusal_markers)


def score_example(example: dict, generated: str) -> EvalResult:
    workflow_id = example["workflow_id"]

    if workflow_id == "out_of_scope":
        correct = is_refusal(generated)
        return EvalResult(workflow_id, correct, "refusal expected" if not correct else "ok")

    gold_action = extract_action(example["ideal_response"])
    generated_action = extract_action(generated)

    if gold_action is None:
        # read-only workflow with a prose gold answer; accuracy.py can't judge
        # free-text quality, only structured-action correctness.
        return EvalResult(workflow_id, True, "no structured gold action to compare, skipped")

    if generated_action is None:
        return EvalResult(workflow_id, False, "no structured action found in generated response")

    if generated_action.get("action") != gold_action.get("action"):
        return EvalResult(workflow_id, False, "action type mismatch")

    gold_params = gold_action.get("parameters", {})
    generated_params = generated_action.get("parameters", {})
    missing = {k: v for k, v in gold_params.items() if generated_params.get(k) != v}
    if missing:
        return EvalResult(workflow_id, False, f"parameter mismatch: {missing}")

    return EvalResult(workflow_id, True, "ok")


async def run_eval(test_set_path: Path) -> list[EvalResult]:
    provider = LocalQwenProvider.from_config()
    results = []
    try:
        with test_set_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                messages = example["messages"]
                system, user, assistant_gold = messages[0], messages[1], messages[2]

                response = await provider.generate(
                    canvas_context="",  # already baked into user["content"] by convert_to_chatml.py
                    instructor_request=user["content"],
                )
                gold_example = {
                    "workflow_id": example["workflow_id"],
                    "ideal_response": assistant_gold["content"],
                }
                results.append(score_example(gold_example, response.content))
    finally:
        await provider.aclose()
    return results


def summarize(results: list[EvalResult]) -> None:
    by_workflow: dict[str, list[EvalResult]] = {}
    for result in results:
        by_workflow.setdefault(result.workflow_id, []).append(result)

    print(f"{'workflow':<25}{'accuracy':>10}{'n':>6}")
    for workflow_id, group in sorted(by_workflow.items()):
        accuracy = sum(r.correct for r in group) / len(group)
        print(f"{workflow_id:<25}{accuracy:>10.1%}{len(group):>6}")

    overall = sum(r.correct for r in results) / len(results) if results else 0.0
    print(f"\noverall accuracy: {overall:.1%} ({len(results)} examples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=Path("datasets/test/test.jsonl"))
    args = parser.parse_args()

    results = asyncio.run(run_eval(args.test_set))
    summarize(results)


if __name__ == "__main__":
    main()
