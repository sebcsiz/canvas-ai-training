"""Measure structured-intent accuracy against the held-out test set.

Scores whether the generated response is valid JSON matching the gold
ParsedIntent's action type and key parameters, for every one of the 18
IntentAction values (production returns the same shape for all of them —
there's no separate free-text path for read-only actions). Requires a
llama.cpp server already running (see scripts/evaluate.sh and
configs/serving.yaml).

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

# Training/production never emit markdown fencing, but a live model can still
# stray from the system prompt's "no markdown fences" instruction — tolerate
# a fenced block rather than hard-failing the example over formatting alone.
FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class EvalResult:
    workflow_id: str
    correct: bool
    reason: str


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = FENCED_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def score_example(gold: dict, generated_text: str) -> EvalResult:
    workflow_id = gold.get("action", "unknown")
    generated = extract_json(generated_text)

    if generated is None:
        return EvalResult(workflow_id, False, "response is not valid JSON")

    if generated.get("action") != gold.get("action"):
        return EvalResult(
            workflow_id, False, f"action mismatch: expected {gold.get('action')!r}, got {generated.get('action')!r}"
        )

    gold_params = gold.get("parameters") or {}
    generated_params = generated.get("parameters") or {}
    # Only the gold's non-null fields are checked — null just means "not
    # relevant to this action," so a generated null/omitted match is fine too.
    missing = {
        k: v for k, v in gold_params.items() if v is not None and generated_params.get(k) != v
    }
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
                user, assistant_gold = messages[1], messages[2]

                gold = json.loads(assistant_gold["content"])
                response = await provider.generate_raw(user["content"])
                results.append(score_example(gold, response.content))
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
