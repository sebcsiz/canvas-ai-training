"""Check whether the model invents Canvas resources not present in context.

Feeds prompts that reference specific assignments/students/dates, with the
retrieval-grounded context deliberately withheld, and checks whether the
model still fabricates specifics (a made-up due date, a student name that
was never given) instead of asking a clarifying question — the behavior
prompts/student.txt requires. This is the RAG-specific counterpart to
evaluation/accuracy.py's structured-action scoring.

Requires a llama.cpp server already running.

Usage:
    python evaluation/hallucination_test.py
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.provider import LocalQwenProvider

# Prompts that reference a resource whose specifics are deliberately absent
# from canvas_context, so any concrete detail in the response is invented.
UNGROUNDED_PROMPTS = [
    ("Course: COSC 499.", "What's the due date for the Milestone 3 assignment?"),
    ("Course: COSC 499.", "Draft feedback for Alex's submission on the final project."),
    ("Course: COSC 499.", "How many points is the midterm quiz worth?"),
]

# A concrete-looking detail with nothing in context to have grounded it in —
# dates, percentages, or a proper name not present anywhere in the prompt.
FABRICATION_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d+\s?(points|pts|%))\b", re.IGNORECASE
)

CLARIFYING_MARKERS = ("which", "could you clarify", "can you confirm", "don't have", "no information", "not sure which")


@dataclass
class HallucinationResult:
    prompt: str
    response: str
    hallucinated: bool


def looks_like_fabrication(response: str) -> bool:
    if any(marker in response.lower() for marker in CLARIFYING_MARKERS):
        return False
    return bool(FABRICATION_RE.search(response))


async def run_test() -> list[HallucinationResult]:
    provider = LocalQwenProvider.from_config(use_retrieval=False)
    results = []
    try:
        for canvas_context, instructor_request in UNGROUNDED_PROMPTS:
            response = await provider.generate(canvas_context, instructor_request)
            hallucinated = looks_like_fabrication(response.content)
            results.append(HallucinationResult(instructor_request, response.content, hallucinated))
    finally:
        await provider.aclose()
    return results


def summarize(results: list[HallucinationResult]) -> None:
    for result in results:
        status = "HALLUCINATED" if result.hallucinated else "ok"
        print(f"[{status}] {result.prompt}")
        print(f"  -> {result.response[:200]}")

    rate = sum(r.hallucinated for r in results) / len(results) if results else 0.0
    print(f"\nhallucination rate: {rate:.1%} ({len(results)} prompts)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    results = asyncio.run(run_test())
    summarize(results)


if __name__ == "__main__":
    main()
