"""Latency/throughput benchmark against the served GGUF model.

Validates FR12.4 (local Docker Compose deployment must actually be usable)
before the model is handed off to the main app — an instructor waiting 30
seconds for a preview isn't "safe by design", it's just unusable.

Requires a llama.cpp server already running (see scripts/evaluate.sh).

Usage:
    python evaluation/benchmark.py --requests 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.provider import LocalQwenProvider

SAMPLE_REQUESTS = [
    {
        "course_name": "Software Engineering",
        "course_code": "COSC 499",
        "available_assignments": ["Milestone 1"],
        "instructor_request": "List all assignments in this course.",
    },
    {
        "course_name": "Software Engineering",
        "course_code": "COSC 499",
        "instructor_request": "Create an assignment called 'Milestone 2' worth 10 points, due 2026-03-01.",
    },
    {
        "course_name": "Software Engineering",
        "course_code": "COSC 499",
        "available_assignments": ["Milestone 1"],
        "instructor_request": "Push the Milestone 1 due date back by one week.",
    },
    {
        "course_name": "Software Engineering",
        "course_code": "COSC 499",
        "available_assignments": ["Milestone 1"],
        "instructor_request": "Draft feedback for Student A's Milestone 1 submission.",
    },
]


@dataclass
class RequestTiming:
    total_seconds: float
    output_chars: int


async def timed_request(provider: LocalQwenProvider, request: dict) -> RequestTiming:
    start = time.perf_counter()
    response = await provider.generate(**request)
    elapsed = time.perf_counter() - start
    return RequestTiming(total_seconds=elapsed, output_chars=len(response.content))


async def run_benchmark(num_requests: int) -> list[RequestTiming]:
    provider = LocalQwenProvider.from_config()
    timings = []
    try:
        for i in range(num_requests):
            request = SAMPLE_REQUESTS[i % len(SAMPLE_REQUESTS)]
            timings.append(await timed_request(provider, request))
    finally:
        await provider.aclose()
    return timings


def summarize(timings: list[RequestTiming]) -> None:
    seconds = [t.total_seconds for t in timings]
    print(f"requests: {len(timings)}")
    print(f"mean latency:   {statistics.mean(seconds):.2f}s")
    print(f"median latency: {statistics.median(seconds):.2f}s")
    print(f"p95 latency:    {sorted(seconds)[int(len(seconds) * 0.95) - 1]:.2f}s")
    print(f"min / max:      {min(seconds):.2f}s / {max(seconds):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()

    timings = asyncio.run(run_benchmark(args.requests))
    summarize(timings)


if __name__ == "__main__":
    main()
