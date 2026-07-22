"""Convert raw synthetic examples into Qwen3 ChatML training examples.

Input: datasets/raw/synthetic_examples.jsonl, one JSON object per line
matching the shape prompts/teacher.txt asks the teacher model to produce:
{workflow_id, course_name, course_code, course_timezone, today_date,
available_assignments, recent_messages, difficulty, quiz_requirements,
instructor_request, ideal_response}. "ideal_response" is itself the
ParsedIntent object — see prompts/system.txt's "Output format" section,
ported verbatim from the main app's app/mcp-server/src/schemas/intent.py.

Output: datasets/processed/chatml/examples.jsonl, one JSON object per line
with a "messages" list (system/user/assistant) in the format TRL's
SFTTrainer / a tokenizer's apply_chat_template expects, plus the workflow_id
carried through for stratified splitting. The user turn is built with
preprocessing/user_turn.py so it exactly matches the main app's
PromptBuilder._build_user_turn, and the assistant turn is the raw
ParsedIntent JSON (no markdown fencing, no prose) — production parses the
model's output directly as JSON (OpenAI structured outputs), so training on
anything else would teach the wrong output shape.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.user_turn import build_user_turn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH = Path("prompts/system.txt")


def load_student_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/raw/synthetic_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/chatml/examples.jsonl"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    student_prompt = load_student_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            required = {"workflow_id", "today_date", "instructor_request", "ideal_response"}
            if not required.issubset(example) or not isinstance(example["ideal_response"], dict):
                skipped += 1
                continue

            user_turn = build_user_turn(
                instructor_request=example["instructor_request"],
                today_date=example["today_date"],
                course_timezone=example.get("course_timezone"),
                course_name=example.get("course_name"),
                course_code=example.get("course_code"),
                available_assignments=example.get("available_assignments"),
                recent_messages=example.get("recent_messages"),
                difficulty=example.get("difficulty"),
                quiz_requirements=example.get("quiz_requirements"),
            )
            assistant_turn = json.dumps(example["ideal_response"], separators=(",", ":"))

            record = {
                "workflow_id": example["workflow_id"],
                "messages": [
                    {"role": "system", "content": student_prompt},
                    {"role": "user", "content": user_turn},
                    {"role": "assistant", "content": assistant_turn},
                ],
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

    logger.info("wrote %d examples -> %s (%d skipped, missing fields)", written, args.output, skipped)


if __name__ == "__main__":
    main()
