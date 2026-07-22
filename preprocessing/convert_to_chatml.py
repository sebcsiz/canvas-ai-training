"""Convert raw synthetic examples into Qwen3 ChatML training examples.

Input: datasets/raw/synthetic_examples.jsonl, one JSON object per line with
{workflow_id, canvas_context, instructor_request, ideal_response} (see
prompts/teacher.txt for the generation schema).

Output: datasets/processed/chatml/examples.jsonl, one JSON object per line
with a "messages" list (system/user/assistant) in the format TRL's
SFTTrainer / a tokenizer's apply_chat_template expects, plus the workflow_id
carried through for stratified splitting.

If --with-retrieval is set and a FAISS index already exists at the path in
configs/retrieval.yaml, each user turn is augmented with a "Retrieved
context" block from embeddings/retrieve.py, so the model is trained on the
same input shape it will see in production (see prompts/student.txt).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STUDENT_PROMPT_PATH = Path("prompts/student.txt")
SYSTEM_PROMPT_PATH = Path("prompts/system.txt")


def load_student_prompt() -> str:
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
    student_template = STUDENT_PROMPT_PATH.read_text().strip()
    return student_template.replace("{{SYSTEM_PROMPT}}", system_prompt)


def build_user_turn(example: dict, retrieved_context: str | None) -> str:
    parts = [f"Canvas context:\n{example['canvas_context']}"]
    if retrieved_context:
        parts.append(f"Retrieved context:\n{retrieved_context}")
    parts.append(f"Instructor request: {example['instructor_request']}")
    return "\n\n".join(parts)


def load_retriever(top_k: int):
    from embeddings.retrieve import Retriever

    try:
        return Retriever.from_config(top_k=top_k)
    except FileNotFoundError:
        logger.warning("no FAISS index found; proceeding without retrieval augmentation")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/raw/synthetic_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/chatml/examples.jsonl"))
    parser.add_argument("--with-retrieval", action="store_true")
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    student_prompt = load_student_prompt()
    retriever = load_retriever(args.retrieval_top_k) if args.with_retrieval else None

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            required = {"workflow_id", "canvas_context", "instructor_request", "ideal_response"}
            if not required.issubset(example):
                skipped += 1
                continue

            retrieved_context = None
            if retriever is not None:
                hits = retriever.retrieve(example["instructor_request"])
                if hits:
                    retrieved_context = "\n".join(f"- {hit.text}" for hit in hits)

            record = {
                "workflow_id": example["workflow_id"],
                "messages": [
                    {"role": "system", "content": student_prompt},
                    {"role": "user", "content": build_user_turn(example, retrieved_context)},
                    {"role": "assistant", "content": example["ideal_response"]},
                ],
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

    logger.info("wrote %d examples -> %s (%d skipped, missing fields)", written, args.output, skipped)


if __name__ == "__main__":
    main()
