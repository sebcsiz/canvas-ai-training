"""Redact student-identifiable information from raw Canvas exports.

Input: JSON files under datasets/raw/canvas_export/ (courses, assignments,
submissions, users — pulled separately via the main app's backend, never
directly from Canvas by this repo).

Output: the same documents under datasets/processed/, with student names,
emails, and Canvas user IDs replaced by stable per-document placeholders
(e.g. "Student A"), so downstream embedding and synthetic-data generation
never see real student PII. Mirrors FR10.4 in the main app.

Instructor and course-level information (course names, assignment titles,
dates) is left intact — it's the content the assistant needs to reason
about and is not student-identifiable.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Canvas JSON keys that hold student-identifiable values.
STUDENT_NAME_KEYS = {"name", "short_name", "sortable_name", "display_name"}
STUDENT_ID_KEYS = {"user_id", "id", "sis_user_id", "login_id"}


class PlaceholderRegistry:
    """Maps real student identifiers to stable placeholders within one document."""

    def __init__(self) -> None:
        self._names: dict[str, str] = {}
        self._ids: dict[str, str] = {}

    def name_for(self, real_name: str) -> str:
        if real_name not in self._names:
            index = len(self._names)
            self._names[real_name] = f"Student {chr(ord('A') + index % 26)}{index // 26 or ''}"
        return self._names[real_name]

    def id_for(self, real_id: str) -> str:
        if real_id not in self._ids:
            self._ids[real_id] = f"student_{len(self._ids):04d}"
        return self._ids[real_id]


def redact_text(text: str, registry: PlaceholderRegistry) -> str:
    return EMAIL_RE.sub(lambda m: f"{registry.id_for(m.group(0))}@example.invalid", text)


def redact_node(node: object, registry: PlaceholderRegistry, *, in_student_record: bool) -> object:
    if isinstance(node, dict):
        record_is_student = in_student_record or node.get("type") == "StudentEnrollment"
        out = {}
        for key, value in node.items():
            if record_is_student and key in STUDENT_NAME_KEYS and isinstance(value, str):
                out[key] = registry.name_for(value)
            elif record_is_student and key in STUDENT_ID_KEYS and isinstance(value, (str, int)):
                out[key] = registry.id_for(str(value))
            else:
                out[key] = redact_node(value, registry, in_student_record=record_is_student)
        return out
    if isinstance(node, list):
        return [redact_node(item, registry, in_student_record=in_student_record) for item in node]
    if isinstance(node, str):
        return redact_text(node, registry)
    return node


def clean_file(input_path: Path, output_path: Path) -> None:
    registry = PlaceholderRegistry()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    cleaned = redact_node(data, registry, in_student_record=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, indent=2))
    logger.info(
        "cleaned %s -> %s (%d student identities redacted)",
        input_path,
        output_path,
        len(registry._names) or len(registry._ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw/canvas_export"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/processed"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_files = sorted(args.input_dir.glob("*.json"))
    if not input_files:
        logger.warning("no input files found under %s", args.input_dir)
        return

    for input_path in input_files:
        clean_file(input_path, args.output_dir / input_path.name)


if __name__ == "__main__":
    main()
