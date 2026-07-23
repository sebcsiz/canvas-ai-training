"""Builds the user turn exactly like the main app's PromptBuilder._build_user_turn.

Ported line-for-line from app/mcp-server/src/services/prompt_builder.py so the
student model trains on (and, via inference/provider.py, is served) the exact
input shape production sends — not an approximation of it. If that file
changes, mirror the change here.

Training examples fix "today" per-example (see prompts/teacher.txt's
"today_date" field) rather than reading the live clock, since a static
anchor date is what makes a training example's date-resolution reproducible.
inference/provider.py calls this with a live "today" instead.
"""

from __future__ import annotations


def _as_text(item: object) -> str:
    """Coerce one list entry (assignment name / message preview) to a plain
    string. The teacher model is instructed to produce plain strings for
    available_assignments/recent_messages, but doesn't always comply for
    recent_messages (occasionally returns {"sender": ..., "text": ...}-style
    objects instead) — normalize rather than crash the whole conversion over
    one non-conformant field in an otherwise-good example."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "preview", "message", "body", "content"):
            value = item.get(key)
            if isinstance(value, str):
                sender = item.get("sender") or item.get("from") or item.get("student_name")
                return f"{sender}: {value}" if sender else value
        import json

        return json.dumps(item)
    return str(item)


def build_user_turn(
    instructor_request: str,
    today_date: str,
    course_timezone: str | None = None,
    course_name: str | None = None,
    course_code: str | None = None,
    available_assignments: list[str] | None = None,
    recent_messages: list[str] | None = None,
    difficulty: str | None = None,
    quiz_requirements: dict | None = None,
) -> str:
    tz_label = course_timezone or "UTC"
    parts = [f"Command: {instructor_request}", f"Today: {today_date} ({tz_label})"]

    if course_name:
        label = course_name
        if course_code:
            label = f"{label} ({course_code})"
        parts.append(f"Course: {label}")

    if available_assignments:
        parts.append(f"Available assignments: {', '.join(_as_text(a) for a in available_assignments)}")

    if recent_messages:
        parts.append(f"Recent messages: {'; '.join(_as_text(m) for m in recent_messages)}")

    if difficulty:
        parts.append(f"Difficulty: {difficulty}")

    quiz_line = build_quiz_requirements_line(quiz_requirements)
    if quiz_line:
        parts.append(quiz_line)

    return "\n".join(parts)


def build_quiz_requirements_line(quiz_requirements: dict | None) -> str | None:
    if not quiz_requirements or not quiz_requirements.get("question_counts"):
        return None

    rows = quiz_requirements["question_counts"]
    parts_text = ", ".join(
        f"{row['count']}x {row['question_type']}"
        + (f" ({_fmt_num(row['points_each'])} pts each)" if row.get("points_each") is not None else "")
        for row in rows
    )
    total_questions = sum(row["count"] for row in rows)
    totals = f"Total: {total_questions} questions"
    if all(row.get("points_each") is not None for row in rows):
        total_points = sum(row["count"] * row["points_each"] for row in rows)
        totals += f", {_fmt_num(total_points)} points"

    return f"Quiz requirements (must match exactly, overrides the default question mix): {parts_text}. {totals}."


def _fmt_num(value: float) -> str:
    # Matches Python's `:g` format used by prompt_builder.py — "5" not "5.0",
    # but "5.5" stays "5.5".
    return f"{value:g}"
