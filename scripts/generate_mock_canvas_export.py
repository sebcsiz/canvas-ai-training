"""Generate a synthetic Canvas API-shaped export for local model training.

Throwaway tooling for the `mock-canvas-data-export` branch — not part of the
regular pipeline (see CLAUDE.md: raw exports are normally "pulled separately
via the main app's backend, never directly from Canvas by this repo"). No
live Canvas instance is available here, so this reproduces the same course
roster and submission-mix rules as the main app's
`infra/local_canvas_instructions/seeds/local_sample_data.rb` seed script,
emitting Canvas API v1 response shapes directly instead of writing to a real
Canvas database.

Beyond what the Ruby seed covers (courses, 2 assignments, one mixed-status
submission set, one discussion/page/module for COSC 499 only), this also
generates quizzes, assignment rubrics, submission comments, announcements,
and teacher-student conversations for every course — so raw data exists for
all 7 MVP workflows (FR3.6): listing, creating assignments, updating
assignment dates, drafting student messages, drafting feedback comments,
creating quizzes, modifying grades.

Output: one JSON file per Canvas resource type under
datasets/raw/canvas_export/, matching the shapes preprocessing/clean_canvas_data.py
expects (STUDENT_NAME_KEYS / STUDENT_ID_KEYS on user-ish records).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUTPUT_DIR = Path("datasets/raw/canvas_export")
STUDENT_COUNT = 180
NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)

MISSING_RATE = 0.15
LATE_RATE = 0.30
UNGRADED_RATE = 0.25

PRIMARY_TEACHER = {"login": "teacher@example.test", "name": "Taylor Teacher"}
MIXED_TYPES = ["online_text_entry", "online_upload", "online_url"]

COURSES = [
    {
        "sis_id": "LOCAL-SAMPLE-COSC-499",
        "name": "Capstone Test COSC 499",
        "code": "COSC 499 CAPSTONE",
        "teacher": PRIMARY_TEACHER,
        "roster_prefix": "cosc499student",
        "roster_label": "COSC 499",
        "syllabus": "<p>Capstone project course. Weekly deliverables, final demo in week 12.</p>",
        "assignments": [
            {"name": "Intro Reflection", "due_offset_days": -4, "points": 5, "mixed": True},
            {"name": "Project Checkpoint", "due_offset_days": 10, "points": 20, "mixed": False},
            {"name": "Final Report", "due_offset_days": 30, "points": 30, "mixed": False},
        ],
        "quiz": {"title": "Sprint Planning Basics", "points": 10, "due_offset_days": 7},
    },
    {
        "sis_id": "LOCAL-SAMPLE-COSC-310",
        "name": "Software Engineering COSC 310",
        "code": "COSC 310",
        "teacher": PRIMARY_TEACHER,
        "roster_prefix": "cosc310student",
        "roster_label": "COSC 310",
        "syllabus": "<p>Software engineering practices: design docs, code review, testing.</p>",
        "assignments": [
            {"name": "Design Document", "due_offset_days": -5, "points": 15, "mixed": True},
            {"name": "Sprint Retrospective", "due_offset_days": 14, "points": 10, "mixed": False},
        ],
        "quiz": {"title": "Agile Terminology Quiz", "points": 8, "due_offset_days": 6},
    },
    {
        "sis_id": "LOCAL-SAMPLE-ENGL-112",
        "name": "Introduction to Composition ENGL 112",
        "code": "ENGL 112",
        "teacher": PRIMARY_TEACHER,
        "roster_prefix": "engl112student",
        "roster_label": "ENGL 112",
        "syllabus": "<p>Composition fundamentals: rhetorical analysis, argumentative writing.</p>",
        "assignments": [
            {"name": "Rhetorical Analysis Essay", "due_offset_days": -6, "points": 20, "mixed": True},
            {"name": "Argumentative Essay", "due_offset_days": 15, "points": 25, "mixed": False},
        ],
        "quiz": {"title": "Citation Styles Quiz", "points": 6, "due_offset_days": 5},
    },
    {
        "sis_id": "LOCAL-SAMPLE-COSC-221",
        "name": "Data Structures COSC 221",
        "code": "COSC 221",
        "teacher": {"login": "teacher2@example.test", "name": "Dana Rivera"},
        "roster_prefix": "cosc221student",
        "roster_label": "COSC 221",
        "syllabus": "<p>Core data structures and algorithmic complexity.</p>",
        "assignments": [
            {"name": "Linked Lists Lab", "due_offset_days": -3, "points": 10, "mixed": True},
            {"name": "Big-O Analysis", "due_offset_days": 11, "points": 15, "mixed": False},
        ],
        "quiz": {"title": "Big-O Notation Quiz", "points": 10, "due_offset_days": 4},
    },
    {
        "sis_id": "LOCAL-SAMPLE-COSC-304",
        "name": "Intro to Databases COSC 304",
        "code": "COSC 304",
        "teacher": {"login": "teacher3@example.test", "name": "Drew Okafor"},
        "roster_prefix": "cosc304student",
        "roster_label": "COSC 304",
        "syllabus": "<p>Relational database design, normalization, SQL.</p>",
        "assignments": [
            {"name": "ER Diagram", "due_offset_days": -4, "points": 10, "mixed": True},
            {"name": "SQL Query Set", "due_offset_days": 13, "points": 20, "mixed": False},
        ],
        "quiz": {"title": "Normalization Quiz", "points": 10, "due_offset_days": 8},
    },
]

FEEDBACK_SNIPPETS = [
    "Solid start — tighten up the conclusion next time.",
    "Nice work, this meets the requirements well.",
    "Good effort, but a few sections need more detail.",
    "Well organized. See inline notes for minor fixes.",
]

MESSAGE_SUBJECTS = [
    ("Question about the assignment", "I had a question about the due date, could you clarify?"),
    ("Extension request", "Would it be possible to get a short extension on this one?"),
    ("Great work reminder", "Just checking in on your progress with the current assignment."),
]


class IdCounter:
    def __init__(self, start: int = 1) -> None:
        self._next = start

    def next(self) -> int:
        value = self._next
        self._next += 1
        return value


user_ids = IdCounter(1000)
course_ids = IdCounter(2000)
enrollment_ids = IdCounter(3000)
assignment_ids = IdCounter(4000)
submission_ids = IdCounter(5000)
comment_ids = IdCounter(6000)
quiz_ids = IdCounter(7000)
question_ids = IdCounter(8000)
discussion_ids = IdCounter(9000)
page_ids = IdCounter(10000)
module_ids = IdCounter(11000)
conversation_ids = IdCounter(12000)
message_ids = IdCounter(13000)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_user(name: str, login: str) -> dict:
    first, *rest = name.split()
    last = " ".join(rest)
    return {
        "id": user_ids.next(),
        "name": name,
        "short_name": first,
        "sortable_name": f"{last}, {first}" if last else first,
        "login_id": login,
        "email": login,
    }


def build_assignment(course_id: int, spec: dict) -> dict:
    due_at = NOW + timedelta(days=spec["due_offset_days"])
    submission_types = MIXED_TYPES if spec["mixed"] else ["online_text_entry"]
    assignment = {
        "id": assignment_ids.next(),
        "course_id": course_id,
        "name": spec["name"],
        "description": f"<p>{spec['name']} — see course syllabus for full instructions.</p>",
        "due_at": iso(due_at),
        "points_possible": spec["points"],
        "submission_types": submission_types,
        "workflow_state": "published",
        "html_url": f"https://canvas.local/courses/{course_id}/assignments/{spec['name'].lower().replace(' ', '-')}",
    }
    if spec["mixed"]:
        assignment["rubric"] = [
            {
                "id": f"{assignment['id']}_criterion_1",
                "description": "Content quality",
                "points": round(spec["points"] * 0.6, 1),
                "ratings": [
                    {"description": "Excellent", "points": round(spec["points"] * 0.6, 1)},
                    {"description": "Satisfactory", "points": round(spec["points"] * 0.35, 1)},
                    {"description": "Needs work", "points": 0},
                ],
            },
            {
                "id": f"{assignment['id']}_criterion_2",
                "description": "Clarity and organization",
                "points": round(spec["points"] * 0.4, 1),
                "ratings": [
                    {"description": "Excellent", "points": round(spec["points"] * 0.4, 1)},
                    {"description": "Satisfactory", "points": round(spec["points"] * 0.2, 1)},
                    {"description": "Needs work", "points": 0},
                ],
            },
        ]
    return assignment, due_at


def seed_mixed_submissions(assignment: dict, due_at: datetime, students: list[dict], teacher: dict, rng: random.Random) -> list[dict]:
    missing_count = round(len(students) * MISSING_RATE)
    submitters = students[missing_count:]
    late_mod_threshold = round(LATE_RATE * 10)
    ungraded_mod_divisor = round(1 / UNGRADED_RATE)

    submissions = []
    for i, student in enumerate(submitters):
        late = (i % 10) < late_mod_threshold
        jitter_hours = rng.randint(1, 48) if late else rng.randint(6, 72)
        submitted_at = due_at + timedelta(hours=jitter_hours) if late else due_at - timedelta(hours=jitter_hours)
        submission_type = MIXED_TYPES[i % 3]

        submission = {
            "id": submission_ids.next(),
            "assignment_id": assignment["id"],
            "user_id": student["id"],
            "submitted_at": iso(submitted_at),
            "submission_type": submission_type,
            "late": late,
            "missing": False,
            "workflow_state": "submitted",
        }
        if submission_type == "online_text_entry":
            submission["body"] = f"Sample submission from {student['name']}."
        elif submission_type == "online_upload":
            submission["attachments"] = [
                {"filename": f"submission-{i}.txt", "display_name": f"{student['short_name'].lower()}-submission.txt", "content_type": "text/plain"}
            ]
        else:
            submission["url"] = f"https://example.com/{student['short_name'].lower()}-project"

        ungraded = (i % ungraded_mod_divisor) == 0
        if not ungraded:
            score = max(assignment["points_possible"] - (i % 3), 0)
            submission["score"] = score
            submission["grade"] = str(score)
            submission["workflow_state"] = "graded"
            submission["grader_id"] = teacher["id"]
            submission["graded_at"] = iso(submitted_at + timedelta(hours=6))
            if i % 5 == 0:
                submission["submission_comments"] = [
                    {
                        "id": comment_ids.next(),
                        "author_id": teacher["id"],
                        "author_name": teacher["name"],
                        "comment": rng.choice(FEEDBACK_SNIPPETS),
                        "created_at": iso(submitted_at + timedelta(hours=6)),
                    }
                ]
        submissions.append(submission)

    for student in students[:missing_count]:
        submissions.append(
            {
                "id": submission_ids.next(),
                "assignment_id": assignment["id"],
                "user_id": student["id"],
                "submitted_at": None,
                "workflow_state": "unsubmitted",
                "late": False,
                "missing": True,
            }
        )

    return submissions


def build_quiz(course_id: int, spec: dict) -> dict:
    due_at = NOW + timedelta(days=spec["due_offset_days"])
    questions = [
        {
            "id": question_ids.next(),
            "question_name": "Question 1",
            "question_type": "multiple_choice_question",
            "question_text": f"Which best describes a core concept from {spec['title'].replace(' Quiz', '')}?",
            "points_possible": spec["points"] / 2,
            "answers": [
                {"text": "Correct answer", "weight": 100},
                {"text": "Distractor A", "weight": 0},
                {"text": "Distractor B", "weight": 0},
            ],
        },
        {
            "id": question_ids.next(),
            "question_name": "Question 2",
            "question_type": "true_false_question",
            "question_text": "True or false: this concept applies in the context covered this week.",
            "points_possible": spec["points"] / 2,
            "answers": [{"text": "True", "weight": 100}, {"text": "False", "weight": 0}],
        },
    ]
    return {
        "id": quiz_ids.next(),
        "course_id": course_id,
        "title": spec["title"],
        "quiz_type": "assignment",
        "points_possible": spec["points"],
        "due_at": iso(due_at),
        "question_count": len(questions),
        "published": True,
        "questions": questions,
    }


def build_discussion(course_id: int, teacher: dict) -> dict:
    return {
        "id": discussion_ids.next(),
        "course_id": course_id,
        "title": "Welcome Discussion",
        "message": "Introduce yourself and describe what you want to validate in this local Canvas instance.",
        "posted_by": {"id": teacher["id"], "name": teacher["name"]},
        "workflow_state": "active",
        "is_announcement": False,
    }


def build_announcement(course_id: int, teacher: dict, course_name: str) -> dict:
    return {
        "id": discussion_ids.next(),
        "course_id": course_id,
        "title": f"Welcome to {course_name}",
        "message": "Course materials are posted. Please review the syllabus and reach out with questions.",
        "posted_by": {"id": teacher["id"], "name": teacher["name"]},
        "workflow_state": "active",
        "is_announcement": True,
        "posted_at": iso(NOW - timedelta(days=7)),
    }


def build_page(course_id: int) -> dict:
    return {
        "page_id": page_ids.next(),
        "course_id": course_id,
        "title": "Local Canvas Test Page",
        "body": "<p>This page exists so everyone has shared content to inspect after seeding.</p>",
        "workflow_state": "active",
    }


def build_module(course_id: int, assignment_ids_list: list[int], discussion_id: int, page_id: int) -> dict:
    items = [{"type": "Assignment", "content_id": aid} for aid in assignment_ids_list]
    items.append({"type": "Discussion", "content_id": discussion_id})
    items.append({"type": "Page", "content_id": page_id})
    return {
        "id": module_ids.next(),
        "course_id": course_id,
        "name": "Sample Module",
        "workflow_state": "active",
        "items": items,
    }


def build_conversations(course_id: int, teacher: dict, students: list[dict], rng: random.Random) -> list[dict]:
    conversations = []
    sample_students = rng.sample(students, k=min(6, len(students)))
    for student, (subject, opener) in zip(sample_students, MESSAGE_SUBJECTS * 2):
        started_at = NOW - timedelta(days=rng.randint(1, 10))
        conversations.append(
            {
                "id": conversation_ids.next(),
                "course_id": course_id,
                "subject": subject,
                "participants": [
                    {"id": student["id"], "name": student["name"]},
                    {"id": teacher["id"], "name": teacher["name"]},
                ],
                "messages": [
                    {
                        "id": message_ids.next(),
                        "author_id": student["id"],
                        "body": opener,
                        "created_at": iso(started_at),
                    }
                ],
            }
        )
    return conversations


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260722)

    teachers: dict[str, dict] = {}
    all_courses = []

    for spec in COURSES:
        teacher_login = spec["teacher"]["login"]
        if teacher_login not in teachers:
            teachers[teacher_login] = build_user(spec["teacher"]["name"], teacher_login)
        teacher = teachers[teacher_login]

        course_id = course_ids.next()
        course = {
            "id": course_id,
            "sis_course_id": spec["sis_id"],
            "name": spec["name"],
            "course_code": spec["code"],
            "workflow_state": "available",
            "start_at": iso(NOW - timedelta(weeks=1)),
            "end_at": iso(NOW + timedelta(weeks=12)),
            "syllabus_body": spec["syllabus"],
        }
        all_courses.append(course)

        students = [
            build_user(f"{spec['roster_label']} Student {i}", f"{spec['roster_prefix']}{i}@example.test")
            for i in range(1, STUDENT_COUNT + 1)
        ]

        enrollments = [
            {
                "id": enrollment_ids.next(),
                "course_id": course_id,
                "type": "TeacherEnrollment",
                "role": "TeacherEnrollment",
                "enrollment_state": "active",
                "user": teacher,
            }
        ]
        enrollments += [
            {
                "id": enrollment_ids.next(),
                "course_id": course_id,
                "type": "StudentEnrollment",
                "role": "StudentEnrollment",
                "enrollment_state": "active",
                "user": student,
            }
            for student in students
        ]
        (OUTPUT_DIR / f"users_{spec['sis_id']}.json").write_text(json.dumps(enrollments, indent=2))

        assignments = []
        due_dates = {}
        for a_spec in spec["assignments"]:
            assignment, due_at = build_assignment(course_id, a_spec)
            assignments.append(assignment)
            due_dates[assignment["id"]] = due_at
        (OUTPUT_DIR / f"assignments_{spec['sis_id']}.json").write_text(json.dumps(assignments, indent=2))

        primary_assignment = assignments[0]
        submissions = seed_mixed_submissions(
            primary_assignment, due_dates[primary_assignment["id"]], students, teacher, rng
        )
        (OUTPUT_DIR / f"submissions_{spec['sis_id']}.json").write_text(json.dumps(submissions, indent=2))

        quiz = build_quiz(course_id, spec["quiz"])
        (OUTPUT_DIR / f"quizzes_{spec['sis_id']}.json").write_text(json.dumps([quiz], indent=2))

        discussion = build_discussion(course_id, teacher)
        announcement = build_announcement(course_id, teacher, spec["name"])
        (OUTPUT_DIR / f"discussion_topics_{spec['sis_id']}.json").write_text(
            json.dumps([discussion, announcement], indent=2)
        )

        page = build_page(course_id)
        (OUTPUT_DIR / f"pages_{spec['sis_id']}.json").write_text(json.dumps([page], indent=2))

        module = build_module(course_id, [a["id"] for a in assignments], discussion["id"], page["page_id"])
        (OUTPUT_DIR / f"modules_{spec['sis_id']}.json").write_text(json.dumps([module], indent=2))

        conversations = build_conversations(course_id, teacher, students, rng)
        (OUTPUT_DIR / f"conversations_{spec['sis_id']}.json").write_text(json.dumps(conversations, indent=2))

        print(f"generated {spec['name']}: {len(students)} students, {len(assignments)} assignments, "
              f"{len(submissions)} submissions, 1 quiz, {len(conversations)} conversations")

    (OUTPUT_DIR / "courses.json").write_text(json.dumps(all_courses, indent=2))
    print(f"\nwrote {len(list(OUTPUT_DIR.glob('*.json')))} files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
