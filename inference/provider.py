"""Reference client for the fine-tuned model, served locally via llama.cpp.

This is *not* wired into the main app yet — it's a portable reference
implementation, written so it can be dropped into `app/mcp-server`'s
provider layer once the model is trained, exported, and running in that
project's Docker Compose stack. Until then it's used by this repo's own
evaluation scripts (evaluation/accuracy.py, evaluation/hallucination_test.py,
evaluation/benchmark.py) to exercise the served model exactly the way
production would.

llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint, so
this talks to it the same way app/mcp-server's src/services/llm.py talks to
OpenAI. Same system prompt (prompts/system.txt, copied verbatim from that
repo's PromptBuilder), same user-turn shape (preprocessing/user_turn.py,
ported line-for-line from PromptBuilder._build_user_turn) — training and
serving both go through that one shared formatter so they can't drift apart.

Note: production gets JSON-schema-constrained decoding for free via OpenAI's
`.beta.chat.completions.parse(response_format=ParsedIntent)`. This client
does not enforce that — llama-server's OpenAI-compatible endpoint supports an
equivalent `response_format: {"type": "json_schema", ...}`, but wiring that
up is left to whoever adapts this into app/mcp-server's actual local-model
provider path, not attempted here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.user_turn import build_user_turn


@dataclass
class ProviderResponse:
    content: str
    model: str
    finish_reason: str


def today_in_course_timezone(course_timezone: str | None) -> tuple[str, str]:
    """Mirrors PromptBuilder._today_in_course_timezone exactly — falls back
    to UTC when the course has no timezone on record, or the name isn't in
    Python's tz database, rather than raising."""
    if course_timezone:
        try:
            return datetime.now(ZoneInfo(course_timezone)).strftime("%Y-%m-%d"), course_timezone
        except ZoneInfoNotFoundError:
            pass
    return datetime.now(UTC).strftime("%Y-%m-%d"), "UTC"


class LocalQwenProvider:
    """Calls a locally-served (llama.cpp) fine-tuned Qwen3-8B."""

    def __init__(
        self,
        base_url: str,
        chat_completions_path: str,
        system_prompt: str,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_completions_path = chat_completions_path
        self._system_prompt = system_prompt
        self._client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_config(cls, serving_config_path: Path = Path("configs/serving.yaml")) -> "LocalQwenProvider":
        serving_config = yaml.safe_load(serving_config_path.read_text(encoding="utf-8"))["server"]
        system_prompt = Path("prompts/system.txt").read_text(encoding="utf-8").strip()
        base_url = f"http://{serving_config['host']}:{serving_config['port']}"
        return cls(base_url, serving_config["chat_completions_path"], system_prompt)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_raw(
        self, user_turn: str, conversation_history: list[dict] | None = None
    ) -> ProviderResponse:
        """Send an already-built user turn as-is — used when the caller has
        exact production-shaped text on hand already (e.g. accuracy.py
        replaying a ChatML test example) rather than raw fields to format."""
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(conversation_history or [])
        messages.append({"role": "user", "content": user_turn})

        response = await self._client.post(
            f"{self._base_url}{self._chat_completions_path}",
            json={"model": "qwen3-8b-canvas-ai", "messages": messages, "temperature": 0.2},
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]

        return ProviderResponse(
            content=choice["message"]["content"],
            model=payload.get("model", "unknown"),
            finish_reason=choice.get("finish_reason", "unknown"),
        )

    async def generate(
        self,
        instructor_request: str,
        course_timezone: str | None = None,
        course_name: str | None = None,
        course_code: str | None = None,
        available_assignments: list[str] | None = None,
        recent_messages: list[str] | None = None,
        difficulty: str | None = None,
        quiz_requirements: dict | None = None,
        conversation_history: list[dict] | None = None,
    ) -> ProviderResponse:
        """Build the user turn from structured Canvas context fields (the
        shape a real caller has on hand) using today's actual date, then
        call generate_raw()."""
        today_date, tz_label = today_in_course_timezone(course_timezone)
        user_turn = build_user_turn(
            instructor_request=instructor_request,
            today_date=today_date,
            course_timezone=tz_label,
            course_name=course_name,
            course_code=course_code,
            available_assignments=available_assignments,
            recent_messages=recent_messages,
            difficulty=difficulty,
            quiz_requirements=quiz_requirements,
        )
        return await self.generate_raw(user_turn, conversation_history)
