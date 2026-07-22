"""Reference client for the fine-tuned model, served locally via llama.cpp.

This is *not* wired into the main app yet — it's a portable reference
implementation, written so it can be dropped into `app/mcp-server`'s
provider layer once the model is trained, exported, and running in that
project's Docker Compose stack. Until then it's used by this repo's own
evaluation scripts (evaluation/accuracy.py, evaluation/hallucination_test.py,
evaluation/benchmark.py) to exercise the served model exactly the way
production would.

llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint, so
this talks to it the same way the main app talks to a hosted provider —
async-only, via httpx, per that repo's conventions. It never calls Canvas;
it only proposes actions, matching the Standard Command Pipeline described
in prompts/system.txt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from embeddings.retrieve import Retriever


@dataclass
class ProviderResponse:
    content: str
    model: str
    finish_reason: str
    retrieved_sources: list[str]


class LocalQwenProvider:
    """Calls a locally-served (llama.cpp) fine-tuned Qwen3-8B, RAG-augmented."""

    def __init__(
        self,
        base_url: str,
        chat_completions_path: str,
        student_prompt: str,
        retriever: Retriever | None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_completions_path = chat_completions_path
        self._student_prompt = student_prompt
        self._retriever = retriever
        self._client = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_config(
        cls,
        serving_config_path: Path = Path("configs/serving.yaml"),
        retrieval_config_path: Path = Path("configs/retrieval.yaml"),
        use_retrieval: bool = True,
    ) -> "LocalQwenProvider":
        serving_config = yaml.safe_load(serving_config_path.read_text())["server"]
        system_prompt = (Path("prompts/system.txt")).read_text().strip()
        student_prompt = (
            Path("prompts/student.txt").read_text().strip().replace("{{SYSTEM_PROMPT}}", system_prompt)
        )

        retriever = None
        if use_retrieval:
            try:
                retriever = Retriever.from_config(retrieval_config_path)
            except FileNotFoundError:
                retriever = None

        base_url = f"http://{serving_config['host']}:{serving_config['port']}"
        return cls(base_url, serving_config["chat_completions_path"], student_prompt, retriever)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_user_turn(self, canvas_context: str, instructor_request: str) -> str:
        parts = [f"Canvas context:\n{canvas_context}"]
        if self._retriever is not None:
            hits = self._retriever.retrieve(instructor_request)
            if hits:
                parts.append("Retrieved context:\n" + "\n".join(f"- {hit.text}" for hit in hits))
        parts.append(f"Instructor request: {instructor_request}")
        return "\n\n".join(parts)

    async def generate(
        self,
        canvas_context: str,
        instructor_request: str,
        conversation_history: list[dict] | None = None,
    ) -> ProviderResponse:
        messages = [{"role": "system", "content": self._student_prompt}]
        messages.extend(conversation_history or [])
        messages.append(
            {"role": "user", "content": self._build_user_turn(canvas_context, instructor_request)}
        )

        response = await self._client.post(
            f"{self._base_url}{self._chat_completions_path}",
            json={"model": "qwen3-8b-canvas-ai", "messages": messages, "temperature": 0.2},
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]

        sources = []
        if self._retriever is not None:
            sources = [hit.source for hit in self._retriever.retrieve(instructor_request)]

        return ProviderResponse(
            content=choice["message"]["content"],
            model=payload.get("model", "unknown"),
            finish_reason=choice.get("finish_reason", "unknown"),
            retrieved_sources=sources,
        )
