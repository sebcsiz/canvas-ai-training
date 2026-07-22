# Canvas AI — Local Model Design Doc

Companion training repo for the **Capstone Team 3 — Canvas AI** app. This repo is
where the local model is fine-tuned and packaged; the app itself (frontend /
backend / mcp-server / db / redis) lives in the main project repo and is not
checked out here.

## Goal

Produce a small, locally-hostable model that can replace (or sit alongside) a
hosted LLM provider as the `mcp-server`'s reasoning engine for the Canvas AI
assistant, while staying consistent with that app's safety architecture:
Standard Command Pipeline (interpret → assess risk → preview → confirm →
execute → record), the 7 in-scope MVP workflows, and PII redaction before
anything reaches the model.

Approach: **QLoRA fine-tune of Qwen3-8B**, teacher/student distillation for
training data, **BGE-small retrieval (RAG)** grounded in Canvas course
content, evaluated for structured-action accuracy and hallucination rate,
then exported to **GGUF for llama.cpp** so the resulting artifact is just a
weights file + index that can be dropped into the main app's Docker Compose
stack (FR12.4 — local deployment).

This repo does the *training*. The main app does the *serving* (a llama.cpp
container calling the exported GGUF weights). `inference/provider.py` here is
a reference implementation of that provider, written to be portable into
`app/mcp-server` once the model is ready.

## Why these choices

- **Qwen3-8B**: small enough to QLoRA-tune and run quantized on a single
  consumer/lab GPU, strong tool-calling support out of the box.
- **QLoRA (PEFT + TRL + bitsandbytes)**: standard HF stack, well documented,
  keeps the whole pipeline debuggable without a heavier framework
  (Axolotl/Unsloth) in the loop.
- **Teacher/student distillation**: a strong hosted model (the "teacher")
  generates gold (instructor request → Canvas action) examples for the 7 MVP
  workflows; Qwen3-8B (the "student") is fine-tuned to reproduce that
  behavior directly, without needing the teacher at inference time.
- **BGE-small + FAISS**: `BAAI/bge-small-en-v1.5` is cheap to embed with and
  runs on CPU. FAISS is a single index file with no server to run — it ships
  next to the GGUF weights as part of the same portable artifact, matching
  the "pull weights, run in a container" deployment plan.
- **GGUF / llama.cpp**: lightweight, well-suited to local Docker Compose
  deployment, and llama-server exposes an OpenAI-compatible
  `/v1/chat/completions` endpoint, so `inference/provider.py` can talk to it
  the same way the app already talks to a hosted provider.

## Pipeline

```
                 ┌────────────────────┐
                 │ Canvas course data  │  (pulled via main app's backend,
                 │ (raw exports)       │   read-only, instructor-consented)
                 └─────────┬───────────┘
                            │
                 preprocessing/clean_canvas_data.py   (redact student PII)
                            │
                            ▼
                 datasets/processed/  ───────────────┐
                            │                          │
        preprocessing/generate_synthetic_data.py       │
        (teacher.txt + hosted LLM → gold examples       embeddings/embed_documents.py
         for the 7 MVP workflows)                        (BGE-small → FAISS index)
                            │                          │
        preprocessing/convert_to_chatml.py              ▼
        (student.txt system prompt +           embeddings/vector_store.py
         Qwen3 ChatML tool-call format)          embeddings/retrieve.py
                            │
        preprocessing/split_dataset.py
        (train / validation / test)
                            │
                            ▼
        training/train.py  (QLoRA SFT via TRL SFTTrainer)
                            │
        training/resume.py  (checkpoint resume)
                            │
        training/merge_lora.py  (merge adapter → full model)
                            │
        training/export_model.py  (→ GGUF, quantized)
                            │
                            ▼
        evaluation/accuracy.py           (structured action accuracy vs. test set)
        evaluation/hallucination_test.py (RAG-grounding check)
        evaluation/benchmark.py          (latency / throughput on llama.cpp)
                            │
                            ▼
        models/gguf/*.gguf  ──── pulled into the main app's Docker Compose,
                                  served via llama.cpp, called through
                                  inference/provider.py's provider interface.
```

## Directory layout

| Path | Owns |
|---|---|
| `configs/` | YAML configs: QLoRA hyperparams, retrieval params, dataset/workflow list, serving/quantization params. Scripts read these; no hardcoded hyperparameters in code. |
| `datasets/raw/` → `processed/` → `train/` `validation/` `test/` | Pipeline stages. `raw` and `processed` may contain redacted-but-real Canvas content; `train/validation/test` are ChatML JSONL, gitignored. |
| `preprocessing/` | Redaction, synthetic data generation, ChatML conversion, splitting. |
| `embeddings/` | BGE-small embedding + FAISS index build/query. |
| `prompts/` | `system.txt` (shared identity + safety rules, mirrors the main app's invariants), `teacher.txt` (data-generation role), `student.txt` (deployed-model role — what actually gets trained in and served). |
| `training/` | QLoRA SFT, resume, LoRA merge, GGUF export. |
| `evaluation/` | Structured accuracy, hallucination/grounding, latency benchmark. |
| `inference/` | `provider.py` — reference client for the served GGUF model, RAG-augmented, shaped to drop into `app/mcp-server`. |
| `models/`, `outputs/` | Generated artifacts (gitignored). |
| `scripts/` | Bash orchestration wrapping the above, one stage per script. |

## Safety alignment with the main app

The fine-tuned model does **not** get more trust than a hosted provider would.
Everything the main `CLAUDE.md` requires still applies once this model is
plugged in:

- The model only ever *proposes* actions; `backend` still validates and the
  user still confirms via the Standard Command Pipeline (FR7, FR9.3–9.4).
  Training data and `student.txt` teach the model to emit previews, not to
  execute anything itself.
- `preprocessing/clean_canvas_data.py` redacts student-identifiable
  information before any Canvas content reaches training data or the
  retrieval index (mirrors FR10.4).
- Training examples are scoped to exactly the 7 MVP workflows listed in the
  main `CLAUDE.md`; out-of-scope requests are trained to be politely refused
  (mirrors FR3.6/FR3.7).
- No Canvas tokens or credentials are ever part of training data, prompts, or
  the retrieval index.

## Open questions

- Exact schema for tool-call turns in ChatML — needs to match whatever tool
  schema `app/mcp-server/src/tools/` settles on. `preprocessing/convert_to_chatml.py`
  isolates this in one place so it's a small change if the schema shifts.
  See `docs/tool_schema.md` (TODO — write once the mcp-server tool schemas
  are stable) for the reconciled schema.
- Where the FAISS index + GGUF weights physically get hosted/pulled from for
  the main app's Docker Compose (git-lfs, object storage, etc.) — not decided
  yet, out of scope for this repo until export is actually run.
