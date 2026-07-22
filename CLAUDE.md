# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Training pipeline for a local model that will replace/supplement the hosted
LLM provider in the **Capstone Team 3 — Canvas AI** app's `mcp-server`
(that app is a separate repo, not checked out here). Approach: **QLoRA
fine-tune of Qwen3-8B**, teacher/student distillation for training data,
**BGE-small + FAISS retrieval** grounded in Canvas course content, exported
to **GGUF for llama.cpp** so the artifact is just weights + an index file
that can be dropped into that app's Docker Compose stack.

**Read `docs/design.md` before making non-trivial changes** — it has the
full pipeline diagram, the rationale for every architecture choice, and how
this repo's output must stay aligned with the main app's safety invariants
(Standard Command Pipeline, PII redaction, MVP workflow scope). This file
covers commands and cross-file structure; `docs/design.md` covers the *why*.

As of now, nothing has actually been trained — the pipeline is scaffolded
and untested end-to-end (no GPU run, no real Canvas export, no generated
data yet).

## Setup

```bash
pip install -r requirements.txt
```

Environment variables needed depending on stage:
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — for `preprocessing/generate_synthetic_data.py` (whichever provider `configs/data.yaml`'s `synthetic_generation.teacher_provider` names).
- `LLAMA_CPP_DIR` — path to a local, built checkout of [llama.cpp](https://github.com/ggml-org/llama.cpp) (needs `convert_hf_to_gguf.py` and a built `llama-quantize` / `llama-server` under `build/bin/`). Needed for `training/export_model.py` and `scripts/evaluate.sh`.

## Commands

The pipeline runs in this order; each stage is a `scripts/*.sh` wrapper
around the underlying Python entry point(s), and every stage reads its
parameters from `configs/*.yaml` rather than flags:

```bash
./scripts/prepare_data.sh   # clean_canvas_data.py -> generate_synthetic_data.py -> convert_to_chatml.py -> split_dataset.py
./scripts/build_index.sh    # embed_documents.py: builds the FAISS retrieval index
./scripts/train.sh          # train.py (add --resume to continue from the latest checkpoint via resume.py)
./scripts/export.sh         # merge_lora.py -> export_model.py: LoRA merge + GGUF export
./scripts/evaluate.sh       # starts llama-server on the exported GGUF, runs the eval suite, tears it down
```

Individual stages can also be run directly, e.g. `python training/train.py --config configs/qlora.yaml`
or `python embeddings/retrieve.py "some query"` for an ad-hoc retrieval check.

Evaluation scripts (`evaluation/accuracy.py`, `evaluation/hallucination_test.py`,
`evaluation/benchmark.py`) each assume a llama.cpp server is already running
per `configs/serving.yaml`; `scripts/evaluate.sh` handles that lifecycle,
but they can be pointed at a manually-started server too.

Lint: `ruff check .` (no `pyproject.toml`/config file committed yet, so
default ruff rules apply). There is no test suite in this repo yet.

## Architecture

**Everything is config-driven.** No script hardcodes hyperparameters, paths,
or model names — they all come from `configs/{qlora,retrieval,data,serving}.yaml`.
When changing a training/retrieval/serving parameter, edit the YAML, not the
script.

**Pipeline stages are directories, and data flows through `datasets/` in
stages matching its subfolder names:** `raw/` (synthetic examples from the
teacher model, and raw Canvas exports) → `processed/` (PII-redacted, and
ChatML-converted under `processed/chatml/`) → `train/` / `validation/` /
`test/` (final stratified splits, gitignored). `preprocessing/*.py` scripts
correspond 1:1 to these transitions, in filename order:
`clean_canvas_data.py` → `generate_synthetic_data.py` → `convert_to_chatml.py`
→ `split_dataset.py`.

**Teacher/student prompt split (`prompts/`)** is the core of the distillation
approach: `system.txt` holds the shared identity + safety rules (mirrors the
main app's `CLAUDE.md` invariants: Standard Command Pipeline, MVP workflow
scope, PII minimization). `teacher.txt` and `student.txt` both template in
`{{SYSTEM_PROMPT}}` and add a role-specific addendum — `teacher.txt` instructs
a strong hosted model on how to *generate* gold training examples
(`preprocessing/generate_synthetic_data.py`), `student.txt` is what actually
gets baked into every training example (`preprocessing/convert_to_chatml.py`)
and used at real inference time (`inference/provider.py`). If the safety
rules change, edit `system.txt` — `teacher.txt`/`student.txt` inherit it.

**Retrieval (`embeddings/`) has three layers with a fixed dependency
direction:** `vector_store.py` (FAISS index + metadata, no ML) is used by
`embed_documents.py` (offline: chunk + embed Canvas content into the index)
and `retrieve.py` (query-time: embed a query, search the index). Both
`preprocessing/convert_to_chatml.py` (optional `--with-retrieval`, to train
on the same input shape production will see) and `inference/provider.py`
(always, at serve time) import `retrieve.py` — never `vector_store.py`
directly.

**`training/` scripts chain into each other's output, not just
config-driven but file-path-driven:** `train.py`/`resume.py` write a LoRA
adapter to `configs/qlora.yaml`'s `training.output_dir`; `merge_lora.py`
reads that adapter (`configs/serving.yaml`'s `merge.adapter_dir` — keep
these two paths in sync if you change one) and writes a merged model;
`export_model.py` shells out to the `LLAMA_CPP_DIR` checkout's conversion
+ quantization tooling to turn that merged model into the GGUF file that's
the actual deployable artifact.

**`inference/provider.py` is a portable reference implementation, not
wired into anything yet.** It's written to the shape the main app's
`mcp-server` provider layer would expect (async, httpx, calls a llama.cpp
server's OpenAI-compatible `/v1/chat/completions` endpoint) specifically so
it can be copied over once the model is trained. Right now its only callers
are this repo's own `evaluation/*.py` scripts, which use it to exercise the
served model the same way production would.

**Known placeholder needing reconciliation:** the structured tool-call
format assumed by `preprocessing/convert_to_chatml.py`'s training examples
and `evaluation/accuracy.py`'s scoring (a fenced ` ```json {"action":
..., "parameters": ...} ``` ` block) is provisional — it must be reconciled
with whatever tool schema the main app's `app/mcp-server/src/tools/`
actually settles on before real training data is generated. See the open
questions in `docs/design.md`.
