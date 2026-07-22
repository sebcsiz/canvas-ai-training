#!/usr/bin/env bash
# Builds the BGE-small / FAISS retrieval index from datasets/processed/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python embeddings/embed_documents.py --config configs/retrieval.yaml
