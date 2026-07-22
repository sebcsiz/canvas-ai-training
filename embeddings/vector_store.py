"""Thin FAISS wrapper storing BGE-small chunk vectors + their source text.

A single self-contained artifact (index file + metadata sidecar) that ships
alongside the exported GGUF weights — no vector DB server to run, per the
design in docs/design.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


class VectorStore:
    """Cosine-similarity search over L2-normalized vectors via faiss.IndexFlatIP."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        normalized = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        self.index.add(normalized.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        normalized = query_vector / np.clip(np.linalg.norm(query_vector), 1e-12, None)
        scores, indices = self.index.search(normalized.reshape(1, -1).astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with metadata_path.open("w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.__dict__) + "\n")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "VectorStore":
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"index not found at {index_path} / {metadata_path}")
        index = faiss.read_index(str(index_path))
        store = cls(dim=index.d)
        store.index = index
        with metadata_path.open() as f:
            store.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return store
