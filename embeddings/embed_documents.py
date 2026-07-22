"""Chunk and embed cleaned Canvas content into a FAISS index.

Input: datasets/processed/*.json (already redacted by
preprocessing/clean_canvas_data.py). Each document's text-bearing fields
(assignment descriptions, syllabus body, announcements, etc.) are chunked
and embedded with BGE-small, per configs/retrieval.yaml.

Output: the FAISS index + metadata sidecar at the paths in
configs/retrieval.yaml, consumed by embeddings/retrieve.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from embeddings.vector_store import Chunk, VectorStore

logger = logging.getLogger(__name__)

# Canvas JSON fields worth indexing as retrievable prose.
TEXT_FIELDS = {"description", "message", "body", "syllabus_body", "name", "title"}


def extract_text_fields(node: object, source: str, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in TEXT_FIELDS and isinstance(value, str) and value.strip():
                out.append(value.strip())
            else:
                extract_text_fields(value, source, out)
    elif isinstance(node, list):
        for item in node:
            extract_text_fields(item, source, out)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(input_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text())
        raw_fields: list[str] = []
        extract_text_fields(data, path.name, raw_fields)
        for field_text in raw_fields:
            for i, piece in enumerate(chunk_text(field_text, chunk_size, chunk_overlap)):
                chunks.append(Chunk(text=piece, source=path.name, chunk_index=i))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/processed"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = yaml.safe_load(args.config.read_text())

    chunks = build_chunks(
        args.input_dir,
        config["chunking"]["chunk_size"],
        config["chunking"]["chunk_overlap"],
    )
    if not chunks:
        logger.warning("no text chunks found under %s", args.input_dir)
        return

    logger.info("embedding %d chunks with %s", len(chunks), config["embedding_model"])
    model = SentenceTransformer(config["embedding_model"], device=config["device"])
    vectors = model.encode([c.text for c in chunks], show_progress_bar=True, convert_to_numpy=True)

    store = VectorStore(dim=config["embedding_dim"])
    store.add(np.asarray(vectors), chunks)
    store.save(Path(config["index"]["path"]), Path(config["index"]["metadata_path"]))

    logger.info("saved index -> %s", config["index"]["path"])


if __name__ == "__main__":
    main()
