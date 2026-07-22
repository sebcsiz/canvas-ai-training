"""Query the FAISS index built by embed_documents.py.

Used both offline (preprocessing/convert_to_chatml.py, to train the model on
retrieval-augmented inputs) and online (inference/provider.py, to ground
production responses in the instructor's actual Canvas content).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

from embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self, store: VectorStore, model: SentenceTransformer, config: dict) -> None:
        self.store = store
        self.model = model
        self.query_instruction = config["query"]["query_instruction"]
        self.default_top_k = config["query"]["top_k"]
        self.score_threshold = config["query"]["score_threshold"]

    @classmethod
    def from_config(cls, config_path: Path = Path("configs/retrieval.yaml"), top_k: int | None = None) -> "Retriever":
        config = yaml.safe_load(config_path.read_text())
        store = VectorStore.load(Path(config["index"]["path"]), Path(config["index"]["metadata_path"]))
        model = SentenceTransformer(config["embedding_model"], device=config["device"])
        if top_k is not None:
            config["query"]["top_k"] = top_k
        return cls(store, model, config)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_vector = self.model.encode(self.query_instruction + query, convert_to_numpy=True)
        results = self.store.search(query_vector, top_k or self.default_top_k)
        return [
            RetrievedChunk(text=chunk.text, source=chunk.source, score=score)
            for chunk, score in results
            if score >= self.score_threshold
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ad-hoc query against the retrieval index.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    retriever = Retriever.from_config(top_k=args.top_k)
    for hit in retriever.retrieve(args.query, args.top_k):
        print(f"[{hit.score:.3f}] {hit.source}: {hit.text}")


if __name__ == "__main__":
    main()
