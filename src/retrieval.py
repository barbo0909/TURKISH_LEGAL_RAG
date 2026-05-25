from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

try:
    from .embeddings import encode_queries, load_embedding_model
except ImportError:
    from embeddings import encode_queries, load_embedding_model


TOKEN_RE = re.compile(r"[0-9A-Za-z\u00c7\u011e\u0130\u00d6\u015e\u00dc\u00e7\u011f\u0131\u00f6\u015f\u00fc]+")


def tokenize_tr(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RetrievalEngine:
    def __init__(
        self,
        index_root: str | Path,
        embedding_model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.index_root = Path(index_root)
        manifest_path = self.index_root / "index_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing index manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.metadata = load_jsonl(self.manifest["metadata_path"])
        self.embedding_model_name = embedding_model or self.manifest["embedding_model"]
        self.device = device
        self._embedding_model = None
        self._dense_index = None
        self._bm25 = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = load_embedding_model(self.embedding_model_name, device=self.device)
        return self._embedding_model

    @property
    def dense_index(self):
        if self._dense_index is None:
            self._dense_index = faiss.read_index(self.manifest["dense_index_path"])
        return self._dense_index

    @property
    def bm25(self):
        if self._bm25 is None:
            with Path(self.manifest["bm25_index_path"]).open("rb") as f:
                self._bm25 = pickle.load(f)["bm25"]
        return self._bm25

    def dense_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        vector = encode_queries(self.embedding_model, [query], self.embedding_model_name).astype("float32")
        scores, indices = self.dense_index.search(vector, top_k)
        return self._make_results(indices[0], scores[0], score_name="dense_score")

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        scores = np.asarray(self.bm25.get_scores(tokenize_tr(query)), dtype="float32")
        if scores.size == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return self._make_results(top_indices, scores[top_indices], score_name="bm25_score")

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        dense_weight: float = 0.55,
        bm25_weight: float = 0.45,
    ) -> list[dict[str, Any]]:
        dense = self.dense_search(query, top_k=candidate_k)
        bm25 = self.bm25_search(query, top_k=candidate_k)
        scores: dict[int, dict[str, float]] = {}

        for result in dense:
            idx = int(result["index_id"])
            scores.setdefault(idx, {"dense_score": 0.0, "bm25_score": 0.0})
            scores[idx]["dense_score"] = float(result["dense_score"])
        for result in bm25:
            idx = int(result["index_id"])
            scores.setdefault(idx, {"dense_score": 0.0, "bm25_score": 0.0})
            scores[idx]["bm25_score"] = float(result["bm25_score"])

        dense_values = np.array([item["dense_score"] for item in scores.values()], dtype="float32")
        bm25_values = np.array([item["bm25_score"] for item in scores.values()], dtype="float32")
        dense_norm = self._minmax(dense_values)
        bm25_norm = self._minmax(bm25_values)

        ranked: list[tuple[int, float, float, float]] = []
        for pos, idx in enumerate(scores.keys()):
            hybrid = dense_weight * float(dense_norm[pos]) + bm25_weight * float(bm25_norm[pos])
            ranked.append((idx, hybrid, scores[idx]["dense_score"], scores[idx]["bm25_score"]))
        ranked.sort(key=lambda item: item[1], reverse=True)

        results = []
        for idx, hybrid, dense_score, bm25_score in ranked[:top_k]:
            record = dict(self.metadata[idx])
            record.update(
                {
                    "score": hybrid,
                    "dense_score": dense_score,
                    "bm25_score": bm25_score,
                    "retriever": "hybrid",
                }
            )
            results.append(record)
        return results

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value == min_value:
            return np.ones_like(values, dtype="float32")
        return (values - min_value) / (max_value - min_value)

    def _make_results(self, indices, scores, score_name: str) -> list[dict[str, Any]]:
        results = []
        for idx, score in zip(indices, scores):
            if int(idx) < 0:
                continue
            record = dict(self.metadata[int(idx)])
            record.update({"score": float(score), score_name: float(score)})
            results.append(record)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a retrieval smoke test.")
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RetrievalEngine(args.index_root, device=args.device)
    if args.mode == "dense":
        results = engine.dense_search(args.query, top_k=args.top_k)
    elif args.mode == "bm25":
        results = engine.bm25_search(args.query, top_k=args.top_k)
    else:
        results = engine.hybrid_search(args.query, top_k=args.top_k)
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.get('citation_label')} | "
            f"{result.get('article_key')} | score={result.get('score'):.4f}"
        )


if __name__ == "__main__":
    main()
