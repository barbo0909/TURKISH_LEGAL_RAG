from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

try:
    from .retrieval import RetrievalEngine
except ImportError:
    from retrieval import RetrievalEngine


def split_gold_keys(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def dedupe_ranked(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_keys: list[str] = []
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
    return unique_keys


def ndcg_at_k(retrieved_keys: list[str], gold_keys: set[str], k: int) -> float:
    if not gold_keys:
        return 0.0
    unique_retrieved = dedupe_ranked(retrieved_keys)
    rel = [1 if key in gold_keys else 0 for key in unique_retrieved[:k]]
    rel = rel + [0] * max(0, k - len(rel))
    ideal_hits = min(len(gold_keys), k)
    ideal = [1] * ideal_hits + [0] * (k - ideal_hits)
    ideal_dcg = dcg(ideal)
    return dcg(rel) / ideal_dcg if ideal_dcg else 0.0


def reciprocal_rank(retrieved_keys: list[str], gold_keys: set[str]) -> float:
    for rank, key in enumerate(dedupe_ranked(retrieved_keys), start=1):
        if key in gold_keys:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_keys: list[str], gold_keys: set[str], k: int) -> int:
    unique_retrieved = dedupe_ranked(retrieved_keys)
    return int(any(key in gold_keys for key in unique_retrieved[:k]))


def recall_at_k(retrieved_keys: list[str], gold_keys: set[str], k: int) -> float:
    if not gold_keys:
        return 0.0
    unique_retrieved = dedupe_ranked(retrieved_keys)
    return len(set(unique_retrieved[:k]) & gold_keys) / len(gold_keys)


def evaluate_retrieval(
    benchmark_csv: str | Path,
    index_root: str | Path,
    output_predictions_csv: str | Path,
    output_summary_json: str | Path,
    mode: str = "hybrid",
    top_k: int = 10,
    candidate_k: int = 30,
    dense_weight: float = 0.55,
    bm25_weight: float = 0.45,
    device: str | None = None,
) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    engine = RetrievalEngine(index_root=index_root, device=device)

    rows: list[dict[str, Any]] = []
    for _, item in tqdm(benchmark.iterrows(), total=len(benchmark), desc=f"Retrieval eval: {mode}"):
        question = item["question"]
        gold_article_keys = split_gold_keys(item.get("gold_article_keys", ""))
        gold_doc_keys = split_gold_keys(item.get("gold_doc_keys", ""))

        if mode == "dense":
            results = engine.dense_search(question, top_k=top_k)
        elif mode == "bm25":
            results = engine.bm25_search(question, top_k=top_k)
        elif mode == "hybrid":
            results = engine.hybrid_search(
                question,
                top_k=top_k,
                candidate_k=candidate_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        retrieved_article_keys = [str(result.get("article_key", "")) for result in results]
        retrieved_doc_keys = [str(result.get("doc_key", "")) for result in results]
        retrieved_labels = [str(result.get("citation_label", "")) for result in results]
        retrieved_scores = [str(round(float(result.get("score", 0.0)), 6)) for result in results]

        row = {
            "question_id": item.get("question_id", ""),
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", ""),
            "question": question,
            "gold_doc_keys": "; ".join(sorted(gold_doc_keys)),
            "gold_article_keys": "; ".join(sorted(gold_article_keys)),
            "retrieved_doc_keys_top10": "; ".join(retrieved_doc_keys),
            "retrieved_article_keys_top10": "; ".join(retrieved_article_keys),
            "retrieved_citations_top10": " | ".join(retrieved_labels),
            "retrieved_scores_top10": "; ".join(retrieved_scores),
            "doc_hit@5": hit_at_k(retrieved_doc_keys, gold_doc_keys, 5),
            "doc_hit@10": hit_at_k(retrieved_doc_keys, gold_doc_keys, 10),
            "article_hit@5": hit_at_k(retrieved_article_keys, gold_article_keys, 5),
            "article_hit@10": hit_at_k(retrieved_article_keys, gold_article_keys, 10),
            "doc_recall@5": recall_at_k(retrieved_doc_keys, gold_doc_keys, 5),
            "doc_recall@10": recall_at_k(retrieved_doc_keys, gold_doc_keys, 10),
            "article_recall@5": recall_at_k(retrieved_article_keys, gold_article_keys, 5),
            "article_recall@10": recall_at_k(retrieved_article_keys, gold_article_keys, 10),
            "doc_mrr": reciprocal_rank(retrieved_doc_keys, gold_doc_keys),
            "article_mrr": reciprocal_rank(retrieved_article_keys, gold_article_keys),
            "doc_ndcg@5": ndcg_at_k(retrieved_doc_keys, gold_doc_keys, 5),
            "doc_ndcg@10": ndcg_at_k(retrieved_doc_keys, gold_doc_keys, 10),
            "article_ndcg@5": ndcg_at_k(retrieved_article_keys, gold_article_keys, 5),
            "article_ndcg@10": ndcg_at_k(retrieved_article_keys, gold_article_keys, 10),
        }
        rows.append(row)

    predictions = pd.DataFrame(rows)
    output_predictions_csv = Path(output_predictions_csv)
    output_summary_json = Path(output_summary_json)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_predictions_csv, index=False, encoding="utf-8-sig")

    metric_cols = [
        "doc_hit@5",
        "doc_hit@10",
        "article_hit@5",
        "article_hit@10",
        "doc_recall@5",
        "doc_recall@10",
        "article_recall@5",
        "article_recall@10",
        "doc_mrr",
        "article_mrr",
        "doc_ndcg@5",
        "doc_ndcg@10",
        "article_ndcg@5",
        "article_ndcg@10",
    ]
    summary = {
        "mode": mode,
        "benchmark_csv": str(benchmark_csv),
        "index_root": str(index_root),
        "question_count": int(len(predictions)),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "predictions_csv": str(output_predictions_csv),
        "metrics": {col: float(predictions[col].mean()) for col in metric_cols},
        "metrics_by_topic": {
            topic: group[metric_cols].mean().to_dict()
            for topic, group in predictions.groupby("topic", sort=True)
        },
        "metrics_by_difficulty": {
            difficulty: group[metric_cols].mean().to_dict()
            for difficulty, group in predictions.groupby("difficulty", sort=True)
        },
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the locked gold benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=0.55)
    parser.add_argument("--bm25-weight", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_retrieval(
        benchmark_csv=args.benchmark,
        index_root=args.index_root,
        output_predictions_csv=args.output_predictions,
        output_summary_json=args.output_summary,
        mode=args.mode,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
