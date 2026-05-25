from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

try:
    from .evaluation_retrieval import (
        hit_at_k,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
        split_gold_keys,
    )
    from .optimized_retrieval import OptimizedRetrievalEngine, expand_query, infer_law_filters
except ImportError:
    from evaluation_retrieval import (
        hit_at_k,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
        split_gold_keys,
    )
    from optimized_retrieval import OptimizedRetrievalEngine, expand_query, infer_law_filters


def evaluate_optimized_retrieval(
    benchmark_csv: str | Path,
    index_root: str | Path,
    output_predictions_csv: str | Path,
    output_summary_json: str | Path,
    top_k: int = 30,
    device: str | None = None,
) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    engine = OptimizedRetrievalEngine(index_root=index_root, device=device)
    rows: list[dict[str, Any]] = []

    for _, item in tqdm(benchmark.iterrows(), total=len(benchmark), desc="Optimized retrieval eval"):
        question = item["question"]
        results = engine.search(question, top_k=top_k)
        retrieved_docs = [str(result.get("doc_key", "")) for result in results]
        retrieved_articles = [str(result.get("article_key", "")) for result in results]
        retrieved_citations = [str(result.get("citation_label", "")) for result in results]
        gold_docs = split_gold_keys(item.get("gold_doc_keys", ""))
        gold_articles = split_gold_keys(item.get("gold_article_keys", ""))

        row: dict[str, Any] = {
            "question_id": item.get("question_id", ""),
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", ""),
            "question": question,
            "expanded_query": expand_query(question),
            "law_filters": "; ".join(infer_law_filters(question)),
            "gold_doc_keys": "; ".join(sorted(gold_docs)),
            "gold_article_keys": "; ".join(sorted(gold_articles)),
            "retrieved_doc_keys_top30": "; ".join(retrieved_docs),
            "retrieved_article_keys_top30": "; ".join(retrieved_articles),
            "retrieved_citations_top30": " | ".join(retrieved_citations),
        }
        for k in (5, 10, 30):
            row[f"doc_hit@{k}"] = hit_at_k(retrieved_docs, gold_docs, k)
            row[f"article_hit@{k}"] = hit_at_k(retrieved_articles, gold_articles, k)
            row[f"doc_recall@{k}"] = recall_at_k(retrieved_docs, gold_docs, k)
            row[f"article_recall@{k}"] = recall_at_k(retrieved_articles, gold_articles, k)
            row[f"doc_ndcg@{k}"] = ndcg_at_k(retrieved_docs, gold_docs, k)
            row[f"article_ndcg@{k}"] = ndcg_at_k(retrieved_articles, gold_articles, k)
        row["doc_mrr"] = reciprocal_rank(retrieved_docs, gold_docs)
        row["article_mrr"] = reciprocal_rank(retrieved_articles, gold_articles)
        rows.append(row)

    predictions = pd.DataFrame(rows)
    output_predictions_csv = Path(output_predictions_csv)
    output_summary_json = Path(output_summary_json)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_predictions_csv, index=False, encoding="utf-8-sig")

    metric_cols = [col for col in predictions.columns if "@" in col or col.endswith("_mrr")]
    summary = {
        "mode": "optimized_law_aware_dense_v2",
        "benchmark_csv": str(benchmark_csv),
        "index_root": str(index_root),
        "question_count": int(len(predictions)),
        "top_k": top_k,
        "predictions_csv": str(output_predictions_csv),
        "metrics": {col: float(predictions[col].mean()) for col in metric_cols},
        "metrics_by_topic": {
            topic: group[metric_cols].mean().to_dict()
            for topic, group in predictions.groupby("topic", sort=True)
        },
        "law_filter_rate": float(predictions["law_filters"].astype(str).str.strip().ne("").mean()),
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optimized law-aware retrieval.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_optimized_retrieval(
        benchmark_csv=args.benchmark,
        index_root=args.index_root,
        output_predictions_csv=args.output_predictions,
        output_summary_json=args.output_summary,
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
