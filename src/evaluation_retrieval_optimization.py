from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
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
    from .retrieval import RetrievalEngine
except ImportError:
    from evaluation_retrieval import (
        hit_at_k,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
        split_gold_keys,
    )
    from retrieval import RetrievalEngine


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype="float32")
    if array.size == 0:
        return array
    lo = float(array.min())
    hi = float(array.max())
    if hi == lo:
        return np.ones_like(array, dtype="float32")
    return (array - lo) / (hi - lo)


def combine_dense_bm25(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    dense_weight: float,
    bm25_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    scores: dict[int, dict[str, float]] = {}

    for result in dense_results:
        idx = int(result["index_id"])
        by_index[idx] = dict(result)
        scores.setdefault(idx, {"dense_score": 0.0, "bm25_score": 0.0})
        scores[idx]["dense_score"] = float(result.get("dense_score", result.get("score", 0.0)))

    for result in bm25_results:
        idx = int(result["index_id"])
        by_index.setdefault(idx, dict(result))
        scores.setdefault(idx, {"dense_score": 0.0, "bm25_score": 0.0})
        scores[idx]["bm25_score"] = float(result.get("bm25_score", result.get("score", 0.0)))

    index_ids = list(scores.keys())
    dense_norm = minmax([scores[idx]["dense_score"] for idx in index_ids])
    bm25_norm = minmax([scores[idx]["bm25_score"] for idx in index_ids])

    ranked: list[dict[str, Any]] = []
    for pos, idx in enumerate(index_ids):
        record = dict(by_index[idx])
        score = dense_weight * float(dense_norm[pos]) + bm25_weight * float(bm25_norm[pos])
        record.update(
            {
                "score": score,
                "dense_score": scores[idx]["dense_score"],
                "bm25_score": scores[idx]["bm25_score"],
                "retriever": "hybrid_sweep",
            }
        )
        ranked.append(record)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def metrics_for_results(results: list[dict[str, Any]], gold_docs: set[str], gold_articles: set[str]) -> dict[str, float]:
    doc_keys = [str(result.get("doc_key", "")) for result in results]
    article_keys = [str(result.get("article_key", "")) for result in results]
    metrics: dict[str, float] = {}
    for k in (5, 10, 30):
        metrics[f"doc_hit@{k}"] = hit_at_k(doc_keys, gold_docs, k)
        metrics[f"article_hit@{k}"] = hit_at_k(article_keys, gold_articles, k)
        metrics[f"doc_recall@{k}"] = recall_at_k(doc_keys, gold_docs, k)
        metrics[f"article_recall@{k}"] = recall_at_k(article_keys, gold_articles, k)
        metrics[f"doc_ndcg@{k}"] = ndcg_at_k(doc_keys, gold_docs, k)
        metrics[f"article_ndcg@{k}"] = ndcg_at_k(article_keys, gold_articles, k)
    metrics["doc_mrr"] = reciprocal_rank(doc_keys, gold_docs)
    metrics["article_mrr"] = reciprocal_rank(article_keys, gold_articles)
    return metrics


def summarize_metric_rows(rows: list[dict[str, Any]], mode: str, dense_weight: float | None = None) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    metric_columns = [col for col in df.columns if "@" in col or col.endswith("_mrr")]
    summary = {
        "mode": mode,
        "dense_weight": dense_weight,
        "bm25_weight": None if dense_weight is None else round(1.0 - dense_weight, 4),
        "question_count": int(len(df)),
    }
    summary.update({col: float(df[col].mean()) for col in metric_columns})
    return summary


def evaluate_weight_sweep(
    benchmark_csv: str | Path,
    index_root: str | Path,
    output_dir: str | Path,
    candidate_k: int = 30,
    device: str | None = None,
    dense_weights: list[float] | None = None,
) -> pd.DataFrame:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    engine = RetrievalEngine(index_root=index_root, device=device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_weights = dense_weights or [0.0, 0.25, 0.5, 0.55, 0.65, 0.75, 0.85, 0.9, 1.0]

    dense_rows: list[dict[str, Any]] = []
    bm25_rows: list[dict[str, Any]] = []
    hybrid_rows_by_weight: dict[float, list[dict[str, Any]]] = {weight: [] for weight in dense_weights}

    for _, item in tqdm(benchmark.iterrows(), total=len(benchmark), desc="Candidate sweep"):
        question = item["question"]
        gold_docs = split_gold_keys(item.get("gold_doc_keys", ""))
        gold_articles = split_gold_keys(item.get("gold_article_keys", ""))

        dense_results = engine.dense_search(question, top_k=candidate_k)
        bm25_results = engine.bm25_search(question, top_k=candidate_k)

        base = {
            "question_id": item.get("question_id", ""),
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", ""),
        }

        dense_rows.append({**base, **metrics_for_results(dense_results, gold_docs, gold_articles)})
        bm25_rows.append({**base, **metrics_for_results(bm25_results, gold_docs, gold_articles)})

        for weight in dense_weights:
            combined = combine_dense_bm25(
                dense_results=dense_results,
                bm25_results=bm25_results,
                dense_weight=weight,
                bm25_weight=1.0 - weight,
                top_k=candidate_k,
            )
            hybrid_rows_by_weight[weight].append({**base, **metrics_for_results(combined, gold_docs, gold_articles)})

    summaries = [
        summarize_metric_rows(dense_rows, mode="dense", dense_weight=1.0),
        summarize_metric_rows(bm25_rows, mode="bm25", dense_weight=0.0),
    ]
    for weight, rows in hybrid_rows_by_weight.items():
        summaries.append(summarize_metric_rows(rows, mode="hybrid", dense_weight=weight))

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        ["article_hit@5", "article_mrr", "article_hit@10"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(output_dir / "retrieval_weight_sweep_summary_v1.csv", index=False, encoding="utf-8-sig")

    report = {
        "benchmark_csv": str(benchmark_csv),
        "index_root": str(index_root),
        "candidate_k": candidate_k,
        "best_by_article_hit@5": summary_df.iloc[0].to_dict(),
        "summary_csv": str(output_dir / "retrieval_weight_sweep_summary_v1.csv"),
    }
    (output_dir / "retrieval_weight_sweep_report_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval top30 and hybrid weight sweep.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = evaluate_weight_sweep(
        benchmark_csv=args.benchmark,
        index_root=args.index_root,
        output_dir=args.output_dir,
        candidate_k=args.candidate_k,
        device=args.device,
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
