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
    from .reranking import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
    from .retrieval import RetrievalEngine
except ImportError:
    from evaluation_retrieval import (
        hit_at_k,
        ndcg_at_k,
        recall_at_k,
        reciprocal_rank,
        split_gold_keys,
    )
    from reranking import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
    from retrieval import RetrievalEngine


def compute_metrics(results: list[dict[str, Any]], gold_docs: set[str], gold_articles: set[str]) -> dict[str, float]:
    doc_keys = [str(result.get("doc_key", "")) for result in results]
    article_keys = [str(result.get("article_key", "")) for result in results]
    metrics: dict[str, float] = {}
    for k in (5, 10):
        metrics[f"doc_hit@{k}"] = hit_at_k(doc_keys, gold_docs, k)
        metrics[f"article_hit@{k}"] = hit_at_k(article_keys, gold_articles, k)
        metrics[f"doc_recall@{k}"] = recall_at_k(doc_keys, gold_docs, k)
        metrics[f"article_recall@{k}"] = recall_at_k(article_keys, gold_articles, k)
        metrics[f"doc_ndcg@{k}"] = ndcg_at_k(doc_keys, gold_docs, k)
        metrics[f"article_ndcg@{k}"] = ndcg_at_k(article_keys, gold_articles, k)
    metrics["doc_mrr"] = reciprocal_rank(doc_keys, gold_docs)
    metrics["article_mrr"] = reciprocal_rank(article_keys, gold_articles)
    return metrics


def evaluate_reranker(
    benchmark_csv: str | Path,
    index_root: str | Path,
    output_predictions_csv: str | Path,
    output_summary_json: str | Path,
    candidate_mode: str = "dense",
    candidate_k: int = 30,
    top_k: int = 10,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    batch_size: int = 16,
    device: str | None = None,
    dense_weight: float = 1.0,
    bm25_weight: float = 0.0,
) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    engine = RetrievalEngine(index_root=index_root, device=device)
    reranker = CrossEncoderReranker(model_name=reranker_model, device=device)

    rows: list[dict[str, Any]] = []
    for _, item in tqdm(benchmark.iterrows(), total=len(benchmark), desc="Reranker eval"):
        question = item["question"]
        gold_docs = split_gold_keys(item.get("gold_doc_keys", ""))
        gold_articles = split_gold_keys(item.get("gold_article_keys", ""))

        if candidate_mode == "dense":
            candidates = engine.dense_search(question, top_k=candidate_k)
        elif candidate_mode == "bm25":
            candidates = engine.bm25_search(question, top_k=candidate_k)
        elif candidate_mode == "hybrid":
            candidates = engine.hybrid_search(
                question,
                top_k=candidate_k,
                candidate_k=candidate_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
        else:
            raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

        reranked = reranker.rerank(
            query=question,
            candidates=candidates,
            text_field="retrieval_text",
            top_k=top_k,
            batch_size=batch_size,
        )
        metrics = compute_metrics(reranked, gold_docs, gold_articles)

        rows.append(
            {
                "question_id": item.get("question_id", ""),
                "topic": item.get("topic", ""),
                "difficulty": item.get("difficulty", ""),
                "question": question,
                "gold_doc_keys": "; ".join(sorted(gold_docs)),
                "gold_article_keys": "; ".join(sorted(gold_articles)),
                "retrieved_doc_keys_top10": "; ".join(str(row.get("doc_key", "")) for row in reranked),
                "retrieved_article_keys_top10": "; ".join(str(row.get("article_key", "")) for row in reranked),
                "retrieved_citations_top10": " | ".join(str(row.get("citation_label", "")) for row in reranked),
                "reranker_scores_top10": "; ".join(str(round(float(row.get("reranker_score", 0.0)), 6)) for row in reranked),
                **metrics,
            }
        )

    predictions = pd.DataFrame(rows)
    output_predictions_csv = Path(output_predictions_csv)
    output_summary_json = Path(output_summary_json)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_predictions_csv, index=False, encoding="utf-8-sig")

    metric_cols = [col for col in predictions.columns if "@" in col or col.endswith("_mrr")]
    summary = {
        "candidate_mode": candidate_mode,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "reranker_model": reranker_model,
        "question_count": int(len(predictions)),
        "predictions_csv": str(output_predictions_csv),
        "metrics": {col: float(predictions[col].mean()) for col in metric_cols},
        "metrics_by_topic": {
            topic: group[metric_cols].mean().to_dict()
            for topic, group in predictions.groupby("topic", sort=True)
        },
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-encoder reranking on retrieval candidates.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--candidate-mode", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_reranker(
        benchmark_csv=args.benchmark,
        index_root=args.index_root,
        output_predictions_csv=args.output_predictions,
        output_summary_json=args.output_summary,
        candidate_mode=args.candidate_mode,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        reranker_model=args.reranker_model,
        batch_size=args.batch_size,
        device=args.device,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
