from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(TOKEN_RE.findall(value))


def benchmark_questions(benchmark_csv: str | Path) -> list[str]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    return [normalize_text(item) for item in benchmark.get("question", pd.Series(dtype=str)).tolist() if item]


def leakage_flags(
    candidate_questions: list[str],
    benchmark: list[str],
    threshold: float = 0.88,
) -> list[tuple[bool, float, str]]:
    if not candidate_questions:
        return []
    if not benchmark:
        return [(False, 0.0, "") for _ in candidate_questions]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(benchmark + candidate_questions)
    bench_matrix = matrix[: len(benchmark)]
    cand_matrix = matrix[len(benchmark) :]
    sims = cosine_similarity(cand_matrix, bench_matrix)

    flags: list[tuple[bool, float, str]] = []
    for row in sims:
        idx = int(row.argmax())
        score = float(row[idx])
        flags.append((score >= threshold, score, benchmark[idx]))
    return flags


def split_rows(
    rows: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled[val_size:], shuffled[:val_size]


def prepare_embedding_data(
    embedding_jsonl: str | Path,
    benchmark_csv: str | Path,
    output_train_jsonl: str | Path,
    output_val_jsonl: str | Path,
    output_report_json: str | Path,
    leakage_threshold: float = 0.88,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    raw_rows = read_jsonl(embedding_jsonl)
    candidates = [normalize_text(row.get("query", "")) for row in raw_rows]
    leakage = leakage_flags(candidates, benchmark_questions(benchmark_csv), threshold=leakage_threshold)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for row, (is_leak, score, matched_question) in zip(raw_rows, leakage):
        if is_leak:
            removed.append(
                {
                    "id": row.get("id", ""),
                    "query": row.get("query", ""),
                    "similarity": score,
                    "matched_benchmark_question": matched_question,
                    "reason": "near_duplicate_benchmark_question",
                }
            )
            continue
        kept.append(
            {
                "id": row.get("id", ""),
                "query": row.get("query", ""),
                "positive_passage": row.get("positive_passage", ""),
                "negative_passage": row.get("negative_passage", ""),
                "positive_id": row.get("positive_id", ""),
                "negative_id": row.get("negative_id", ""),
                "positive_citation": row.get("positive_citation", ""),
                "negative_citation": row.get("negative_citation", ""),
                "negative_type": row.get("negative_type", ""),
                "source": row.get("source", ""),
            }
        )

    train, val = split_rows(kept, val_ratio=val_ratio, seed=seed)
    write_jsonl(output_train_jsonl, train)
    write_jsonl(output_val_jsonl, val)

    report = {
        "input_jsonl": str(embedding_jsonl),
        "raw_records": len(raw_rows),
        "kept_records": len(kept),
        "removed_records": len(removed),
        "train_records": len(train),
        "val_records": len(val),
        "leakage_threshold": leakage_threshold,
        "removed_reason_counts": Counter(item["reason"] for item in removed),
        "removed_examples": removed[:10],
        "output_train_jsonl": str(output_train_jsonl),
        "output_val_jsonl": str(output_val_jsonl),
    }
    Path(output_report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_reranker_data(
    reranker_jsonl: str | Path,
    benchmark_csv: str | Path,
    output_train_jsonl: str | Path,
    output_val_jsonl: str | Path,
    output_report_json: str | Path,
    leakage_threshold: float = 0.88,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    raw_rows = read_jsonl(reranker_jsonl)
    candidates = [normalize_text(row.get("query", "")) for row in raw_rows]
    leakage = leakage_flags(candidates, benchmark_questions(benchmark_csv), threshold=leakage_threshold)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for row, (is_leak, score, matched_question) in zip(raw_rows, leakage):
        if is_leak:
            removed.append(
                {
                    "id": row.get("id", ""),
                    "query": row.get("query", ""),
                    "similarity": score,
                    "matched_benchmark_question": matched_question,
                    "reason": "near_duplicate_benchmark_question",
                }
            )
            continue
        kept.append(
            {
                "id": row.get("id", ""),
                "query_id": row.get("query_id", ""),
                "query": row.get("query", ""),
                "candidate_passage": row.get("candidate_passage", ""),
                "label": int(row.get("label", 0)),
                "candidate_id": row.get("candidate_id", ""),
                "citation_label": row.get("citation_label", ""),
                "source": row.get("source", ""),
                "negative_type": row.get("negative_type", ""),
                "audit_status": row.get("audit_status", ""),
            }
        )

    train, val = split_rows(kept, val_ratio=val_ratio, seed=seed)
    write_jsonl(output_train_jsonl, train)
    write_jsonl(output_val_jsonl, val)

    report = {
        "input_jsonl": str(reranker_jsonl),
        "raw_records": len(raw_rows),
        "kept_records": len(kept),
        "removed_records": len(removed),
        "train_records": len(train),
        "val_records": len(val),
        "label_counts": Counter(str(item["label"]) for item in kept),
        "leakage_threshold": leakage_threshold,
        "removed_reason_counts": Counter(item["reason"] for item in removed),
        "removed_examples": removed[:10],
        "output_train_jsonl": str(output_train_jsonl),
        "output_val_jsonl": str(output_val_jsonl),
    }
    Path(output_report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare no-leakage embedding and reranker tuning data.")
    parser.add_argument("--embedding-jsonl", required=True)
    parser.add_argument("--reranker-jsonl", required=True)
    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--leakage-threshold", type=float, default=0.88)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    embedding_report = prepare_embedding_data(
        embedding_jsonl=args.embedding_jsonl,
        benchmark_csv=args.benchmark_csv,
        output_train_jsonl=output_dir / "embedding_tune_train.jsonl",
        output_val_jsonl=output_dir / "embedding_tune_val.jsonl",
        output_report_json=reports_dir / "embedding_tuning_data_report.json",
        leakage_threshold=args.leakage_threshold,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    reranker_report = prepare_reranker_data(
        reranker_jsonl=args.reranker_jsonl,
        benchmark_csv=args.benchmark_csv,
        output_train_jsonl=output_dir / "reranker_tune_train.jsonl",
        output_val_jsonl=output_dir / "reranker_tune_val.jsonl",
        output_report_json=reports_dir / "reranker_tuning_data_report.json",
        leakage_threshold=args.leakage_threshold,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps({"embedding": embedding_report, "reranker": reranker_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
