from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_law_no(value: str) -> str:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def normalize_gold_article(value: str) -> str:
    text = str(value).upper()
    text = text.replace("GEÇICI", "GEÇİCİ").replace("MUKERRER", "MÜKERRER")
    match = re.search(r"(\d+(?:/[A-ZÇĞİÖŞÜ0-9]+)?)", text)
    if not match:
        return ""
    number = match.group(1)
    if "GEÇİCİ" in text:
        return f"GEÇİCİ MADDE {number}"
    if re.search(r"\bEK\b", text):
        return f"EK MADDE {number}"
    if "MÜKERRER" in text:
        return f"MÜKERRER MADDE {number}"
    return f"MADDE {number}"


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def join_unique(values: pd.Series) -> str:
    return "; ".join(sorted({str(value) for value in values if str(value).strip()}))


def prepare_gold_benchmark(
    benchmark_input: str | Path,
    normalized_corpus: str | Path,
    output_csv: str | Path,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    benchmark_df = read_csv(benchmark_input)
    corpus_df = read_csv(normalized_corpus)

    required_benchmark = {
        "id",
        "legal_domain",
        "difficulty",
        "question",
        "gold_answer",
        "gold_source",
        "gold_article",
    }
    required_corpus = {
        "doc_key",
        "article_key",
        "law_no_norm",
        "law_name_norm",
        "article_no_norm",
        "source_url",
    }
    missing_benchmark = sorted(required_benchmark - set(benchmark_df.columns))
    missing_corpus = sorted(required_corpus - set(corpus_df.columns))
    if missing_benchmark:
        raise ValueError(f"Benchmark input is missing columns: {missing_benchmark}")
    if missing_corpus:
        raise ValueError(f"Normalized corpus is missing columns: {missing_corpus}")

    article_source = "gold_article_raw" if "gold_article_raw" in benchmark_df.columns else "gold_article"
    benchmark_df["law_no_norm"] = benchmark_df["gold_source"].map(normalize_law_no)
    benchmark_df["article_no_norm"] = benchmark_df[article_source].where(
        benchmark_df[article_source].astype(str).str.strip().ne(""),
        benchmark_df["gold_article"],
    ).map(normalize_gold_article)

    lookup = corpus_df[
        [
            "doc_key",
            "article_key",
            "law_no_norm",
            "law_name_norm",
            "article_no_norm",
            "source_url",
            "citation_label",
        ]
    ].drop_duplicates()

    merged = benchmark_df.merge(lookup, on=["law_no_norm", "article_no_norm"], how="left", suffixes=("_benchmark", "_corpus"))
    if "source_url_corpus" not in merged.columns:
        merged["source_url_corpus"] = ""
    for optional_column in ["benchmark_tier", "question_type", "verification_status", "review_priority", "notes"]:
        if optional_column not in merged.columns:
            merged[optional_column] = ""

    grouped = (
        merged.groupby("id", sort=False)
        .agg(
            question_id=("id", "first"),
            topic=("legal_domain", "first"),
            question=("question", "first"),
            gold_answer=("gold_answer", "first"),
            gold_doc_keys=("doc_key", join_unique),
            gold_article_keys=("article_key", join_unique),
            gold_law=("law_name_norm", lambda s: next((str(v) for v in s if str(v).strip()), "")),
            gold_article_no=("article_no_norm", "first"),
            difficulty=("difficulty", "first"),
            source=("gold_source", "first"),
            source_url=("source_url_corpus", join_unique),
            benchmark_tier=("benchmark_tier", "first"),
            question_type=("question_type", "first"),
            verification_status=("verification_status", "first"),
            review_priority=("review_priority", "first"),
            notes=("notes", "first"),
        )
        .reset_index(drop=True)
    )

    output_columns = [
        "question_id",
        "topic",
        "question",
        "gold_answer",
        "gold_doc_keys",
        "gold_article_keys",
        "gold_law",
        "gold_article_no",
        "difficulty",
        "source",
        "source_url",
        "benchmark_tier",
        "question_type",
        "verification_status",
        "review_priority",
        "notes",
    ]
    grouped = grouped[output_columns]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report = {
        "input_rows": int(len(benchmark_df)),
        "output_rows": int(len(grouped)),
        "matched_rows": int(grouped["gold_article_keys"].astype(str).str.strip().ne("").sum()),
        "unmatched_rows": int(grouped["gold_article_keys"].astype(str).str.strip().eq("").sum()),
        "duplicate_question_ids": int(benchmark_df["id"].duplicated().sum()),
        "topic_counts": grouped["topic"].value_counts().to_dict(),
        "difficulty_counts": grouped["difficulty"].value_counts().to_dict(),
        "output_csv": str(output_csv),
    }

    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare locked Q-A-Doc gold benchmark.")
    parser.add_argument("--input", required=True, help="Input benchmark candidate CSV")
    parser.add_argument("--corpus", required=True, help="Normalized official law corpus CSV")
    parser.add_argument("--output", required=True, help="Output gold_benchmark_v1.csv")
    parser.add_argument("--report", default=None, help="Optional benchmark preparation report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_gold_benchmark(
        benchmark_input=args.input,
        normalized_corpus=args.corpus,
        output_csv=args.output,
        report_json=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
