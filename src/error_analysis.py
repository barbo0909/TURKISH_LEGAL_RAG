from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ERROR_PRIORITY = {
    "retrieval_miss": 1,
    "wrong_or_unsupported_citation": 2,
    "missing_citation": 3,
    "low_answer_overlap": 4,
    "acceptable_automatic": 5,
}


def make_error_analysis(eval_csv: str | Path, output_csv: str | Path, per_type_limit: int = 30) -> pd.DataFrame:
    df = pd.read_csv(eval_csv, dtype=str, keep_default_na=False)
    numeric_cols = [
        "token_f1",
        "rouge_l",
        "retrieval_gold_available",
        "citation_present",
        "citation_gold_match",
        "grounded_citation_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "error_type_auto" not in df.columns:
        raise ValueError("eval_csv must contain error_type_auto. Re-run evaluation_qa first.")

    df["error_priority"] = df["error_type_auto"].map(ERROR_PRIORITY).fillna(99)
    df = df.sort_values(["error_priority", "token_f1"], ascending=[True, True])

    selected = (
        df.groupby("error_type_auto", sort=False)
        .head(per_type_limit)
        .reset_index(drop=True)
    )

    columns = [
        "question_id",
        "topic",
        "difficulty",
        "question",
        "gold_answer",
        "generated_answer",
        "gold_article_keys",
        "retrieved_article_keys",
        "retrieved_citations",
        "token_f1",
        "rouge_l",
        "retrieval_gold_available",
        "citation_present",
        "citation_gold_match",
        "grounded_citation_score",
        "error_type_auto",
    ]
    columns = [col for col in columns if col in selected.columns]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(output_csv, index=False, encoding="utf-8-sig")
    return selected[columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create automatic Base RAG error analysis CSV.")
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-type-limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = make_error_analysis(args.eval_csv, args.output, per_type_limit=args.per_type_limit)
    print(df["error_type_auto"].value_counts().to_string())


if __name__ == "__main__":
    main()
