from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_generation_summary(system_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "system_name": system_name,
        "question_count": summary.get("question_count"),
        "predictions_csv": summary.get("predictions_csv"),
        "eval_csv": summary.get("eval_csv"),
    }
    row.update(summary.get("metrics", {}))
    return row


def compare_generation_systems(
    base_summary_json: str | Path,
    finetuned_summary_json: str | Path,
    output_csv: str | Path,
    output_markdown: str | Path,
) -> pd.DataFrame:
    base = flatten_generation_summary("base_rag", read_json(base_summary_json))
    ft = flatten_generation_summary("finetuned_rag", read_json(finetuned_summary_json))
    df = pd.DataFrame([base, ft])

    metric_cols = [
        "token_f1",
        "rouge_l",
        "retrieval_gold_available",
        "citation_present",
        "citation_gold_match",
        "grounded_citation_score",
        "unsupported_or_missing_citation",
    ]
    delta = {"system_name": "delta_finetuned_minus_base", "question_count": ""}
    for col in metric_cols:
        if col in df.columns:
            delta[col] = float(df.loc[df["system_name"] == "finetuned_rag", col].iloc[0]) - float(
                df.loc[df["system_name"] == "base_rag", col].iloc[0]
            )
    df = pd.concat([df, pd.DataFrame([delta])], ignore_index=True)

    output_csv = Path(output_csv)
    output_markdown = Path(output_markdown)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_markdown.write_text(df.to_markdown(index=False), encoding="utf-8")
    return df


def build_experiment_summary(
    normalization_report_json: str | Path,
    leakage_report_json: str | Path,
    external_leakage_report_json: str | Path,
    base_summary_json: str | Path,
    finetuned_summary_json: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    normalization = read_json(normalization_report_json)
    leakage = read_json(leakage_report_json)
    external_leakage = read_json(external_leakage_report_json)
    base = read_json(base_summary_json)
    ft = read_json(finetuned_summary_json)

    rows = [
        {
            "experiment_id": "corpus_normalization_v3",
            "stage": "corpus",
            "raw_rows": normalization.get("raw_rows"),
            "valid_law_rows": normalization.get("valid_law_rows"),
            "qa_rows": normalization.get("qa_rows"),
            "rejected_rows": normalization.get("rejected_rows"),
            "notes": "Official-law corpus only; QA separated as auxiliary.",
        },
        {
            "experiment_id": "finetune_dataset_qa_aux",
            "stage": "data",
            "train_records": leakage.get("train_records"),
            "val_records": leakage.get("val_records"),
            "removed_records": leakage.get("removed_records"),
            "notes": "QA auxiliary fine-tune data after benchmark leakage filtering.",
        },
        {
            "experiment_id": "finetune_dataset_external_combined",
            "stage": "data",
            "train_records": external_leakage.get("combined_train_records"),
            "val_records": external_leakage.get("combined_val_records"),
            "removed_records": external_leakage.get("removed_external_records"),
            "notes": "External instructor SFT plus QA auxiliary; benchmark leakage filtered.",
        },
        {
            "experiment_id": "base_rag_dense_top10",
            "stage": "generation",
            **base.get("metrics", {}),
            "notes": "Base checkpoint, dense retrieval, top_k_context=10.",
        },
        {
            "experiment_id": "finetuned_rag_lora_medium_dense_top10",
            "stage": "generation",
            **ft.get("metrics", {}),
            "notes": "Same base checkpoint plus LoRA medium adapter; same retrieval/prompt/decoding.",
        },
    ]
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready experiment summary tables.")
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--finetuned-summary", required=True)
    parser.add_argument("--normalization-report", required=True)
    parser.add_argument("--leakage-report", required=True)
    parser.add_argument("--external-leakage-report", required=True)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--comparison-md", required=True)
    parser.add_argument("--experiment-summary-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_generation_systems(
        base_summary_json=args.base_summary,
        finetuned_summary_json=args.finetuned_summary,
        output_csv=args.comparison_csv,
        output_markdown=args.comparison_md,
    )
    build_experiment_summary(
        normalization_report_json=args.normalization_report,
        leakage_report_json=args.leakage_report,
        external_leakage_report_json=args.external_leakage_report,
        base_summary_json=args.base_summary,
        finetuned_summary_json=args.finetuned_summary,
        output_csv=args.experiment_summary_csv,
    )


if __name__ == "__main__":
    main()
