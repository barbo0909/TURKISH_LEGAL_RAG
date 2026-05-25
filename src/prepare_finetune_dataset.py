from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .leakage_check import max_similarity, normalize_for_leakage
except ImportError:
    from leakage_check import max_similarity, normalize_for_leakage


INSTRUCTION = (
    "Aşağıdaki soru veya bağlamdan hareketle Türkçe, kısa, hukuki ve kaynak bilincine sahip bir cevap yaz. "
    "Emin olmadığın veya bağlamda bulunmayan hukuk kuralını uydurma."
)


def make_training_record(question: str, answer: str, context: str = "") -> dict[str, str]:
    input_text = f"Soru: {question}".strip()
    if context:
        input_text += f"\n\nBağlam:\n{context.strip()}"
    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": str(answer).strip(),
    }


def load_questions(path: str | Path) -> list[str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "question" not in df.columns:
        raise ValueError(f"Benchmark is missing question column: {path}")
    return df["question"].astype(str).tolist()


def write_jsonl(records: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_finetune_dataset(
    qa_auxiliary_csv: str | Path,
    benchmark_csv: str | Path,
    output_train_jsonl: str | Path,
    output_val_jsonl: str | Path,
    leakage_report_json: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    similarity_threshold: float = 0.88,
    max_samples: int | None = None,
) -> dict[str, Any]:
    qa_df = pd.read_csv(qa_auxiliary_csv, dtype=str, keep_default_na=False)
    benchmark_questions = load_questions(benchmark_csv)
    benchmark_norm_set = {normalize_for_leakage(question) for question in benchmark_questions}

    required = {"question", "answer"}
    missing = sorted(required - set(qa_df.columns))
    if missing:
        raise ValueError(f"QA auxiliary file is missing columns: {missing}")

    kept_records: list[dict[str, str]] = []
    removed: list[dict[str, Any]] = []

    for _, row in qa_df.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            removed.append({"question": question, "reason": "empty_question_or_answer", "similarity": 0.0})
            continue

        normalized = normalize_for_leakage(question)
        if normalized in benchmark_norm_set:
            removed.append({"question": question, "reason": "exact_benchmark_question", "similarity": 1.0})
            continue

        score, matched_question = max_similarity(question, benchmark_questions)
        if score >= similarity_threshold:
            removed.append(
                {
                    "question": question,
                    "reason": "near_duplicate_benchmark_question",
                    "similarity": score,
                    "matched_benchmark_question": matched_question,
                }
            )
            continue

        kept_records.append(make_training_record(question=question, answer=answer))

    random.Random(seed).shuffle(kept_records)
    if max_samples is not None:
        kept_records = kept_records[:max_samples]

    val_size = max(1, int(len(kept_records) * val_ratio)) if kept_records else 0
    val_records = kept_records[:val_size]
    train_records = kept_records[val_size:]

    write_jsonl(train_records, output_train_jsonl)
    write_jsonl(val_records, output_val_jsonl)

    report = {
        "qa_auxiliary_csv": str(qa_auxiliary_csv),
        "benchmark_csv": str(benchmark_csv),
        "raw_qa_rows": int(len(qa_df)),
        "kept_records": int(len(kept_records)),
        "train_records": int(len(train_records)),
        "val_records": int(len(val_records)),
        "removed_records": int(len(removed)),
        "similarity_threshold": similarity_threshold,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_samples": max_samples,
        "removed_reason_counts": pd.Series([row["reason"] for row in removed]).value_counts().to_dict() if removed else {},
        "removed_examples": removed[:30],
        "output_train_jsonl": str(output_train_jsonl),
        "output_val_jsonl": str(output_val_jsonl),
    }
    report_path = Path(leakage_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe LoRA fine-tuning dataset.")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--leakage-report", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_finetune_dataset(
        qa_auxiliary_csv=args.qa,
        benchmark_csv=args.benchmark,
        output_train_jsonl=args.train_output,
        output_val_jsonl=args.val_output,
        leakage_report_json=args.leakage_report,
        val_ratio=args.val_ratio,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
