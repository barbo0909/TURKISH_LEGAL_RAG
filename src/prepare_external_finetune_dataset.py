from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .leakage_check import max_similarity, normalize_for_leakage
except ImportError:
    from leakage_check import max_similarity, normalize_for_leakage


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_jsonl(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_benchmark_questions(path: str | Path) -> list[str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df["question"].astype(str).tolist()


def extract_question_from_user(user_content: str) -> str:
    matches = re.findall(r"Soru:\s*(.+)", str(user_content), flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return str(user_content).strip()


def convert_external_record(record: dict[str, Any]) -> dict[str, str] | None:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return None

    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    if not user.strip() or not assistant.strip():
        return None

    instruction = system.strip() or (
        "Aşağıdaki kaynak bağlamına dayanarak Türkçe, kısa ve kaynaklı bir hukuki cevap yaz. "
        "Kaynakta olmayan bilgiyi üretme."
    )
    return {
        "instruction": instruction,
        "input": user.strip(),
        "output": assistant.strip(),
    }


def prepare_external_and_combined_finetune_dataset(
    external_llm_jsonl: str | Path,
    benchmark_csv: str | Path,
    existing_train_jsonl: str | Path,
    existing_val_jsonl: str | Path,
    output_train_jsonl: str | Path,
    output_val_jsonl: str | Path,
    output_external_train_jsonl: str | Path,
    output_external_val_jsonl: str | Path,
    report_json: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    similarity_threshold: float = 0.88,
    max_external_samples: int | None = None,
) -> dict[str, Any]:
    raw_records = read_jsonl(external_llm_jsonl)
    benchmark_questions = load_benchmark_questions(benchmark_csv)
    benchmark_norm = {normalize_for_leakage(q) for q in benchmark_questions}

    kept: list[dict[str, str]] = []
    removed: list[dict[str, Any]] = []
    for record in raw_records:
        converted = convert_external_record(record)
        if converted is None:
            removed.append({"id": record.get("id", ""), "reason": "missing_user_or_assistant"})
            continue

        question_like = extract_question_from_user(converted["input"])
        normalized = normalize_for_leakage(question_like)
        if normalized in benchmark_norm:
            removed.append({"id": record.get("id", ""), "reason": "exact_benchmark_question", "question": question_like})
            continue

        score, matched = max_similarity(question_like, benchmark_questions)
        if score >= similarity_threshold:
            removed.append(
                {
                    "id": record.get("id", ""),
                    "reason": "near_duplicate_benchmark_question",
                    "similarity": score,
                    "question": question_like,
                    "matched_benchmark_question": matched,
                }
            )
            continue

        kept.append(converted)

    rng = random.Random(seed)
    rng.shuffle(kept)
    if max_external_samples is not None:
        kept = kept[:max_external_samples]

    val_size = max(1, int(len(kept) * val_ratio)) if kept else 0
    external_val = kept[:val_size]
    external_train = kept[val_size:]

    existing_train = load_existing_jsonl(existing_train_jsonl)
    existing_val = load_existing_jsonl(existing_val_jsonl)
    combined_train = existing_train + external_train
    combined_val = existing_val + external_val
    rng.shuffle(combined_train)
    rng.shuffle(combined_val)

    write_jsonl(external_train, output_external_train_jsonl)
    write_jsonl(external_val, output_external_val_jsonl)
    write_jsonl(combined_train, output_train_jsonl)
    write_jsonl(combined_val, output_val_jsonl)

    report = {
        "external_llm_jsonl": str(external_llm_jsonl),
        "benchmark_csv": str(benchmark_csv),
        "raw_external_records": len(raw_records),
        "kept_external_records": len(kept),
        "external_train_records": len(external_train),
        "external_val_records": len(external_val),
        "existing_train_records": len(existing_train),
        "existing_val_records": len(existing_val),
        "combined_train_records": len(combined_train),
        "combined_val_records": len(combined_val),
        "removed_external_records": len(removed),
        "removed_reason_counts": pd.Series([r["reason"] for r in removed]).value_counts().to_dict() if removed else {},
        "removed_examples": removed[:30],
        "similarity_threshold": similarity_threshold,
        "max_external_samples": max_external_samples,
        "output_train_jsonl": str(output_train_jsonl),
        "output_val_jsonl": str(output_val_jsonl),
        "output_external_train_jsonl": str(output_external_train_jsonl),
        "output_external_val_jsonl": str(output_external_val_jsonl),
    }
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external SFT data and combine it with QA auxiliary fine-tune data.")
    parser.add_argument("--external-llm", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--existing-train", required=True)
    parser.add_argument("--existing-val", required=True)
    parser.add_argument("--combined-train-output", required=True)
    parser.add_argument("--combined-val-output", required=True)
    parser.add_argument("--external-train-output", required=True)
    parser.add_argument("--external-val-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--max-external-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_external_and_combined_finetune_dataset(
        external_llm_jsonl=args.external_llm,
        benchmark_csv=args.benchmark,
        existing_train_jsonl=args.existing_train,
        existing_val_jsonl=args.existing_val,
        output_train_jsonl=args.combined_train_output,
        output_val_jsonl=args.combined_val_output,
        output_external_train_jsonl=args.external_train_output,
        output_external_val_jsonl=args.external_val_output,
        report_json=args.report,
        val_ratio=args.val_ratio,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
        max_external_samples=args.max_external_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
