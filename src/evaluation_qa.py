from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


TOKEN_RE = re.compile(r"[0-9A-Za-z\u00c7\u011e\u0130\u00d6\u015e\u00dc\u00e7\u011f\u0131\u00f6\u015f\u00fc]+")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = value.replace("ı", "i")
    return " ".join(TOKEN_RE.findall(value))


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, gold: str) -> float:
    pred = tokenize(prediction)
    ref = tokenize(gold)
    if not pred or not ref:
        return float(pred == ref)
    prev = [0] * (len(ref) + 1)
    for token in pred:
        curr = [0]
        for j, ref_token in enumerate(ref, start=1):
            if token == ref_token:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def citation_present(answer: str) -> float:
    text = str(answer).lower()
    has_article = bool(
        re.search(r"\b(m\.|m|madde)\s*[:.]?\s*\d+", text, flags=re.IGNORECASE)
        or re.search(r"\b\d+\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s+madde", text, flags=re.IGNORECASE)
    )
    has_law = "kanun" in text or "anayasa" in text
    has_source_section = "dayanak" in text or "kaynak" in text
    return float(has_article and (has_law or has_source_section))


def citation_gold_match(answer: str, gold_law: str, gold_article_no: str) -> float:
    answer_norm = normalize_text(answer)
    law_tokens = [token for token in tokenize(gold_law) if len(token) > 2]
    article_numbers = re.findall(r"\d+(?:/[A-ZÇĞİÖŞÜ0-9]+)?", str(gold_article_no).upper())
    law_match = any(token in answer_norm for token in law_tokens[:4]) if law_tokens else False
    article_match = any(number.lower() in answer_norm for number in article_numbers)
    return float(law_match and article_match)


def retrieval_gold_available(retrieved_article_keys: str, gold_article_keys: str) -> float:
    retrieved = set(split_semicolon(retrieved_article_keys))
    gold = set(split_semicolon(gold_article_keys))
    return float(bool(retrieved & gold))


def cited_gold_available(answer: str, gold_law: str, gold_article_no: str, retrieved_article_keys: str, gold_article_keys: str) -> float:
    retrieval_hit = retrieval_gold_available(retrieved_article_keys, gold_article_keys)
    return float(bool(retrieval_hit and citation_present(answer) and citation_gold_match(answer, gold_law, gold_article_no)))


def classify_error(row: pd.Series) -> str:
    retrieval_hit = float(row.get("retrieval_gold_available", 0.0))
    citation_has = float(row.get("citation_present", 0.0))
    citation_match = float(row.get("citation_gold_match", 0.0))
    f1 = float(row.get("token_f1", 0.0))

    if retrieval_hit == 0.0:
        return "retrieval_miss"
    if citation_has == 0.0:
        return "missing_citation"
    if citation_match == 0.0:
        return "wrong_or_unsupported_citation"
    if f1 < 0.15:
        return "low_answer_overlap"
    return "acceptable_automatic"


def evaluate_generation_predictions(
    predictions_csv: str | Path,
    output_eval_csv: str | Path,
    output_summary_json: str | Path,
) -> dict[str, Any]:
    df = pd.read_csv(predictions_csv, dtype=str, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        generated = row.get("generated_answer", "")
        gold = row.get("gold_answer", "")
        retrieval_hit = retrieval_gold_available(row.get("retrieved_article_keys", ""), row.get("gold_article_keys", ""))
        citation_has = citation_present(generated)
        citation_match = citation_gold_match(generated, row.get("gold_law", ""), row.get("gold_article_no", ""))
        result = {
            **row.to_dict(),
            "exact_match": exact_match(generated, gold),
            "token_f1": token_f1(generated, gold),
            "rouge_l": rouge_l(generated, gold),
            "retrieval_gold_available": retrieval_hit,
            "citation_present": citation_has,
            "citation_gold_match": citation_match,
            "grounded_citation_score": cited_gold_available(
                generated,
                row.get("gold_law", ""),
                row.get("gold_article_no", ""),
                row.get("retrieved_article_keys", ""),
                row.get("gold_article_keys", ""),
            ),
            "unsupported_or_missing_citation": float(not (citation_has and retrieval_hit)),
        }
        result["error_type_auto"] = classify_error(pd.Series(result))
        rows.append(result)

    eval_df = pd.DataFrame(rows)
    output_eval_csv = Path(output_eval_csv)
    output_summary_json = Path(output_summary_json)
    output_eval_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_eval_csv, index=False, encoding="utf-8-sig")

    metric_cols = [
        "exact_match",
        "token_f1",
        "rouge_l",
        "retrieval_gold_available",
        "citation_present",
        "citation_gold_match",
        "grounded_citation_score",
        "unsupported_or_missing_citation",
    ]
    summary = {
        "predictions_csv": str(predictions_csv),
        "eval_csv": str(output_eval_csv),
        "question_count": int(len(eval_df)),
        "metrics": {col: float(eval_df[col].mean()) for col in metric_cols},
        "metrics_by_topic": {
            topic: group[metric_cols].mean().to_dict()
            for topic, group in eval_df.groupby("topic", sort=True)
        },
        "error_type_counts": eval_df["error_type_auto"].value_counts().to_dict(),
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated RAG answers.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_generation_predictions(
        predictions_csv=args.predictions,
        output_eval_csv=args.output_eval,
        output_summary_json=args.output_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
