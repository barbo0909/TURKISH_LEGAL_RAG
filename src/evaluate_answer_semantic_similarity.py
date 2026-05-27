from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a_norm * b_norm, axis=1)


def evaluate_semantic_similarity(
    predictions_csv: str | Path,
    output_eval_csv: str | Path,
    output_summary_json: str | Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    predictions_csv = Path(predictions_csv)
    output_eval_csv = Path(output_eval_csv)
    output_summary_json = Path(output_summary_json)

    df = pd.read_csv(predictions_csv, dtype=str, keep_default_na=False)
    if "gold_answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("Input CSV must contain gold_answer and generated_answer columns.")

    model = SentenceTransformer(model_name, device=device)
    gold_embeddings = model.encode(
        df["gold_answer"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    generated_embeddings = model.encode(
        df["generated_answer"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    df["answer_semantic_similarity"] = cosine_rows(generated_embeddings, gold_embeddings)

    output_eval_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_eval_csv, index=False, encoding="utf-8-sig")

    summary = {
        "predictions_csv": str(predictions_csv),
        "eval_csv": str(output_eval_csv),
        "question_count": int(len(df)),
        "embedding_model": model_name,
        "metrics": {
            "answer_semantic_similarity": float(df["answer_semantic_similarity"].mean()),
            "answer_semantic_similarity_median": float(df["answer_semantic_similarity"].median()),
            "answer_semantic_similarity_std": float(df["answer_semantic_similarity"].std(ddof=0)),
        },
    }
    if "topic" in df.columns:
        summary["metrics_by_topic"] = {
            topic: {
                "answer_semantic_similarity": float(group["answer_semantic_similarity"].mean()),
                "count": int(len(group)),
            }
            for topic, group in df.groupby("topic", sort=True)
        }

    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer semantic similarity with sentence embeddings.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_semantic_similarity(
        predictions_csv=args.predictions,
        output_eval_csv=args.output_eval,
        output_summary_json=args.output_summary,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
