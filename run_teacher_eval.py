from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.build_index import build_indexes
from src.evaluation_qa import evaluate_generation_predictions
from src.evaluation_reranker import evaluate_reranker
from src.evaluation_retrieval import evaluate_retrieval
from src.generation import run_rag_generation
from src.ingest_custom_documents import ingest_custom_documents


ROOT = Path(__file__).resolve().parent

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-32B"


def choose_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def extract_first_number(value: str) -> str:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def extract_article_no(value: str) -> str:
    match = re.search(r"\d+(?:/[A-Za-z0-9]+)?", str(value))
    return match.group(0).upper() if match else ""


def corpus_gold_key_maps(corpus_csv: Path | None) -> dict[str, dict[str, str]]:
    if not corpus_csv or not corpus_csv.exists():
        return {"doc_by_law_no": {}, "article_by_law_article": {}}
    corpus = pd.read_csv(corpus_csv, dtype=str, keep_default_na=False)
    doc_by_law_no: dict[str, str] = {}
    article_by_law_article: dict[str, str] = {}
    for _, row in corpus.iterrows():
        source_url = str(row.get("source_url", ""))
        law_no_match = re.search(r"MevzuatNo=(\d+)", source_url)
        law_no = law_no_match.group(1) if law_no_match else ""
        if not law_no:
            continue
        doc_key = str(row.get("doc_key", "")).strip()
        article_key = str(row.get("article_key", "")).strip()
        article_no = extract_article_no(str(row.get("article_no_norm", "")))
        if doc_key:
            doc_by_law_no.setdefault(law_no, doc_key)
        if article_key and article_no:
            article_by_law_article.setdefault(f"{law_no}:{article_no}", article_key)
    return {"doc_by_law_no": doc_by_law_no, "article_by_law_article": article_by_law_article}


def normalize_benchmark_csv(input_csv: Path, output_csv: Path, corpus_csv: Path | None = None) -> dict[str, Any]:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    columns = list(df.columns)

    question_col = choose_existing_column(columns, ["question", "soru", "query"])
    if not question_col:
        raise ValueError(
            f"Custom benchmark must include a question column. Available columns: {columns}"
        )

    gold_answer_col = choose_existing_column(columns, ["gold_answer", "answer", "expected_answer"])
    gold_doc_col = choose_existing_column(
        columns,
        [
            "gold_doc_keys",
            "gold_docs",
            "doc_keys",
            "gold_source",
            "gold_source_canonical",
            "source",
            "source_title",
        ],
    )
    gold_article_col = choose_existing_column(
        columns,
        [
            "gold_article_keys",
            "gold_articles",
            "article_keys",
            "madde_keys",
            "gold_article",
            "gold_article_normalized",
            "article",
            "article_no",
        ],
    )
    topic_col = choose_existing_column(columns, ["topic", "domain", "category"])
    difficulty_col = choose_existing_column(columns, ["difficulty", "level"])
    question_id_col = choose_existing_column(columns, ["question_id", "id"])
    key_maps = corpus_gold_key_maps(corpus_csv)

    gold_doc_values = []
    gold_article_values = []
    mapped_doc_count = 0
    mapped_article_count = 0
    for _, row in df.iterrows():
        raw_doc = str(row.get(gold_doc_col, "")) if gold_doc_col else ""
        raw_article = str(row.get(gold_article_col, "")) if gold_article_col else ""
        law_no = extract_first_number(raw_doc)
        article_no = extract_article_no(raw_article)
        mapped_doc = key_maps["doc_by_law_no"].get(law_no, raw_doc)
        mapped_article = key_maps["article_by_law_article"].get(f"{law_no}:{article_no}", raw_article)
        if mapped_doc != raw_doc:
            mapped_doc_count += 1
        if mapped_article != raw_article:
            mapped_article_count += 1
        gold_doc_values.append(mapped_doc)
        gold_article_values.append(mapped_article)

    normalized = pd.DataFrame(
        {
            "question_id": (
                df[question_id_col].astype(str)
                if question_id_col
                else [f"custom_{idx + 1:04d}" for idx in range(len(df))]
            ),
            "topic": df[topic_col].astype(str) if topic_col else "",
            "difficulty": df[difficulty_col].astype(str) if difficulty_col else "",
            "question": df[question_col].astype(str),
            "gold_answer": df[gold_answer_col].astype(str) if gold_answer_col else "",
            "gold_doc_keys": gold_doc_values if gold_doc_col else "",
            "gold_article_keys": gold_article_values if gold_article_col else "",
            "gold_law": "",
            "gold_article_no": "",
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(normalized)),
        "question_column": question_col,
        "gold_answer_column": gold_answer_col or "",
        "gold_doc_keys_column": gold_doc_col or "",
        "gold_article_keys_column": gold_article_col or "",
        "mapped_doc_keys": mapped_doc_count,
        "mapped_article_keys": mapped_article_count,
        "topic_column": topic_col or "",
        "difficulty_column": difficulty_col or "",
        "question_id_column": question_id_col or "",
    }


def has_any_nonempty(series: pd.Series) -> bool:
    return any(str(value).strip() for value in series.astype(str))


def print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.6f}")
        else:
            print(f"- {key}: {value}")


def print_final_console_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("TEACHER EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Device: {report.get('device', '')}")
    print("System profile: final best retrieval stack + Qwen3-32B generation")

    ingestion = report.get("ingestion_report") or {}
    if ingestion:
        print("\nCustom ingestion")
        print(f"- input_files: {ingestion.get('input_files', '')}")
        print(f"- output_records: {ingestion.get('output_records', '')}")
        print(f"- rejected_chunks: {ingestion.get('rejected_chunks', '')}")

    benchmark = report.get("normalized_benchmark") or {}
    if benchmark:
        print("\nCustom benchmark")
        print(f"- rows: {benchmark.get('rows', '')}")
        print(f"- question_column: {benchmark.get('question_column', '')}")
        if benchmark.get("gold_answer_column"):
            print(f"- gold_answer_column: {benchmark.get('gold_answer_column')}")
        if benchmark.get("gold_doc_keys_column"):
            print(f"- gold_doc_keys_column: {benchmark.get('gold_doc_keys_column')}")
        if benchmark.get("gold_article_keys_column"):
            print(f"- gold_article_keys_column: {benchmark.get('gold_article_keys_column')}")

    retrieval = report.get("retrieval_eval") or {}
    if retrieval and not retrieval.get("skipped"):
        print_metric_block("Retrieval metrics", retrieval.get("metrics", {}))
    elif retrieval:
        print(f"\nRetrieval metrics skipped: {retrieval.get('reason', '')}")

    generation = report.get("generation_eval") or {}
    if generation and not generation.get("skipped"):
        print_metric_block("Generation / grounding metrics", generation.get("metrics", {}))
        error_counts = generation.get("error_type_counts", {})
        if error_counts:
            print_metric_block("Automatic error / hallucination categories", error_counts)
    elif generation:
        print(f"\nGeneration metrics skipped: {generation.get('reason', '')}")

    print("\nSaved outputs")
    print(f"- combined report: {report.get('report_path', '')}")
    print("=" * 72)


def resolve_benchmark_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.name == "custom_benchmark.csv" and path.parent.exists():
        csv_files = sorted(
            candidate
            for candidate in path.parent.glob("*.csv")
            if candidate.name.lower() != "custom_benchmark.csv"
        )
        if len(csv_files) == 1:
            print(f"Custom benchmark file not found at {path}.")
            print(f"Using detected benchmark CSV instead: {csv_files[0]}")
            return csv_files[0]
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-entry custom teacher evaluation pipeline for the Turkish Legal RAG project."
    )
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "custom_docs"))
    parser.add_argument("--benchmark", default=str(ROOT / "data" / "custom_benchmark" / "custom_benchmark.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "teacher_eval"))
    parser.add_argument("--index-root", default=str(ROOT / "indexes" / "custom_teacher_eval"))
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-generate-answers", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    benchmark_path = resolve_benchmark_path(Path(args.benchmark))
    output_dir = Path(args.output_dir)
    index_root = Path(args.index_root)
    processed_dir = Path(args.processed_dir)
    reports_dir = Path(args.reports_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    custom_csv = processed_dir / "custom_teacher_corpus.csv"
    custom_jsonl = processed_dir / "custom_teacher_corpus.jsonl"
    ingest_report_json = reports_dir / "custom_teacher_ingestion_report.json"

    print("[1/5] Ingesting custom documents...")
    try:
        ingestion_report = ingest_custom_documents(
            input_dir=input_dir,
            output_csv=custom_csv,
            output_jsonl=custom_jsonl,
            report_json=ingest_report_json,
            text_column=args.text_column,
        )
    except ValueError as exc:
        print("\nNo usable custom documents were found.")
        print(f"Input folder: {input_dir}")
        print("Please add at least one .txt, .csv, or .jsonl source document to data/custom_docs/.")
        print("The placeholder file put_custom_documents_here.txt is intentionally ignored.")
        print(f"Original error: {exc}")
        return

    print("[2/5] Building indexes...")
    manifest = build_indexes(
        corpus_path=custom_csv,
        index_root=index_root,
        embedding_model=args.embedding_model,
        text_field="retrieval_text",
        batch_size=args.batch_size if args.device == "cuda" else min(args.batch_size, 8),
        device=args.device,
        build_dense=True,
        build_bm25=True,
    )

    final_report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "benchmark": str(benchmark_path),
        "device": args.device,
        "ingestion_report": ingestion_report,
        "index_manifest": manifest,
        "normalized_benchmark": None,
        "retrieval_eval": None,
        "generation_eval": None,
    }

    if not benchmark_path.exists():
        final_report["benchmark_note"] = "Custom benchmark not found. Ingestion and indexing completed."
        final_report_path = output_dir / "teacher_eval_report.json"
        final_report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(final_report, ensure_ascii=False, indent=2))
        return

    print("[3/5] Normalizing custom benchmark...")
    normalized_benchmark_csv = output_dir / "custom_benchmark_normalized.csv"
    benchmark_info = normalize_benchmark_csv(benchmark_path, normalized_benchmark_csv, corpus_csv=custom_csv)
    final_report["normalized_benchmark"] = benchmark_info

    benchmark_df = pd.read_csv(normalized_benchmark_csv, dtype=str, keep_default_na=False)
    has_gold_docs = has_any_nonempty(benchmark_df["gold_doc_keys"])
    has_gold_articles = has_any_nonempty(benchmark_df["gold_article_keys"])
    has_gold_answers = has_any_nonempty(benchmark_df["gold_answer"])

    use_reranker = not args.no_reranker
    generate_answers = not args.no_generate_answers

    if has_gold_docs or has_gold_articles:
        print("[4/5] Running retrieval evaluation...")
        retrieval_predictions = output_dir / "custom_retrieval_predictions.csv"
        retrieval_summary = output_dir / "custom_retrieval_summary.json"
        if use_reranker:
            retrieval_result = evaluate_reranker(
                benchmark_csv=normalized_benchmark_csv,
                index_root=index_root,
                output_predictions_csv=retrieval_predictions,
                output_summary_json=retrieval_summary,
                candidate_mode="dense",
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                reranker_model=args.reranker_model,
                batch_size=args.reranker_batch_size,
                device=args.device,
            )
        else:
            retrieval_result = evaluate_retrieval(
                benchmark_csv=normalized_benchmark_csv,
                index_root=index_root,
                output_predictions_csv=retrieval_predictions,
                output_summary_json=retrieval_summary,
                mode="dense",
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                device=args.device,
            )
        final_report["retrieval_eval"] = retrieval_result
    else:
        final_report["retrieval_eval"] = {
            "skipped": True,
            "reason": "No gold_doc_keys or gold_article_keys columns/content found in custom benchmark.",
        }

    if generate_answers and has_gold_answers:
        print("[5/5] Running answer generation and QA evaluation...")
        generation_predictions = output_dir / "custom_generation_predictions.csv"
        generation_config = output_dir / "custom_generation_run_config.json"
        generation_eval_csv = output_dir / "custom_generation_eval.csv"
        generation_summary = output_dir / "custom_generation_summary.json"

        run_rag_generation(
            benchmark_csv=normalized_benchmark_csv,
            index_root=index_root,
            output_predictions_csv=generation_predictions,
            output_run_config_json=generation_config,
            llm_model=args.llm_model,
            retriever_mode="dense",
            top_k_context=args.top_k,
            candidate_k=args.candidate_k,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_context_chars=args.max_context_chars,
            load_in_4bit=args.device == "cuda",
            adapter_path=args.adapter_path,
            system_name="teacher_custom_eval",
            reranker_model=args.reranker_model if use_reranker else None,
            reranker_batch_size=args.reranker_batch_size,
        )
        generation_result = evaluate_generation_predictions(
            predictions_csv=generation_predictions,
            output_eval_csv=generation_eval_csv,
            output_summary_json=generation_summary,
        )
        final_report["generation_eval"] = generation_result
    elif generate_answers and not has_gold_answers:
        final_report["generation_eval"] = {
            "skipped": True,
            "reason": "Answer generation was requested, but the custom benchmark does not contain a gold_answer column/content.",
        }
    else:
        final_report["generation_eval"] = {
            "skipped": True,
            "reason": "Answer generation not requested. Use --generate-answers to run the full QA pipeline.",
        }

    final_report_path = output_dir / "teacher_eval_report.json"
    final_report["report_path"] = str(final_report_path)
    final_report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_final_console_summary(final_report)


if __name__ == "__main__":
    main()
