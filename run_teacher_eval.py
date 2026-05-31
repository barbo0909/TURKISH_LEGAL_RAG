from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.build_index import build_indexes
from src.generation import generate_text, load_llm
from src.ingest_custom_documents import ingest_custom_documents
from src.prompting import build_rag_prompt
from src.reranking import Qwen3CausalRerankerDemo
from src.retrieval import RetrievalEngine

ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-32B"


def free_memory(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def resolve_benchmark_csv(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    csv_files = sorted(path for path in preferred.parent.glob("*.csv") if path.is_file()) if preferred.parent.exists() else []
    if len(csv_files) == 1:
        print("custom_benchmark.csv not found; using the only CSV in the folder:", csv_files[0])
        return csv_files[0]
    if len(csv_files) > 1:
        print("custom_benchmark.csv not found; multiple CSV files exist. Using the first one:", csv_files[0])
        for path in csv_files:
            print("-", path)
        return csv_files[0]
    return preferred


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def has_nonempty_column(df: pd.DataFrame, candidates: list[str]) -> bool:
    for col in candidates:
        if col in df.columns and any(str(value).strip() for value in df[col].astype(str)):
            return True
    return False


def split_keys(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {part.strip() for part in text.replace("|", ";").replace(",", ";").split(";") if part.strip()}


def norm_key(value: Any) -> str:
    value = str(value or "").lower().strip()
    value = value.translate(str.maketrans("çğıöşüıİ", "cgiosuii"))
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def extract_law_numbers(value: Any) -> set[str]:
    return set(re.findall(r"\b\d{3,5}\b", str(value or "")))


def extract_article_numbers(value: Any) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", str(value or ""))}


def result_text(result: dict[str, Any]) -> str:
    return " ".join(str(result.get(k, "")) for k in [
        "article_key", "doc_key", "record_id", "citation_label", "law_name_norm", "law_name_raw",
        "doc_title", "article_no_norm", "article_no_raw", "retrieval_text", "generation_text",
    ])


def gold_identity(item: pd.Series) -> dict[str, Any]:
    values: set[str] = set()
    sources: list[str] = []
    articles: list[str] = []
    law_numbers: set[str] = set()
    article_numbers: set[str] = set()
    doc_keys: set[str] = set()
    article_keys: set[str] = set()

    for col in ["gold_doc_keys", "relevant_doc_keys", "doc_key", "gold_documents", "relevant_documents"]:
        if col in item.index:
            for value in split_keys(item.get(col, "")):
                doc_keys.add(value); values.add(value); values.add(norm_key(value)); law_numbers |= extract_law_numbers(value)
    for col in ["gold_article_keys", "relevant_article_keys", "article_key"]:
        if col in item.index:
            for value in split_keys(item.get(col, "")):
                article_keys.add(value); values.add(value); values.add(norm_key(value)); law_numbers |= extract_law_numbers(value); article_numbers |= extract_article_numbers(value)
    for col in ["gold_source_canonical", "gold_source", "source_url_canonical", "source_url", "gold_law"]:
        if col in item.index:
            value = str(item.get(col, "")).strip()
            if value:
                sources.append(value); values.add(value); values.add(norm_key(value)); law_numbers |= extract_law_numbers(value)
    for col in ["gold_article_normalized", "gold_article", "gold_article_raw", "gold_article_no"]:
        if col in item.index:
            value = str(item.get(col, "")).strip()
            if value:
                articles.append(value); values.add(value); values.add(norm_key(value)); article_numbers |= extract_article_numbers(value)
    for source in sources:
        for article in articles:
            for pair in [f"{source} {article}", f"{source}_{article}", f"{source} m.{article}", f"{source} madde {article}"]:
                values.add(pair); values.add(norm_key(pair))
    return {"values": values, "sources": sources, "articles": articles, "law_numbers": law_numbers,
            "article_numbers": article_numbers, "doc_keys": doc_keys, "article_keys": article_keys}


def is_doc_relevant(result: dict[str, Any], gold: dict[str, Any]) -> int:
    doc_key = str(result.get("doc_key", "")); article_key = str(result.get("article_key", "")); text_norm = norm_key(result_text(result))
    if gold["doc_keys"] and (doc_key in gold["doc_keys"] or any(article_key.startswith(k) for k in gold["doc_keys"])):
        return 1
    if gold["law_numbers"] and any(n in doc_key or n in article_key or n in text_norm for n in gold["law_numbers"]):
        return 1
    if gold["sources"] and any(norm_key(s) and norm_key(s) in text_norm for s in gold["sources"]):
        return 1
    return 0


def is_article_relevant(result: dict[str, Any], gold: dict[str, Any]) -> int:
    article_key = str(result.get("article_key", "")); doc_key = str(result.get("doc_key", ""))
    article_no = str(result.get("article_no_norm", "") or result.get("article_no_raw", "")); text_norm = norm_key(result_text(result))
    if gold["article_keys"] and article_key in gold["article_keys"]:
        return 1
    law_match = True if not gold["law_numbers"] else any(n in article_key or n in doc_key or n in text_norm for n in gold["law_numbers"])
    art_match = True if not gold["article_numbers"] else any(
        f"MADDE_{n}" in article_key or f"MADDE_{n}" in doc_key or f"madde_{n}" in text_norm or article_no.endswith(n)
        for n in gold["article_numbers"]
    )
    return 1 if law_match and art_match else 0


def is_general_relevant(result: dict[str, Any], gold: dict[str, Any]) -> int:
    ids = {str(result.get(k, "")).strip() for k in ["article_key", "doc_key", "record_id", "citation_label"]}
    ids |= {norm_key(v) for v in ids if v}
    if ids & gold["values"]:
        return 1
    if is_article_relevant(result, gold):
        return 1
    if is_doc_relevant(result, gold) and not gold["article_keys"] and not gold["article_numbers"]:
        return 1
    return 0


def dcg_at_k(rels: list[int], k: int) -> float:
    return sum((1.0 / math.log2(i + 1)) for i, rel in enumerate(rels[:k], start=1) if rel)


def ndcg_at_k(rels: list[int], k: int) -> float:
    ideal_count = min(sum(1 for rel in rels if rel), k)
    return 0.0 if ideal_count == 0 else dcg_at_k(rels, k) / dcg_at_k([1] * ideal_count, k)


def metric_values(rels: list[int]) -> dict[str, float]:
    first = next((i for i, rel in enumerate(rels, start=1) if rel), None)
    return {"hit@5": float(any(rels[:5])), "hit@10": float(any(rels[:10])),
            "recall@5": float(any(rels[:5])), "recall@10": float(any(rels[:10])),
            "mrr": 1.0 / first if first else 0.0, "ndcg@5": ndcg_at_k(rels, 5), "ndcg@10": ndcg_at_k(rels, 10)}


def mean_metrics(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    keys = ["hit@5", "hit@10", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"]
    return {f"{prefix}{k}": float(sum(row[k] for row in rows) / len(rows)) for k in keys} if rows else {}


def evaluate_retrieval_like_notebook25(args: argparse.Namespace, benchmark_csv: Path, index_root: Path, output_dir: Path) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    question_col = first_existing_column(benchmark, ["question", "query", "soru"])
    if not question_col:
        raise ValueError(f"Benchmark must include question/query/soru. Columns: {list(benchmark.columns)}")
    print("Pass 1/2: Dense retrieval candidates are being generated...")
    engine = RetrievalEngine(index_root, device=args.device)
    dense_rows = []
    for _, item in benchmark.iterrows():
        question = str(item.get(question_col, "")).strip()
        dense_rows.append({"question": question, "gold": gold_identity(item),
                           "candidates": engine.dense_search(question, top_k=args.candidate_k if not args.no_reranker else args.top_k)})
    del engine; free_memory(args.device)
    reranker = None
    if not args.no_reranker:
        print(f"Pass 2/2: Reranking with {args.reranker_model}...")
        reranker = Qwen3CausalRerankerDemo(args.reranker_model, device=args.device)
    else:
        print("Pass 2/2: Reranker disabled; evaluating dense candidates directly...")
    rows, general_rows, doc_rows, article_rows = [], [], [], []
    for item in dense_rows:
        candidates = item["candidates"]
        if reranker:
            candidates = reranker.rerank(item["question"], candidates, top_k=args.top_k, batch_size=args.reranker_batch_size)
        candidates = candidates[:args.top_k]
        gen = [is_general_relevant(r, item["gold"]) for r in candidates]
        doc = [is_doc_relevant(r, item["gold"]) for r in candidates]
        art = [is_article_relevant(r, item["gold"]) for r in candidates]
        gm, dm, am = metric_values(gen), metric_values(doc), metric_values(art)
        general_rows.append(gm); doc_rows.append(dm); article_rows.append(am)
        rows.append({"question": item["question"], "gold_sources": ";".join(item["gold"]["sources"]),
                     "gold_articles": ";".join(item["gold"]["articles"]),
                     "gold_law_numbers": ";".join(sorted(item["gold"]["law_numbers"])),
                     "gold_article_numbers": ";".join(sorted(item["gold"]["article_numbers"])),
                     "retrieved_keys_top10": ";".join(str(r.get("article_key") or r.get("doc_key") or r.get("record_id") or "") for r in candidates),
                     **gm, **{f"doc_{k}": v for k, v in dm.items()}, **{f"article_{k}": v for k, v in am.items()}})
    if reranker:
        del reranker
    free_memory(args.device)
    df = pd.DataFrame(rows)
    metrics_csv = output_dir / "custom_retrieval_metrics.csv"
    summary_json = output_dir / "custom_retrieval_summary.json"
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    summary = {"benchmark_csv": str(benchmark_csv), "index_root": str(index_root), "embedding_model": args.embedding_model,
               "reranker_model": None if args.no_reranker else args.reranker_model, "question_count": int(len(df)),
               "metrics": {**mean_metrics(general_rows), **mean_metrics(doc_rows, "doc_"), **mean_metrics(article_rows, "article_")}}
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^0-9a-zçğıöşü\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: Any, gold: Any) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref: return 0.0
    counts = {}
    for token in pred: counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in ref:
        if counts.get(token, 0) > 0:
            overlap += 1; counts[token] -= 1
    if overlap == 0: return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: Any, gold: Any) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref: return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, token in enumerate(pred, start=1):
        for j, ref_token in enumerate(ref, start=1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if token == ref_token else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bleu(prediction: Any, gold: Any) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref: return 0.0
    precisions = []
    for n in range(1, 5):
        pred_grams = [tuple(pred[i:i+n]) for i in range(len(pred) - n + 1)]
        ref_grams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
        if not pred_grams:
            precisions.append(0.0); continue
        ref_counts = {}
        for gram in ref_grams: ref_counts[gram] = ref_counts.get(gram, 0) + 1
        overlap = 0
        for gram in pred_grams:
            if ref_counts.get(gram, 0) > 0:
                overlap += 1; ref_counts[gram] -= 1
        precisions.append((overlap + 1) / (len(pred_grams) + 1))
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(len(pred), 1))
    return float(brevity * math.exp(sum(math.log(p) for p in precisions) / 4))


def citation_present(answer: Any, retrieved: list[dict[str, Any]]) -> float:
    answer_text = str(answer or "").lower()
    if "dayanak" in answer_text or "kaynak" in answer_text or "madde" in answer_text:
        return 1.0
    return 1.0 if any(str(r.get("citation_label", "")).lower().strip() in answer_text for r in retrieved if r.get("citation_label")) else 0.0


def classify_error(retrieval_ok: float, citation_ok: float, citation_gold_ok: float, f1: float) -> str:
    if not retrieval_ok: return "retrieval_miss"
    if not citation_ok: return "missing_citation"
    if not citation_gold_ok: return "wrong_or_unsupported_citation"
    if f1 < 0.15: return "low_answer_overlap"
    return "acceptable_automatic"


def evaluate_qa_like_notebook25(args: argparse.Namespace, benchmark_csv: Path, index_root: Path, output_dir: Path) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    question_col = first_existing_column(benchmark, ["question", "query", "soru"])
    answer_col = first_existing_column(benchmark, ["gold_answer", "answer", "cevap"])
    if not question_col or not answer_col:
        raise ValueError(f"QA eval needs question/query and gold_answer/answer columns. Columns: {list(benchmark.columns)}")
    print("Pass 1/3: Retrieving dense candidates for QA benchmark...")
    engine = RetrievalEngine(index_root, device=args.device)
    items = []
    for _, row in benchmark.iterrows():
        question = str(row.get(question_col, "")).strip()
        items.append({"question": question, "gold_answer": str(row.get(answer_col, "")).strip(), "gold": gold_identity(row),
                      "candidates": engine.dense_search(question, top_k=args.candidate_k if not args.no_reranker else args.top_k)})
    del engine; free_memory(args.device)
    reranker = None
    if not args.no_reranker:
        print(f"Pass 2/3: Reranking QA contexts with {args.reranker_model}...")
        reranker = Qwen3CausalRerankerDemo(args.reranker_model, device=args.device)
        for item in items:
            item["retrieved"] = reranker.rerank(item["question"], item["candidates"], top_k=args.top_k, batch_size=args.reranker_batch_size)
    else:
        print("Pass 2/3: Reranker disabled for QA contexts...")
        for item in items:
            item["retrieved"] = item["candidates"][:args.top_k]
    if reranker: del reranker
    for item in items: item.pop("candidates", None)
    free_memory(args.device)
    print(f"Pass 3/3: Generating answers with {args.llm_model}...")
    tokenizer, model = load_llm(args.llm_model, device=args.device, load_in_4bit=args.device == "cuda", adapter_path=args.adapter_path)
    rows = []
    for item in items:
        answer = generate_text(tokenizer, model, build_rag_prompt(item["question"], item["retrieved"], max_context_chars=args.max_context_chars),
                               max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, input_max_length=8192)
        f1 = token_f1(answer, item["gold_answer"])
        retrieval_ok = 1.0 if any(is_general_relevant(r, item["gold"]) for r in item["retrieved"]) else 0.0
        cite_ok = citation_present(answer, item["retrieved"])
        cite_gold = 1.0 if cite_ok and retrieval_ok else 0.0
        grounded = 1.0 if retrieval_ok and cite_ok and cite_gold else 0.0
        rows.append({"question": item["question"], "gold_answer": item["gold_answer"], "generated_answer": answer,
                     "retrieved_keys_top10": ";".join(str(r.get("article_key") or r.get("doc_key") or r.get("record_id") or "") for r in item["retrieved"]),
                     "exact_match": float(normalize_answer(answer) == normalize_answer(item["gold_answer"])),
                     "token_f1": f1, "rouge_l": rouge_l(answer, item["gold_answer"]), "bleu": bleu(answer, item["gold_answer"]),
                     "retrieval_gold_available": retrieval_ok, "citation_present": cite_ok, "citation_gold_match": cite_gold,
                     "grounded_citation_score": grounded, "unsupported_or_missing_citation": 1.0 - grounded,
                     "error_type_auto": classify_error(retrieval_ok, cite_ok, cite_gold, f1)})
    del tokenizer, model; free_memory(args.device)
    df = pd.DataFrame(rows)
    metrics_csv = output_dir / "custom_generation_metrics.csv"
    summary_json = output_dir / "custom_generation_summary.json"
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    metric_cols = ["exact_match", "token_f1", "rouge_l", "bleu", "retrieval_gold_available", "citation_present",
                   "citation_gold_match", "grounded_citation_score", "unsupported_or_missing_citation"]
    summary = {"question_count": int(len(df)), "llm_model": args.llm_model, "reranker_model": None if args.no_reranker else args.reranker_model,
               "metrics": {col: float(df[col].mean()) for col in metric_cols},
               "error_type_counts": df["error_type_auto"].value_counts().to_dict()}
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_metric_block(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}")


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("TEACHER EVALUATION SUMMARY")
    print("=" * 72)
    print("System profile: Qwen3-Embedding-8B top-30 + Qwen3-Reranker-8B top-10 + Qwen3-32B")
    ing = report.get("ingestion_report") or {}
    if ing:
        print("\nCustom ingestion")
        print("- input_files:", ing.get("input_files", "")); print("- output_records:", ing.get("output_records", "")); print("- rejected_chunks:", ing.get("rejected_chunks", ""))
    ret = report.get("retrieval_eval") or {}
    gen = report.get("generation_eval") or {}
    if ret and not ret.get("skipped"): print_metric_block("Retrieval metrics", ret.get("metrics", {}))
    elif ret: print("\nRetrieval metrics skipped:", ret.get("reason", ""))
    if gen and not gen.get("skipped"):
        print_metric_block("QA / citation / grounding metrics", gen.get("metrics", {}))
        if gen.get("error_type_counts"): print_metric_block("Automatic error / hallucination categories", gen["error_type_counts"])
    elif gen: print("\nGeneration metrics skipped:", gen.get("reason", ""))
    rows = []
    for section, payload in [("retrieval", ret), ("qa_citation_grounding", gen)]:
        if payload and not payload.get("skipped"):
            rows += [{"section": section, "metric": k, "value": v} for k, v in payload.get("metrics", {}).items()]
    if rows:
        print("\nCombined metric table")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved outputs")
    print("- combined report:", report.get("report_path", ""))
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher custom-data RAG evaluation, matching notebook 25.")
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-generate-answers", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    benchmark_path = resolve_benchmark_csv(Path(args.benchmark))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    index_root = Path(args.index_root)
    processed_dir = Path(args.processed_dir); processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    custom_csv = processed_dir / "custom_teacher_corpus.csv"
    custom_jsonl = processed_dir / "custom_teacher_corpus.jsonl"
    ingest_report_json = reports_dir / "custom_teacher_ingestion_report.json"
    print("[1/5] Ingesting custom documents...")
    ingestion_report = ingest_custom_documents(input_dir, custom_csv, custom_jsonl, ingest_report_json, text_column=args.text_column)
    print("[2/5] Building custom indexes...")
    manifest = build_indexes(custom_csv, index_root, embedding_model=args.embedding_model, text_field="retrieval_text",
                             batch_size=args.batch_size, device=args.device, build_dense=True, build_bm25=True)
    report: dict[str, Any] = {"input_dir": str(input_dir), "benchmark": str(benchmark_path), "device": args.device,
                              "ingestion_report": ingestion_report, "index_manifest": manifest}
    if not benchmark_path.exists():
        report["benchmark_note"] = "Custom benchmark not found. Ingestion and indexing completed."
        report["retrieval_eval"] = {"skipped": True, "reason": "Custom benchmark not found."}
        report["generation_eval"] = {"skipped": True, "reason": "Custom benchmark not found."}
    else:
        benchmark_df = pd.read_csv(benchmark_path, dtype=str, keep_default_na=False)
        if first_existing_column(benchmark_df, ["question", "query", "soru"]) is None:
            raise ValueError(f"Custom benchmark must include question/query/soru. Columns: {list(benchmark_df.columns)}")
        has_refs = has_nonempty_column(benchmark_df, ["gold_article_keys", "relevant_article_keys", "gold_doc_keys", "relevant_doc_keys", "gold_source", "gold_source_canonical", "gold_article", "gold_article_normalized", "gold_article_raw", "gold_law", "gold_article_no"])
        has_answers = has_nonempty_column(benchmark_df, ["gold_answer", "answer", "cevap"])
        if has_refs:
            print("[3/5] Running retrieval metrics...")
            report["retrieval_eval"] = evaluate_retrieval_like_notebook25(args, benchmark_path, index_root, output_dir)
        else:
            report["retrieval_eval"] = {"skipped": True, "reason": "No gold document/article reference columns found."}
        if not args.no_generate_answers and has_answers:
            print("[4/5] Running QA / citation / grounding metrics...")
            report["generation_eval"] = evaluate_qa_like_notebook25(args, benchmark_path, index_root, output_dir)
        elif args.no_generate_answers:
            report["generation_eval"] = {"skipped": True, "reason": "Answer generation disabled with --no-generate-answers."}
        else:
            report["generation_eval"] = {"skipped": True, "reason": "No gold_answer/answer/cevap column found."}
    print("[5/5] Writing final report...")
    final_report_path = output_dir / "teacher_eval_report.json"
    report["report_path"] = str(final_report_path)
    final_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
