from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .prompting import DEFAULT_PROMPT_VERSION, build_rag_prompt
    from .reranking import CrossEncoderReranker
    from .retrieval import RetrievalEngine
except ImportError:
    from prompting import DEFAULT_PROMPT_VERSION, build_rag_prompt
    from reranking import CrossEncoderReranker
    from retrieval import RetrievalEngine


def load_llm(
    model_name: str,
    device: str | None = None,
    load_in_4bit: bool = True,
    adapter_path: str | Path | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if device == "cuda" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if load_in_4bit and device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if device != "cuda":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 384,
    temperature: float = 0.2,
    top_p: float = 0.9,
    input_max_length: int = 8192,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if isinstance(input_ids, dict) or hasattr(input_ids, "keys"):
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids["input_ids"]
        else:
            attention_mask = None
        if input_ids.shape[1] > input_max_length:
            input_ids = input_ids[:, -input_max_length:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -input_max_length:]
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask if attention_mask is not None else torch.ones_like(input_ids),
        }
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def retrieve_for_question(
    engine: RetrievalEngine,
    question: str,
    retriever_mode: str,
    top_k_context: int,
    candidate_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[dict[str, Any]]:
    if retriever_mode == "dense":
        return engine.dense_search(question, top_k=top_k_context)
    if retriever_mode == "bm25":
        return engine.bm25_search(question, top_k=top_k_context)
    if retriever_mode == "hybrid":
        return engine.hybrid_search(
            question,
            top_k=top_k_context,
            candidate_k=candidate_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    raise ValueError(f"Unsupported retriever_mode: {retriever_mode}")


def retrieve_and_optionally_rerank(
    engine: RetrievalEngine,
    question: str,
    retriever_mode: str,
    top_k_context: int,
    candidate_k: int,
    dense_weight: float,
    bm25_weight: float,
    reranker: CrossEncoderReranker | None = None,
    reranker_batch_size: int = 1,
) -> list[dict[str, Any]]:
    retrieval_top_k = candidate_k if reranker else top_k_context
    candidates = retrieve_for_question(
        engine=engine,
        question=question,
        retriever_mode=retriever_mode,
        top_k_context=retrieval_top_k,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    if not reranker:
        return candidates[:top_k_context]
    return reranker.rerank(
        query=question,
        candidates=candidates,
        text_field="retrieval_text",
        top_k=top_k_context,
        batch_size=reranker_batch_size,
    )


def load_precomputed_retrieval(
    precomputed_retrieval_csv: str | Path | None,
    metadata: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if not precomputed_retrieval_csv:
        return {}
    path = Path(precomputed_retrieval_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing precomputed retrieval CSV: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    article_lookup = {str(row.get("article_key", "")): row for row in metadata}
    lookup: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        question_id = str(row.get("question_id", "")).strip()
        article_keys = [
            item.strip()
            for item in str(row.get("retrieved_article_keys_top10", "")).split(";")
            if item.strip()
        ]
        records: list[dict[str, Any]] = []
        for article_key in article_keys:
            source_record = article_lookup.get(article_key)
            if source_record:
                records.append(dict(source_record))
        if question_id:
            lookup[question_id] = records
    return lookup


def run_rag_generation(
    benchmark_csv: str | Path,
    index_root: str | Path,
    output_predictions_csv: str | Path,
    output_run_config_json: str | Path,
    llm_model: str,
    retriever_mode: str = "dense",
    top_k_context: int = 5,
    candidate_k: int = 30,
    dense_weight: float = 1.0,
    bm25_weight: float = 0.0,
    device: str | None = None,
    max_new_tokens: int = 384,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_context_chars: int = 9000,
    input_max_length: int = 8192,
    limit: int | None = None,
    load_in_4bit: bool = True,
    adapter_path: str | Path | None = None,
    system_name: str = "base_rag",
    reranker_model: str | None = None,
    reranker_batch_size: int = 1,
    precomputed_retrieval_csv: str | Path | None = None,
) -> dict[str, Any]:
    benchmark = pd.read_csv(benchmark_csv, dtype=str, keep_default_na=False)
    if limit:
        benchmark = benchmark.head(limit).copy()

    engine = RetrievalEngine(index_root=index_root, device=device)
    precomputed_retrieval = load_precomputed_retrieval(precomputed_retrieval_csv, engine.metadata)
    reranker = (
        CrossEncoderReranker(model_name=reranker_model, device=device)
        if reranker_model and not precomputed_retrieval
        else None
    )
    tokenizer, model = load_llm(
        llm_model,
        device=device,
        load_in_4bit=load_in_4bit,
        adapter_path=adapter_path,
    )

    rows: list[dict[str, Any]] = []
    for _, item in tqdm(benchmark.iterrows(), total=len(benchmark), desc="Base RAG generation"):
        question = item["question"]
        question_id = str(item.get("question_id", "")).strip()
        if question_id in precomputed_retrieval:
            retrieved = precomputed_retrieval[question_id][:top_k_context]
        else:
            retrieved = retrieve_and_optionally_rerank(
                engine=engine,
                question=question,
                retriever_mode=retriever_mode,
                top_k_context=top_k_context,
                candidate_k=candidate_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
                reranker=reranker,
                reranker_batch_size=reranker_batch_size,
            )
        prompt = build_rag_prompt(question, retrieved, max_context_chars=max_context_chars)
        answer = generate_text(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            input_max_length=input_max_length,
        )
        rows.append(
            {
                "question_id": item.get("question_id", ""),
                "topic": item.get("topic", ""),
                "difficulty": item.get("difficulty", ""),
                "question": question,
                "gold_answer": item.get("gold_answer", ""),
                "gold_doc_keys": item.get("gold_doc_keys", ""),
                "gold_article_keys": item.get("gold_article_keys", ""),
                "gold_law": item.get("gold_law", ""),
                "gold_article_no": item.get("gold_article_no", ""),
                "generated_answer": answer,
                "retrieved_article_keys": "; ".join(str(row.get("article_key", "")) for row in retrieved),
                "retrieved_doc_keys": "; ".join(str(row.get("doc_key", "")) for row in retrieved),
                "retrieved_citations": " | ".join(str(row.get("citation_label", "")) for row in retrieved),
                "prompt_version": DEFAULT_PROMPT_VERSION,
                "llm_model": llm_model,
                "adapter_path": str(adapter_path) if adapter_path else "",
                "system_name": system_name,
                "retriever_mode": retriever_mode,
                "reranker_model": reranker_model or "",
                "precomputed_retrieval_csv": str(precomputed_retrieval_csv) if precomputed_retrieval_csv else "",
                "top_k_context": top_k_context,
            }
        )

    predictions = pd.DataFrame(rows)
    output_predictions_csv = Path(output_predictions_csv)
    output_run_config_json = Path(output_run_config_json)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    output_run_config_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_predictions_csv, index=False, encoding="utf-8-sig")

    run_config = {
        "benchmark_csv": str(benchmark_csv),
        "index_root": str(index_root),
        "output_predictions_csv": str(output_predictions_csv),
        "llm_model": llm_model,
        "adapter_path": str(adapter_path) if adapter_path else "",
        "system_name": system_name,
        "retriever_mode": retriever_mode,
        "reranker_model": reranker_model or "",
        "reranker_batch_size": reranker_batch_size,
        "precomputed_retrieval_csv": str(precomputed_retrieval_csv) if precomputed_retrieval_csv else "",
        "top_k_context": top_k_context,
        "candidate_k": candidate_k,
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "max_context_chars": max_context_chars,
        "input_max_length": input_max_length,
        "limit": limit,
    }
    output_run_config_json.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Base RAG generation on benchmark questions.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--retriever-mode", default="dense", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top-k-context", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--input-max-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--system-name", default="base_rag")
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--precomputed-retrieval-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = run_rag_generation(
        benchmark_csv=args.benchmark,
        index_root=args.index_root,
        output_predictions_csv=args.output_predictions,
        output_run_config_json=args.output_config,
        llm_model=args.llm_model,
        retriever_mode=args.retriever_mode,
        top_k_context=args.top_k_context,
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        input_max_length=args.input_max_length,
        limit=args.limit,
        load_in_4bit=not args.no_4bit,
        adapter_path=args.adapter_path,
        system_name=args.system_name,
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
        precomputed_retrieval_csv=args.precomputed_retrieval_csv,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
