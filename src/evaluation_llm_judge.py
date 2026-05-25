from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional for judge-only eval
    PeftModel = None


JUDGE_INSTRUCTION = """Sen Turk hukuk RAG ciktilari icin tarafsiz bir degerlendiricisin.
Sana soru, altin cevap, getirilen kaynak/citation bilgisi ve model cevabi verilecek.
Sadece verilen bilgilere gore puanla. JSON disinda metin yazma.

Puan araliklari:
- correctness: 1-5, cevap altin cevabin hukuki icerigini ne kadar karsiliyor?
- faithfulness: 1-5, cevap getirilen baglama ne kadar sadik?
- citation_support: 1-5, citation/kaynak kullanimi ne kadar destekli ve dogru?
- hallucination_risk: 1-5, 1 dusuk risk, 5 yuksek risk.

JSON formati:
{"correctness": 1, "faithfulness": 1, "citation_support": 1, "hallucination_risk": 1, "rationale": "kisa gerekce"}
"""


def build_judge_prompt(row: pd.Series) -> str:
    return f"""{JUDGE_INSTRUCTION}

Soru:
{row.get('question', '')}

Altin cevap:
{row.get('gold_answer', '')}

Altin madde anahtarlari:
{row.get('gold_article_keys', '')}

Getirilen madde anahtarlari:
{row.get('retrieved_article_keys', '')}

Getirilen citation bilgisi:
{row.get('retrieved_citations', '')}

Model cevabi:
{row.get('generated_answer', '')}

JSON:
"""


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text[:200]}")
    return json.loads(match.group(0))


def clamp_score(value: Any, default: float = 1.0) -> float:
    try:
        score = float(value)
    except Exception:
        score = default
    return max(1.0, min(5.0, score))


def load_judge_llm(
    model_name: str,
    device: str | None = None,
    load_in_4bit: bool = True,
    adapter_path: str | Path | None = None,
):
    """Load a judge LLM without importing retrieval modules or FAISS."""
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
        if PeftModel is None:
            raise ImportError("peft is required to load an adapter_path.")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if device != "cuda":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_judge_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256,
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
        if hasattr(input_ids, "data") and "input_ids" in input_ids:
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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_with_llm_judge(
    predictions_csv: str | Path,
    output_csv: str | Path,
    output_summary_json: str | Path,
    judge_model: str,
    device: str | None = None,
    limit: int | None = None,
    load_in_4bit: bool = True,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    df = pd.read_csv(predictions_csv, dtype=str, keep_default_na=False)
    if limit:
        df = df.head(limit).copy()

    tokenizer, model = load_judge_llm(judge_model, device=device, load_in_4bit=load_in_4bit)
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM judge"):
        prompt = build_judge_prompt(row)
        raw = generate_judge_text(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_max_length=8192,
        )
        try:
            parsed = extract_json(raw)
        except Exception as exc:
            parsed = {
                "correctness": 1,
                "faithfulness": 1,
                "citation_support": 1,
                "hallucination_risk": 5,
                "rationale": f"parse_error: {exc}",
            }
        rows.append(
            {
                **row.to_dict(),
                "judge_model": judge_model,
                "judge_raw": raw,
                "judge_correctness": clamp_score(parsed.get("correctness")),
                "judge_faithfulness": clamp_score(parsed.get("faithfulness")),
                "judge_citation_support": clamp_score(parsed.get("citation_support")),
                "judge_hallucination_risk": clamp_score(parsed.get("hallucination_risk"), default=5.0),
                "judge_rationale": str(parsed.get("rationale", "")),
            }
        )

    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_summary_json = Path(output_summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    metric_cols = [
        "judge_correctness",
        "judge_faithfulness",
        "judge_citation_support",
        "judge_hallucination_risk",
    ]
    summary = {
        "predictions_csv": str(predictions_csv),
        "output_csv": str(output_csv),
        "judge_model": judge_model,
        "question_count": int(len(out)),
        "limit": limit,
        "metrics": {col: float(out[col].mean()) for col in metric_cols},
        "metrics_by_topic": {
            topic: group[metric_cols].mean().to_dict()
            for topic, group in out.groupby("topic", sort=True)
        } if "topic" in out.columns else {},
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional LLM judge evaluation for RAG outputs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_with_llm_judge(
        predictions_csv=args.predictions,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary,
        judge_model=args.judge_model,
        device=args.device,
        limit=args.limit,
        load_in_4bit=not args.no_4bit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
