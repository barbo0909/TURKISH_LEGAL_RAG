from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
import torch

try:
    from peft import LoraConfig, TaskType
except ImportError:  # pragma: no cover - handled in notebooks/Colab installs
    LoraConfig = None
    TaskType = None


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def to_examples(rows: list[dict[str, Any]], use_hard_negative: bool = True) -> list[InputExample]:
    examples: list[InputExample] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive_passage", "")).strip()
        negative = str(row.get("negative_passage", "")).strip()
        if not query or not positive:
            continue
        texts = [query, positive]
        if use_hard_negative and negative:
            texts.append(negative)
        examples.append(InputExample(texts=texts))
    return examples


def finetune_embedding_model(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    model_name: str = "intfloat/multilingual-e5-base",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    batch_size: int = 8,
    epochs: int = 1,
    warmup_ratio: float = 0.1,
    use_hard_negative: bool = True,
    device: str | None = None,
    max_seq_length: int | None = None,
    use_lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    torch_dtype: str = "bfloat16",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16 if torch_dtype == "float16" else None
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    tokenizer_kwargs = {"padding_side": "left"} if "qwen3" in model_name.lower() else {}

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs or None,
        tokenizer_kwargs=tokenizer_kwargs or None,
    )
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)

    if use_lora:
        if LoraConfig is None or TaskType is None:
            raise ImportError("peft is required for LoRA embedding fine-tuning. Install peft first.")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model.add_adapter(peft_config)
        try:
            transformer = model._first_module()
            auto_model = getattr(transformer, "auto_model", None)
            if auto_model is not None:
                auto_model.gradient_checkpointing_enable()
        except Exception:
            pass
    train_rows = read_jsonl(train_jsonl, limit=max_train_samples)
    val_rows = read_jsonl(val_jsonl, limit=max_val_samples)
    train_examples = to_examples(train_rows, use_hard_negative=use_hard_negative)
    val_examples = to_examples(val_rows, use_hard_negative=use_hard_negative)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=str(output_dir),
        save_best_model=True,
    )

    run_config = {
        "model_name": model_name,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "output_dir": str(output_dir),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "use_hard_negative": use_hard_negative,
        "max_seq_length": max_seq_length,
        "use_lora": use_lora,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "torch_dtype": torch_dtype,
    }
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    all_params = sum(param.numel() for param in model.parameters())
    run_config["trainable_params"] = int(trainable_params)
    run_config["all_params"] = int(all_params)
    run_config["trainable_percent"] = float(100 * trainable_params / all_params) if all_params else 0.0
    (output_dir / "embedding_finetune_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an embedding model with contrastive legal pairs.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-hard-negative", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = finetune_embedding_model(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        use_hard_negative=not args.no_hard_negative,
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        torch_dtype=args.torch_dtype,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
