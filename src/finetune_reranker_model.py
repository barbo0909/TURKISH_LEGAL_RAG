from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - handled in notebooks/Colab installs
    LoraConfig = None
    TaskType = None
    get_peft_model = None


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


def to_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        passage = str(row.get("candidate_passage", "")).strip()
        if not query or not passage:
            continue
        pairs.append({"query": query, "passage": passage, "label": float(row.get("label", 0))})
    return pairs


def finetune_reranker_model(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    model_name: str = "Qwen/Qwen3-Reranker-8B",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    batch_size: int = 1,
    epochs: int = 1,
    warmup_ratio: float = 0.1,
    learning_rate: float = 1e-5,
    device: str | None = None,
    max_length: int | None = 2048,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_jsonl, limit=max_train_samples)
    val_rows = read_jsonl(val_jsonl, limit=max_val_samples)
    train_pairs = to_pairs(train_rows)
    val_pairs = to_pairs(val_rows)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = int(max_length or 2048)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        if LoraConfig is None or TaskType is None or get_peft_model is None:
            raise ImportError("peft is required for LoRA reranker fine-tuning. Install peft first.")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    model.to(device)
    model.train()

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [row["query"] for row in batch],
            [row["passage"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([float(row["label"]) for row in batch], dtype=torch.float32)
        return encoded

    train_loader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)
    total_steps = max(1, (len(train_loader) * epochs) // max(1, gradient_accumulation_steps))
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = BCEWithLogitsLoss()

    global_step = 0
    loss_history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"Reranker fine-tune epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits.view(-1).float(), labels)
            (loss / max(1, gradient_accumulation_steps)).backward()

            if step % max(1, gradient_accumulation_steps) == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                raw_loss = float(loss.detach().cpu().item())
                progress.set_postfix({"loss": f"{raw_loss:.4f}", "step": global_step})
                if global_step % 100 == 0 or step == len(train_loader):
                    loss_history.append({"step": float(global_step), "training_loss": raw_loss})

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    run_config = {
        "model_name": model_name,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "output_dir": str(output_dir),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "train_examples": len(train_pairs),
        "val_examples": len(val_pairs),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "use_lora": use_lora,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "loss_history": loss_history,
    }
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    all_params = sum(param.numel() for param in model.parameters())
    run_config["trainable_params"] = int(trainable_params)
    run_config["all_params"] = int(all_params)
    run_config["trainable_percent"] = float(100 * trainable_params / all_params) if all_params else 0.0
    (output_dir / "reranker_finetune_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if use_lora:
        adapter_config = output_dir / "adapter_config.json"
        adapter_model_safe = output_dir / "adapter_model.safetensors"
        adapter_model_bin = output_dir / "adapter_model.bin"
        if not adapter_config.exists() or not (adapter_model_safe.exists() or adapter_model_bin.exists()):
            raise RuntimeError("LoRA adapter was not saved correctly.")
    return run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen reranker with legal pairs.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = finetune_reranker_model(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        max_length=args.max_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
