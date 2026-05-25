from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


PROMPT_TEMPLATE = """### Talimat
{instruction}

### Girdi
{input}

### Cevap
"""


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


class InstructionDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], tokenizer, max_length: int = 2048) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        prompt = PROMPT_TEMPLATE.format(
            instruction=record.get("instruction", "").strip(),
            input=record.get("input", "").strip(),
        )
        answer = record.get("output", "").strip()
        full_text = prompt + answer + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].shape[0] for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            length = feature["input_ids"].shape[0]
            pad_len = max_len - length
            batch["input_ids"].append(
                torch.cat([feature["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([feature["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
        return {key: torch.stack(value) for key, value in batch.items()}


def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def train_lora(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    model_name: str,
    max_length: int = 2048,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    logging_steps: int = 20,
    eval_steps: int = 100,
    save_steps: int = 100,
    warmup_ratio: float = 0.03,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(train_jsonl, limit=max_train_samples)
    val_records = read_jsonl(val_jsonl, limit=max_val_samples)
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit=use_4bit)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()

    train_dataset = InstructionDataset(train_records, tokenizer=tokenizer, max_length=max_length)
    val_dataset = InstructionDataset(val_records, tokenizer=tokenizer, max_length=max_length)
    collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        eval_strategy="steps",
        save_strategy="steps",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_4bit and torch.cuda.is_available() else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    run_config = {
        "base_model": model_name,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "output_dir": str(output_dir),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "max_length": max_length,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_4bit": use_4bit,
    }
    (output_dir / "finetune_run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the same base LLM.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = train_lora(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_4bit=not args.no_4bit,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
