from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import CrossEncoder

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - handled in Colab installs
    PeftModel = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
except ImportError:  # pragma: no cover - handled in Colab installs
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"


def is_peft_adapter_dir(model_name: str) -> bool:
    path = Path(str(model_name))
    return path.exists() and (path / "adapter_config.json").exists()


def adapter_base_model_name(adapter_dir: str) -> str:
    config_path = Path(adapter_dir) / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_QWEN3_RERANKER_MODEL
    return str(config.get("base_model_name_or_path") or DEFAULT_QWEN3_RERANKER_MODEL)


def ensure_seq_cls_adapter_config(adapter_dir: str) -> None:
    """Patch older reranker LoRA adapters saved with a CAUSAL_LM task type.

    Early notebook versions accidentally saved Qwen reranker adapters with
    task_type=CAUSAL_LM. Loading that adapter on a sequence-classification
    reranker makes PEFT look for prepare_inputs_for_generation(), which this
    model does not have. The LoRA weights are still attached to the same
    transformer modules, so correcting the adapter metadata is enough for eval.
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("task_type") == "SEQ_CLS":
        return
    backup_path = Path(adapter_dir) / "adapter_config.original_task_type.json"
    if not backup_path.exists():
        backup_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    config["task_type"] = "SEQ_CLS"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def is_qwen_reranker_model(model_name: str) -> bool:
    """Check if the model is a Qwen reranker (vs standard CrossEncoder)."""
    model_name_lower = str(model_name).lower()
    return "qwen" in model_name_lower and "reranker" in model_name_lower


class Qwen3RerankerReranker:
    """Specialized handler for Qwen3-Reranker models."""
    
    def __init__(self, model_name: str, device: str | None = None) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required to load Qwen3 reranker models.")
        
        self.model_name = model_name
        self.device = device
        
        # Check if this is a LoRA adapter
        if is_peft_adapter_dir(model_name):
            ensure_seq_cls_adapter_config(model_name)
            base_model = adapter_base_model_name(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=1, torch_dtype=torch.float16, trust_remote_code=True
            )
            if PeftModel is None:
                raise ImportError("peft is required to load a LoRA reranker adapter.")
            self.model = PeftModel.from_pretrained(self.model, model_name)
            self.model_name = f"{base_model}+adapter:{model_name}"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, torch_dtype=torch.float16, trust_remote_code=True
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        if device:
            self.model.to(device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_field: str = "retrieval_text",
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        
        # Process in batches
        scores = []
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i : i + batch_size]
            texts_a = [query for _ in batch_candidates]
            texts_b = [str(candidate.get(text_field, "")) for candidate in batch_candidates]

            inputs = self.tokenizer(
                texts_a,
                texts_b,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=2048,
            )
            
            # Move to device
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract scores
            batch_scores = outputs.logits.cpu().numpy().flatten().tolist()
            scores.extend(batch_scores)
        
        # Combine candidates with scores
        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            row = dict(candidate)
            row["reranker_score"] = float(score)
            row["retriever_score"] = float(candidate.get("score", 0.0))
            row["score"] = float(score)
            row["retriever"] = f"{candidate.get('retriever', 'retrieval')}+reranker"
            reranked.append(row)
        
        # Sort by reranker score (descending)
        reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        return reranked[:top_k]


class Qwen3CausalRerankerDemo:
    """Teacher-demo reranker for the base Qwen3-Reranker checkpoint.

    This class intentionally does not replace CrossEncoderReranker. It is used
    only by the custom-data notebook when the base Qwen3-Reranker checkpoint
    does not expose a sequence-classification score head.
    """

    def __init__(self, model_name: str = DEFAULT_QWEN3_RERANKER_MODEL, device: str | None = None) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is required to load Qwen3 causal reranker models.")
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.true_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        if device:
            self.model.to(device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_field: str = "retrieval_text",
        top_k: int = 10,
        batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scores: list[float] = []
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i : i + batch_size]
            prompts = [
                self._format_pair(query, str(candidate.get(text_field, "")))
                for candidate in batch_candidates
            ]
            inputs = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=2048,
            )
            if self.device:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            selected = logits[:, [self.false_token_id, self.true_token_id]]
            scores.extend(torch.log_softmax(selected, dim=1)[:, 1].cpu().numpy().flatten().tolist())

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            row = dict(candidate)
            row["reranker_score"] = float(score)
            row["retriever_score"] = float(candidate.get("score", 0.0))
            row["score"] = float(score)
            row["retriever"] = f"{candidate.get('retriever', 'retrieval')}+qwen3_causal_reranker_demo"
            reranked.append(row)
        reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _format_pair(query: str, document: str) -> str:
        instruction = "Given a legal question, judge whether the document is relevant to answer it."
        system = (
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
        )
        user = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )



class CrossEncoderReranker:
    def __new__(cls, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        """Factory method: creates appropriate reranker class based on model type."""
        if cls is not CrossEncoderReranker:
            return super().__new__(cls)

        if is_qwen_reranker_model(model_name) or is_peft_adapter_dir(model_name):
            # Check if LoRA adapter is based on Qwen
            if is_peft_adapter_dir(model_name):
                base_model = adapter_base_model_name(model_name)
                if is_qwen_reranker_model(base_model):
                    instance = object.__new__(Qwen3RerankerReranker)
                    Qwen3RerankerReranker.__init__(instance, model_name, device)
                    return instance
            elif is_qwen_reranker_model(model_name):
                instance = object.__new__(Qwen3RerankerReranker)
                Qwen3RerankerReranker.__init__(instance, model_name, device)
                return instance
        
        instance = object.__new__(StandardCrossEncoderReranker)
        StandardCrossEncoderReranker.__init__(instance, model_name, device)
        return instance
    
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None) -> None:
        # This will be overridden by __new__, but kept for type hints
        pass


class StandardCrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None) -> None:
        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model_name = model_name
        if is_peft_adapter_dir(model_name):
            base_model = adapter_base_model_name(model_name)
            self.model = CrossEncoder(base_model, num_labels=1, **kwargs)
            if PeftModel is None:
                raise ImportError("peft is required to load a LoRA reranker adapter.")
            self.model.model = PeftModel.from_pretrained(self.model.model, model_name)
            self.model.model.eval()
            self.model_name = f"{base_model}+adapter:{model_name}"
        else:
            self.model = CrossEncoder(model_name, **kwargs)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_field: str = "retrieval_text",
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, str(candidate.get(text_field, ""))) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            row = dict(candidate)
            row["reranker_score"] = float(score)
            row["retriever_score"] = float(candidate.get("score", 0.0))
            row["score"] = float(score)
            row["retriever"] = f"{candidate.get('retriever', 'retrieval')}+reranker"
            reranked.append(row)
        reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        return reranked[:top_k]
