from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


E5_PREFIXES = ("intfloat/multilingual-e5", "intfloat/e5")
QWEN3_EMBEDDING_MARKER = "qwen3-embedding"


def is_e5_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return lowered.startswith(E5_PREFIXES) or "/e5-" in lowered


def is_qwen3_embedding_model(model_name: str) -> bool:
    return QWEN3_EMBEDDING_MARKER in model_name.lower()


def format_passages(texts: list[str], model_name: str) -> list[str]:
    if is_e5_model(model_name):
        return [f"passage: {text}" for text in texts]
    return texts


def format_queries(texts: list[str], model_name: str) -> list[str]:
    if is_e5_model(model_name):
        return [f"query: {text}" for text in texts]
    return texts


def load_embedding_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def encode_passages(
    model: SentenceTransformer,
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    return model.encode(
        format_passages(texts, model_name),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def encode_queries(
    model: SentenceTransformer,
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    if is_qwen3_embedding_model(model_name):
        try:
            return model.encode(
                texts,
                prompt_name="query",
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except TypeError:
            pass

    return model.encode(
        format_queries(texts, model_name),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def save_numpy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
