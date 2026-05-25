from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    doc_key: str
    article_key: str
    law_name_norm: str
    article_no_norm: str
    retrieval_text: str
    generation_text: str
    citation_label: str
    source_url: str


def load_retrieval_corpus(path: str, text_field: str = "retrieval_text") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "record_id",
        "doc_key",
        "article_key",
        "law_name_norm",
        "article_no_norm",
        text_field,
        "generation_text",
        "citation_label",
        "source_url",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Corpus is missing required columns: {missing}")
    df = df[df[text_field].astype(str).str.strip().ne("")].copy()
    df = df.reset_index(drop=True)
    df["index_id"] = df.index.astype(int)
    return df


def metadata_records(df: pd.DataFrame, text_field: str = "retrieval_text") -> list[dict[str, str | int]]:
    columns = [
        "index_id",
        "record_id",
        "doc_key",
        "article_key",
        "law_name_norm",
        "article_no_norm",
        text_field,
        "generation_text",
        "citation_label",
        "source_url",
        "quality_flag",
        "body_length",
    ]
    available = [col for col in columns if col in df.columns]
    return df[available].to_dict(orient="records")


def iter_text_batches(texts: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for text in texts:
        batch.append(str(text))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
