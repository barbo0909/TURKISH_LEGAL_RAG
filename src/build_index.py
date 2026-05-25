from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

try:
    from .chunking import iter_text_batches, load_retrieval_corpus, metadata_records
    from .embeddings import encode_passages, load_embedding_model, save_numpy
except ImportError:
    from chunking import iter_text_batches, load_retrieval_corpus, metadata_records
    from embeddings import encode_passages, load_embedding_model, save_numpy


TOKEN_RE = re.compile(r"[0-9A-Za-z\u00c7\u011e\u0130\u00d6\u015e\u00dc\u00e7\u011f\u0131\u00f6\u015f\u00fc]+")


def tokenize_tr(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def build_dense_index(
    texts: list[str],
    model_name: str,
    output_dir: Path,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Any]:
    model = load_embedding_model(model_name, device=device)
    vectors: list[np.ndarray] = []
    total_batches = math.ceil(len(texts) / batch_size) if texts else 0
    for batch in tqdm(iter_text_batches(texts, batch_size), total=total_batches, desc="Encoding corpus"):
        emb = encode_passages(model, batch, model_name=model_name, batch_size=batch_size)
        vectors.append(emb.astype("float32"))

    embeddings = np.vstack(vectors).astype("float32") if vectors else np.zeros((0, 0), dtype="float32")
    if embeddings.size == 0:
        raise ValueError("No embeddings were created. Check corpus text field.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    save_numpy(output_dir / "embeddings.npy", embeddings)

    return {
        "dense_index_path": str(output_dir / "faiss.index"),
        "embeddings_path": str(output_dir / "embeddings.npy"),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_count": int(embeddings.shape[0]),
    }


def build_bm25_index(texts: list[str], output_dir: Path) -> dict[str, Any]:
    tokenized = [tokenize_tr(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized}, f)
    return {
        "bm25_index_path": str(output_dir / "bm25.pkl"),
        "bm25_doc_count": len(tokenized),
    }


def write_metadata(records: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metadata_path


def build_indexes(
    corpus_path: str | Path,
    index_root: str | Path,
    embedding_model: str,
    text_field: str = "retrieval_text",
    batch_size: int = 32,
    device: str | None = None,
    build_dense: bool = True,
    build_bm25: bool = True,
) -> dict[str, Any]:
    index_root = Path(index_root)
    df = load_retrieval_corpus(str(corpus_path), text_field=text_field)
    texts = df[text_field].astype(str).tolist()
    records = metadata_records(df, text_field=text_field)

    manifest: dict[str, Any] = {
        "corpus_path": str(corpus_path),
        "text_field": text_field,
        "record_count": int(len(df)),
        "embedding_model": embedding_model,
        "batch_size": batch_size,
        "dense_enabled": build_dense,
        "bm25_enabled": build_bm25,
    }

    metadata_path = write_metadata(records, index_root / "metadata")
    manifest["metadata_path"] = str(metadata_path)

    if build_dense:
        manifest.update(
            build_dense_index(
                texts=texts,
                model_name=embedding_model,
                output_dir=index_root / "dense",
                batch_size=batch_size,
                device=device,
            )
        )

    if build_bm25:
        manifest.update(build_bm25_index(texts=texts, output_dir=index_root / "bm25"))

    manifest_path = index_root / "index_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense and BM25 retrieval indexes.")
    parser.add_argument("--corpus", required=True, help="Normalized corpus CSV path")
    parser.add_argument("--index-root", required=True, help="Output index root")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--text-field", default="retrieval_text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, e.g. cuda or cpu")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_indexes(
        corpus_path=args.corpus,
        index_root=args.index_root,
        embedding_model=args.embedding_model,
        text_field=args.text_field,
        batch_size=args.batch_size,
        device=args.device,
        build_dense=not args.no_dense,
        build_bm25=not args.no_bm25,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
