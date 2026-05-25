from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_SUFFIXES = {".txt", ".csv", ".jsonl"}
PLACEHOLDER_FILENAMES = {
    "put_custom_documents_here.txt",
    "put_custom_benchmark_here.txt",
}
TEXT_COLUMN_CANDIDATES = [
    "text",
    "chunk_text",
    "content",
    "body",
    "article_body",
    "retrieval_text",
    "generation_text",
    "metin",
    "icerik",
]
TITLE_COLUMN_CANDIDATES = ["title", "doc_title", "law_name", "law_name_norm", "başlık", "baslik"]
SECTION_COLUMN_CANDIDATES = ["section", "section_title", "article_no", "article_no_norm", "madde", "başlık", "baslik"]


def clean_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\ufeff", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = value.translate(str.maketrans("çğıöşüİ", "cgiosui"))
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "document"


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def choose_text_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested text column not found: {explicit}. Available columns: {list(df.columns)}")
        return explicit
    chosen = choose_column(list(df.columns), TEXT_COLUMN_CANDIDATES)
    if chosen:
        return chosen
    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    if not object_columns:
        raise ValueError("Could not infer a text column from custom CSV/JSONL.")
    return max(object_columns, key=lambda col: df[col].astype(str).str.len().mean())


def chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 180) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap_chars)
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            prefix = current[-overlap_chars:].strip() if overlap_chars and current else ""
            current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_rows_from_txt(path: Path) -> list[dict[str, Any]]:
    return [{"doc_title": path.stem, "section_title": "", "text": path.read_text(encoding="utf-8", errors="ignore")}]


def source_rows_from_csv(path: Path, text_column: str | None = None) -> list[dict[str, Any]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    text_col = choose_text_column(df, explicit=text_column)
    title_col = choose_column(list(df.columns), TITLE_COLUMN_CANDIDATES)
    section_col = choose_column(list(df.columns), SECTION_COLUMN_CANDIDATES)
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "doc_title": clean_text(row.get(title_col, "")) if title_col else path.stem,
                "section_title": clean_text(row.get(section_col, "")) if section_col else f"row_{idx + 1}",
                "text": clean_text(row.get(text_col, "")),
                "metadata": {column: row.get(column, "") for column in df.columns if column != text_col},
            }
        )
    return rows


def source_rows_from_jsonl(path: Path, text_column: str | None = None) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(path)
    if not raw_rows:
        return []
    df = pd.DataFrame(raw_rows).fillna("")
    text_col = choose_text_column(df, explicit=text_column)
    title_col = choose_column(list(df.columns), TITLE_COLUMN_CANDIDATES)
    section_col = choose_column(list(df.columns), SECTION_COLUMN_CANDIDATES)
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "doc_title": clean_text(row.get(title_col, "")) if title_col else path.stem,
                "section_title": clean_text(row.get(section_col, "")) if section_col else f"row_{idx + 1}",
                "text": clean_text(row.get(text_col, "")),
                "metadata": {column: row.get(column, "") for column in df.columns if column != text_col},
            }
        )
    return rows


def read_source_rows(path: Path, text_column: str | None = None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".txt":
        return source_rows_from_txt(path)
    if path.suffix.lower() == ".csv":
        return source_rows_from_csv(path, text_column=text_column)
    if path.suffix.lower() == ".jsonl":
        return source_rows_from_jsonl(path, text_column=text_column)
    return []


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name.lower() not in PLACEHOLDER_FILENAMES
    )


def ingest_custom_documents(
    input_dir: str | Path,
    output_csv: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    text_column: str | None = None,
    max_chars: int = 1800,
    overlap_chars: int = 180,
    min_chars: int = 30,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_jsonl = Path(output_jsonl)
    report_json = Path(report_json)
    if not input_dir.exists():
        raise FileNotFoundError(f"Custom input directory not found: {input_dir}")

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    files = iter_input_files(input_dir)
    for file_idx, path in enumerate(files, start=1):
        doc_slug = slugify(path.stem)
        source_rows = read_source_rows(path, text_column=text_column)
        for row_idx, row in enumerate(source_rows, start=1):
            doc_title = clean_text(row.get("doc_title", "")) or path.stem
            section_title = clean_text(row.get("section_title", "")) or f"section_{row_idx}"
            text = clean_text(row.get("text", ""))
            chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
            if not chunks:
                rejected.append({"source_path": str(path), "row": row_idx, "reason": "empty_text"})
                continue
            for chunk_idx, chunk in enumerate(chunks, start=1):
                if len(chunk) < min_chars:
                    rejected.append({"source_path": str(path), "row": row_idx, "chunk": chunk_idx, "reason": "too_short"})
                    continue
                custom_doc_id = f"CUSTOM_{file_idx:04d}"
                doc_key = f"custom_{doc_slug}"
                article_no_norm = f"CHUNK_{row_idx:04d}_{chunk_idx:04d}"
                article_key = f"{doc_key}_{article_no_norm}"
                citation_label = f"{doc_title} - {section_title} - {article_no_norm}"
                generation_text = (
                    f"Doküman: {doc_title}\n"
                    f"Bölüm: {section_title}\n"
                    f"Kaynak dosya: {path.name}\n"
                    f"Metin: {chunk}"
                )
                records.append(
                    {
                        "record_id": f"CUSTOM_{len(records) + 1:06d}",
                        "custom_doc_id": custom_doc_id,
                        "doc_key": doc_key,
                        "article_key": article_key,
                        "source_type": "custom_document",
                        "authority_level": "custom_instructor_document",
                        "official_source": False,
                        "dataset_name": "custom_instructor_documents",
                        "law_name_raw": doc_title,
                        "law_name_norm": doc_title,
                        "law_no_norm": "",
                        "mevzuat_no": "",
                        "mevzuat_tur": "",
                        "mevzuat_tertip": "",
                        "source_url": "",
                        "source_path": str(path),
                        "doc_title": doc_title,
                        "section_title": section_title,
                        "article_no_raw": article_no_norm,
                        "article_no_norm": article_no_norm,
                        "article_type": "custom_chunk",
                        "article_number": str(chunk_idx),
                        "article_body": chunk,
                        "chunk_id": article_no_norm,
                        "chunk_text": chunk,
                        "retrieval_text": f"{doc_title} {section_title}. {chunk}".strip(),
                        "generation_text": generation_text,
                        "citation_label": citation_label,
                        "canonical_source": str(path),
                        "category": "custom",
                        "publication_date": "",
                        "text_length": len(text),
                        "body_length": len(chunk),
                        "quality_flag": "custom_chunk",
                        "duplicate_group_id": "",
                        "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=False),
                    }
                )

    if not records:
        raise ValueError(f"No usable custom document chunks were created from {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "input_dir": str(input_dir),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "input_files": len(files),
        "output_records": len(records),
        "rejected_chunks": len(rejected),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "min_chars": min_chars,
        "rejected_examples": rejected[:20],
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest instructor-provided custom documents into the RAG schema.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=180)
    parser.add_argument("--min-chars", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ingest_custom_documents(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        text_column=args.text_column,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        min_chars=args.min_chars,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
