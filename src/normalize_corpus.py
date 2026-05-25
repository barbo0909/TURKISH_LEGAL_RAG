from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd


ARTICLE_RE = re.compile(
    r"\b((?:EK\s+|GEÇ[İI]C[İI]\s+|MÜKERRER\s+|MUKERRER\s+)?MADDE)\s+"
    r"([0-9]+(?:/[A-ZÇĞİÖŞÜ0-9]+)?(?:\s*[A-ZÇĞİÖŞÜ])?)\b",
    re.IGNORECASE,
)

METADATA_KEYWORDS = (
    "resmî gazete",
    "resmi gazete",
    "kanun no",
    "kabul tarihi",
    "yayımlandığı",
    "mevzuat bilgi sistemi",
    "mevzuat no",
    "mevzuat tür",
    "mevzuat tertip",
)

AMENDMENT_MARKERS = (
    "değişik:",
    "mülga:",
    "ek:",
    "iptal:",
    "yürürlükten kaldırılmıştır",
    "bu madde başlığı ile birlikte değiştirilmiştir",
)

TEXT_COLUMNS = (
    "article_body",
    "article_text",
    "clean_text",
    "text",
    "content",
    "metin",
    "raw_text",
    "full_text",
    "document_text",
    "chunk_text",
)


@dataclass(frozen=True)
class OutputPaths:
    main_csv: Path
    main_jsonl: Path
    qa_csv: Path
    rejected_csv: Path
    report_json: Path


def read_csv_smart(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    encodings = ("utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_column_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def row_get(row: pd.Series, aliases: tuple[str, ...], column_map: dict[str, str]) -> str:
    for alias in aliases:
        normalized = normalize_column_name(alias)
        if normalized in column_map:
            value = normalize_space(row.get(column_map[normalized], ""))
            if value:
                return value
    return ""


def normalize_law_no(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    text = re.sub(r"\.0$", "", text)
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def parse_mevzuat_url(source_url: str) -> dict[str, str]:
    if not source_url:
        return {"mevzuat_no": "", "mevzuat_tur": "", "mevzuat_tertip": ""}
    parsed = urlparse(source_url)
    query = {key.lower(): values for key, values in parse_qs(parsed.query).items()}

    def pick(name: str) -> str:
        values = query.get(name.lower(), [])
        return normalize_space(values[0]) if values else ""

    return {
        "mevzuat_no": normalize_law_no(pick("MevzuatNo")),
        "mevzuat_tur": normalize_law_no(pick("MevzuatTur")),
        "mevzuat_tertip": normalize_law_no(pick("MevzuatTertip")),
    }


def normalize_law_name(value: str) -> str:
    text = normalize_space(value)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text


def looks_like_qa_row(row: pd.Series, column_map: dict[str, str]) -> bool:
    source_type = row_get(row, ("source_type", "record_type", "type", "dataset_name"), column_map).lower()
    if any(marker in source_type for marker in ("qa", "soru", "question", "cevap", "answer")):
        return True

    question = row_get(row, ("question", "soru", "query"), column_map)
    answer = row_get(row, ("answer", "cevap", "response", "gold_answer"), column_map)
    if question and answer:
        return True

    text = row_get(row, TEXT_COLUMNS, column_map).lower()
    return bool(re.search(r"\bsoru\s*:", text) and re.search(r"\bcevap\s*:", text))


def extract_article_no(raw_article_no: str, text: str) -> tuple[str, str, str]:
    candidate = normalize_space(raw_article_no)
    if not candidate:
        match = ARTICLE_RE.search(text)
        if match:
            candidate = f"{match.group(1)} {match.group(2)}"

    candidate = candidate.replace(":", " ").replace("-", " ")
    candidate = re.sub(r"\s+", " ", candidate).strip().upper()
    candidate = candidate.replace("GEÇICI", "GEÇİCİ").replace("MUKERRER", "MÜKERRER")

    if not candidate:
        return "", "", ""

    if re.fullmatch(r"\d+(?:/[A-ZÇĞİÖŞÜ0-9]+)?(?:\s*[A-ZÇĞİÖŞÜ])?", candidate):
        candidate = f"MADDE {candidate}"

    match = ARTICLE_RE.search(candidate)
    if match:
        article_type = match.group(1).upper().replace("GEÇICI", "GEÇİCİ").replace("MUKERRER", "MÜKERRER")
        number = normalize_space(match.group(2)).upper()
        return f"{article_type} {number}", article_type, number

    return candidate, "MADDE", re.sub(r"[^0-9A-ZÇĞİÖŞÜ/]+", "", candidate)


def extract_article_body(text: str, article_no_norm: str) -> str:
    cleaned = normalize_space(text)
    if not cleaned:
        return ""

    if article_no_norm:
        pattern = re.compile(re.escape(article_no_norm).replace(r"\ ", r"\s+") + r"\s*[-–—:]?\s*", re.IGNORECASE)
        matches = list(pattern.finditer(cleaned))
        if matches:
            cleaned = cleaned[matches[-1].end() :]

    if not article_no_norm:
        match = ARTICLE_RE.search(cleaned)
        if match:
            cleaned = cleaned[match.end() :]
            cleaned = re.sub(r"^\s*[-–—:]?\s*", "", cleaned)

    cleaned = re.sub(r"^(metin|madde metni)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    return normalize_space(cleaned)


def classify_quality(article_body: str, article_no_norm: str) -> str:
    body = normalize_space(article_body)
    lower = body.lower()
    if not body or len(body) < 12:
        return "empty_or_too_short"

    metadata_hits = sum(1 for keyword in METADATA_KEYWORDS if keyword in lower)
    if metadata_hits >= 2 and len(body) < 220:
        return "metadata_only"

    if any(marker in lower for marker in AMENDMENT_MARKERS) and len(body) < 180:
        return "amendment_fragment"

    delimiter_count = body.count("|") + body.count("\t") + body.count(";")
    if delimiter_count >= 8 and len(body) < 500:
        return "table_fragment"

    if len(body) < 80:
        return "short_but_valid" if article_no_norm else "empty_or_too_short"

    return "valid_article"


def stable_hash(text: str) -> str:
    normalized = normalize_space(text).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def slugify_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:80] or "unknown"


def make_doc_key(mevzuat_tur: str, mevzuat_tertip: str, mevzuat_no: str, law_no: str, law_name: str) -> str:
    if mevzuat_tur and mevzuat_tertip and mevzuat_no:
        return f"{mevzuat_tur}_{mevzuat_tertip}_{mevzuat_no}"
    if law_no:
        return f"law_{law_no}"
    return f"law_{slugify_key(law_name)}"


def make_citation_label(law_name: str, article_type: str, article_number: str, article_no_norm: str) -> str:
    law = law_name or "Bilinmeyen Kanun"
    if not article_number:
        return f"{law} {article_no_norm}".strip()
    article_type_upper = article_type.upper()
    if article_type_upper.startswith("EK MADDE"):
        return f"{law} ek m.{article_number}"
    if article_type_upper.startswith("GEÇİCİ MADDE"):
        return f"{law} geçici m.{article_number}"
    if article_type_upper.startswith("MÜKERRER MADDE"):
        return f"{law} mükerrer m.{article_number}"
    return f"{law} m.{article_number}"


def normalize_law_rows(df: pd.DataFrame, column_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        source_url = row_get(row, ("source_url", "url", "official_url", "canonical_url"), column_map)
        parsed_url = parse_mevzuat_url(source_url)
        raw_text = row_get(row, TEXT_COLUMNS, column_map)
        raw_article_no = row_get(row, ("article_no", "madde_no", "article_number", "madde", "article"), column_map)
        article_no_norm, article_type, article_number = extract_article_no(raw_article_no, raw_text)
        article_body = extract_article_body(raw_text, article_no_norm)

        law_name_raw = row_get(
            row,
            ("law_name_raw", "law_name", "kanun_adi", "mevzuat_adi", "legislation_name", "title", "doc_title"),
            column_map,
        )
        law_name_norm = normalize_law_name(law_name_raw)
        law_no_norm = normalize_law_no(
            row_get(row, ("law_no", "law_number", "kanun_no", "kanun_numarasi"), column_map)
        )
        mevzuat_no = parsed_url["mevzuat_no"] or normalize_law_no(
            row_get(row, ("mevzuat_no", "mevzuatno"), column_map)
        )
        mevzuat_tur = parsed_url["mevzuat_tur"] or normalize_law_no(
            row_get(row, ("mevzuat_tur", "mevzuattur"), column_map)
        )
        mevzuat_tertip = parsed_url["mevzuat_tertip"] or normalize_law_no(
            row_get(row, ("mevzuat_tertip", "mevzuattertip"), column_map)
        )
        if not law_no_norm:
            law_no_norm = mevzuat_no

        doc_key = make_doc_key(mevzuat_tur, mevzuat_tertip, mevzuat_no, law_no_norm, law_name_norm)
        article_key = f"{doc_key}_{article_no_norm.replace(' ', '_')}" if article_no_norm else f"{doc_key}_ARTICLE_UNKNOWN_{position:06d}"
        quality_flag = classify_quality(article_body, article_no_norm)
        official_source = "mevzuat.gov.tr" in source_url.lower()
        citation_label = make_citation_label(law_name_norm, article_type, article_number, article_no_norm)

        record = {
            "record_id": f"LAW_{position:06d}",
            "doc_key": doc_key,
            "article_key": article_key,
            "source_type": "official_law",
            "authority_level": "official_law" if official_source else "unverified_law_record",
            "official_source": official_source,
            "dataset_name": row_get(row, ("dataset_name", "dataset", "source_dataset"), column_map),
            "law_name_raw": law_name_raw,
            "law_name_norm": law_name_norm,
            "law_no_norm": law_no_norm,
            "mevzuat_no": mevzuat_no,
            "mevzuat_tur": mevzuat_tur,
            "mevzuat_tertip": mevzuat_tertip,
            "source_url": source_url,
            "article_no_raw": raw_article_no,
            "article_no_norm": article_no_norm,
            "article_type": article_type,
            "article_number": article_number,
            "article_body": article_body,
            "retrieval_text": normalize_space(f"{law_name_norm} {article_no_norm}. {article_body}"),
            "generation_text": f"Kanun: {law_name_norm}\nMadde: {article_no_norm}\nMetin: {article_body}",
            "citation_label": citation_label,
            "canonical_source": source_url or row_get(row, ("canonical_source", "source", "source_name"), column_map),
            "category": row_get(row, ("category", "topic", "legal_area", "hukuk_alani"), column_map),
            "publication_date": row_get(row, ("publication_date", "date", "resmi_gazete_tarihi"), column_map),
            "text_length": len(raw_text),
            "body_length": len(article_body),
            "quality_flag": quality_flag,
            "duplicate_group_id": "",
            "body_hash": stable_hash(article_body),
        }

        records.append(record)

    normalized = pd.DataFrame(records)
    if normalized.empty:
        return normalized, pd.DataFrame(rejected)

    normalized["duplicate_group_id"] = ""
    duplicate_counter = 0

    duplicate_body_mask = normalized.duplicated("body_hash", keep=False) & normalized["body_hash"].ne(stable_hash(""))
    for _, group in normalized[duplicate_body_mask].groupby("body_hash", sort=False):
        if len(group) < 2:
            continue
        duplicate_counter += 1
        group_id = f"DUP_BODY_{duplicate_counter:04d}"
        normalized.loc[group.index, "duplicate_group_id"] = group_id
        keep_idx = group.sort_values(["body_length", "record_id"], ascending=[False, True]).index[0]
        drop_indexes = [idx for idx in group.index if idx != keep_idx]
        normalized.loc[drop_indexes, "quality_flag"] = "duplicate_candidate"

    article_conflict_mask = normalized.duplicated("article_key", keep=False)
    for _, group in normalized[article_conflict_mask].groupby("article_key", sort=False):
        if group["body_hash"].nunique() <= 1:
            continue
        duplicate_counter += 1
        group_id = f"DUP_CONFLICT_{duplicate_counter:04d}"
        normalized.loc[group.index, "duplicate_group_id"] = group_id
        keep_idx = group.sort_values(["body_length", "record_id"], ascending=[False, True]).index[0]
        drop_indexes = [idx for idx in group.index if idx != keep_idx]
        normalized.loc[drop_indexes, "quality_flag"] = "duplicate_conflict"

    valid_flags = {"valid_article", "short_but_valid"}
    main = normalized[normalized["quality_flag"].isin(valid_flags)].copy()
    rejected_df = normalized[~normalized["quality_flag"].isin(valid_flags)].copy()

    main = main.drop(columns=["body_hash"])
    rejected_df = rejected_df.drop(columns=["body_hash"])
    return main.reset_index(drop=True), rejected_df.reset_index(drop=True)


def normalize_qa_rows(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        question = row_get(row, ("question", "soru", "query"), column_map)
        answer = row_get(row, ("answer", "cevap", "response", "gold_answer"), column_map)
        text = row_get(row, TEXT_COLUMNS, column_map)
        records.append(
            {
                "record_id": f"QA_{position:06d}",
                "source_type": "auxiliary_qa",
                "authority_level": "auxiliary_qa",
                "official_source": False,
                "dataset_name": row_get(row, ("dataset_name", "dataset", "source_dataset"), column_map),
                "question": question,
                "answer": answer,
                "text": text,
                "source_url": row_get(row, ("source_url", "url", "official_url", "canonical_url"), column_map),
                "category": row_get(row, ("category", "topic", "legal_area", "hukuk_alani"), column_map),
                "canonical_source": row_get(row, ("canonical_source", "source", "source_name"), column_map),
            }
        )
    return pd.DataFrame(records)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(raw_df: pd.DataFrame, law_df: pd.DataFrame, qa_df: pd.DataFrame, main_df: pd.DataFrame, rejected_df: pd.DataFrame) -> dict[str, Any]:
    combined_law = pd.concat([main_df, rejected_df], ignore_index=True) if not rejected_df.empty else main_df.copy()
    report = {
        "raw_rows": int(len(raw_df)),
        "law_rows": int(len(law_df)),
        "qa_rows": int(len(qa_df)),
        "valid_law_rows": int(len(main_df)),
        "rejected_rows": int(len(rejected_df)),
        "suspicious_rows": int(len(rejected_df)),
        "missing_source_url": int((combined_law.get("source_url", pd.Series(dtype=str)) == "").sum()) if not combined_law.empty else 0,
        "missing_article_no": int((combined_law.get("article_no_norm", pd.Series(dtype=str)) == "").sum()) if not combined_law.empty else 0,
        "missing_article_body": int((combined_law.get("article_body", pd.Series(dtype=str)) == "").sum()) if not combined_law.empty else 0,
        "duplicate_groups": int(combined_law.get("duplicate_group_id", pd.Series(dtype=str)).replace("", pd.NA).dropna().nunique()) if not combined_law.empty else 0,
        "canonical_source_missing": int((combined_law.get("canonical_source", pd.Series(dtype=str)) == "").sum()) if not combined_law.empty else 0,
        "category_missing": int((combined_law.get("category", pd.Series(dtype=str)) == "").sum()) if not combined_law.empty else 0,
        "quality_flag_counts": rejected_df["quality_flag"].value_counts().to_dict() if not rejected_df.empty else {},
        "main_quality_flag_counts": main_df["quality_flag"].value_counts().to_dict() if not main_df.empty else {},
    }
    return report


def normalize_curated_corpus(input_csv: str | Path, outputs: OutputPaths) -> dict[str, Any]:
    raw_df = read_csv_smart(input_csv)
    column_map = {normalize_column_name(col): col for col in raw_df.columns}

    if "source_type" in column_map:
        source_type = raw_df[column_map["source_type"]].astype(str).str.lower()
        qa_mask = source_type.str.contains(r"\b(?:qa|soru|question|cevap|answer)\b", regex=True, na=False)
        question_col = column_map.get("question") or column_map.get("soru") or column_map.get("query")
        answer_col = column_map.get("answer") or column_map.get("cevap") or column_map.get("response")
        if question_col and answer_col:
            qa_mask = qa_mask | (raw_df[question_col].astype(str).str.strip().ne("") & raw_df[answer_col].astype(str).str.strip().ne(""))
    else:
        qa_mask = raw_df.apply(lambda row: looks_like_qa_row(row, column_map), axis=1)
    qa_raw = raw_df[qa_mask].copy()
    law_raw = raw_df[~qa_mask].copy()

    main_df, rejected_df = normalize_law_rows(law_raw, column_map)
    qa_df = normalize_qa_rows(qa_raw, column_map)

    for path in (outputs.main_csv, outputs.main_jsonl, outputs.qa_csv, outputs.rejected_csv, outputs.report_json):
        path.parent.mkdir(parents=True, exist_ok=True)

    main_df.to_csv(outputs.main_csv, index=False, encoding="utf-8-sig")
    write_jsonl(main_df, outputs.main_jsonl)
    qa_df.to_csv(outputs.qa_csv, index=False, encoding="utf-8-sig")
    rejected_df.to_csv(outputs.rejected_csv, index=False, encoding="utf-8-sig")

    report = build_report(raw_df, law_raw, qa_raw, main_df, rejected_df)
    with outputs.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize the curated Turkish legal corpus.")
    parser.add_argument("--input", required=True, help="Path to legal_documents_curated.csv")
    parser.add_argument("--processed-dir", required=True, help="Output directory for processed corpus files")
    parser.add_argument("--reports-dir", required=True, help="Output directory for reports")
    parser.add_argument("--version", default="v3", help="Corpus version suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    reports_dir = Path(args.reports_dir)
    outputs = OutputPaths(
        main_csv=processed_dir / f"legal_main_law_corpus_{args.version}.csv",
        main_jsonl=processed_dir / f"legal_main_law_corpus_{args.version}.jsonl",
        qa_csv=processed_dir / f"legal_qa_auxiliary_{args.version}.csv",
        rejected_csv=processed_dir / f"legal_rejected_review_{args.version}.csv",
        report_json=reports_dir / f"normalization_report_{args.version}.json",
    )
    report = normalize_curated_corpus(args.input, outputs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
