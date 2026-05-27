from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


MOJIBAKE_MARKERS = ("Ã", "Ä", "Å", "â")
ID_PAIR_RE = re.compile(r"(?:^|_)(\d{4})_.*?_m(\d+)", re.IGNORECASE)


def fix_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        try:
            return text.encode("latin1").decode("utf-8")
        except Exception:
            return text
    return text


def clean_article_no(value: Any) -> str:
    match = re.search(r"\d+", str(value or ""))
    return match.group(0) if match else ""


def pair_from_external_id(value: Any) -> tuple[str, str] | None:
    match = ID_PAIR_RE.search(str(value or ""))
    if not match:
        return None
    return match.group(1), match.group(2)


def pair_from_gold_source(source: dict[str, Any]) -> tuple[str, str] | None:
    law_no = clean_article_no(source.get("law_no"))
    article_no = clean_article_no(source.get("article_no"))
    if law_no and article_no:
        return law_no, article_no
    return pair_from_external_id(source.get("source_id") or source.get("corpus_row_id"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def official_maps(corpus_csv: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    corpus = pd.read_csv(corpus_csv, dtype=str, keep_default_na=False)
    required = {"mevzuat_no", "article_number", "doc_key", "article_key", "retrieval_text", "generation_text", "citation_label"}
    missing = sorted(required - set(corpus.columns))
    if missing:
        raise ValueError(f"Official corpus is missing required columns: {missing}")

    pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in corpus.iterrows():
        pair = (clean_article_no(row.get("mevzuat_no")), clean_article_no(row.get("article_number")))
        if pair[0] and pair[1] and pair not in pair_map:
            pair_map[pair] = row.to_dict()
    return corpus, pair_map


def same_doc_negative(corpus: pd.DataFrame, positive: dict[str, Any], used_article_keys: set[str]) -> dict[str, Any] | None:
    doc_key = str(positive.get("doc_key") or "")
    subset = corpus[
        (corpus["doc_key"].astype(str) == doc_key)
        & (~corpus["article_key"].astype(str).isin(used_article_keys))
    ]
    if subset.empty:
        return None
    return subset.sample(n=1, random_state=42).iloc[0].to_dict()


def random_negative(corpus: pd.DataFrame, rng: random.Random, used_article_keys: set[str]) -> dict[str, Any] | None:
    if corpus.empty:
        return None
    for _ in range(100):
        row = corpus.iloc[rng.randrange(len(corpus))].to_dict()
        if str(row.get("article_key") or "") not in used_article_keys:
            return row
    return None


def extract_question_from_user(content: str) -> str:
    text = fix_text(content)
    marker = "Soru:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text[-600:].strip()


def strip_external_source(answer: str) -> str:
    text = fix_text(answer)
    for marker in ("\n\nKaynak:", "\nKaynak:"):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text.strip()


def make_llm_input(question: str, rows: list[dict[str, Any]], max_context_chars: int) -> str:
    blocks = [f"Soru: {question}", "", "Kaynaklar:"]
    used = 0
    for idx, row in enumerate(rows, start=1):
        citation = str(row.get("citation_label") or row.get("article_key") or "")
        article_key = str(row.get("article_key") or "")
        text = str(row.get("generation_text") or row.get("retrieval_text") or "").strip()
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        used += len(text)
        blocks.append(f"[{idx}] {citation}\narticle_key={article_key}\n{text}")
    return "\n\n".join(blocks).strip()


def make_llm_output(answer: str, rows: list[dict[str, Any]]) -> str:
    citations = []
    for row in rows:
        citation = str(row.get("citation_label") or row.get("article_key") or "")
        article_key = str(row.get("article_key") or "")
        citations.append(f"- Kaynak: {citation} | article_key={article_key}")
    return "\n".join(
        [
            "1. Kisa cevap",
            answer.strip(),
            "",
            "2. Dayanak maddeler",
            "\n".join(citations),
        ]
    ).strip()


def split_train_val(rows: list[dict[str, Any]], rng: random.Random, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if not shuffled:
        return [], []
    val_count = max(1, round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def prepare_external_mapped_tuning_data(
    corpus_csv: Path,
    external_dir: Path,
    output_dir: Path,
    report_path: Path,
    seed: int = 42,
    val_ratio: float = 0.1,
    max_context_chars: int = 9000,
) -> dict[str, Any]:
    rng = random.Random(seed)
    corpus, pair_map = official_maps(corpus_csv)

    embedding_rows: list[dict[str, Any]] = []
    reranker_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "external_embedding_rows": 0,
        "external_embedding_mapped": 0,
        "external_reranker_rows": 0,
        "external_reranker_mapped": 0,
        "external_llm_rows": 0,
        "external_llm_mapped": 0,
        "external_gold_questions": 0,
        "external_gold_mapped": 0,
    }

    for row in read_jsonl(external_dir / "embedding.jsonl"):
        counts["external_embedding_rows"] += 1
        pos_pair = pair_from_external_id(row.get("positive_id"))
        neg_pair = pair_from_external_id(row.get("negative_id"))
        if not pos_pair or pos_pair not in pair_map:
            continue
        positive = pair_map[pos_pair]
        negative = pair_map.get(neg_pair) if neg_pair else None
        if negative is None:
            used = {str(positive.get("article_key") or "")}
            negative = same_doc_negative(corpus, positive, used) or random_negative(corpus, rng, used)
        if negative is None:
            continue
        counts["external_embedding_mapped"] += 1
        embedding_rows.append(
            {
                "id": f"external_mapped_emb_{counts['external_embedding_mapped']:06d}",
                "query": fix_text(row.get("query")),
                "positive_passage": str(positive.get("retrieval_text") or ""),
                "negative_passage": str(negative.get("retrieval_text") or ""),
                "positive_id": str(positive.get("article_key") or ""),
                "negative_id": str(negative.get("article_key") or ""),
                "positive_citation": str(positive.get("citation_label") or ""),
                "negative_citation": str(negative.get("citation_label") or ""),
                "negative_type": fix_text(row.get("negative_type")) or "external_or_sampled_negative",
                "source": "external_mapped_official_law",
            }
        )

    for row in read_jsonl(external_dir / "reranker.jsonl"):
        counts["external_reranker_rows"] += 1
        pair = pair_from_external_id(row.get("candidate_id"))
        if not pair or pair not in pair_map:
            continue
        candidate = pair_map[pair]
        counts["external_reranker_mapped"] += 1
        reranker_rows.append(
            {
                "id": f"external_mapped_rr_{counts['external_reranker_mapped']:06d}",
                "query_id": fix_text(row.get("query_id")) or f"external_rr_q_{counts['external_reranker_mapped']:06d}",
                "query": fix_text(row.get("query")),
                "candidate_passage": str(candidate.get("retrieval_text") or ""),
                "label": int(float(row.get("label", 0))),
                "candidate_id": str(candidate.get("article_key") or ""),
                "citation_label": str(candidate.get("citation_label") or ""),
                "source": "external_mapped_official_law",
                "negative_type": fix_text(row.get("negative_type")),
            }
        )

    gold_items = json.loads((external_dir / "gold_benchmark.json").read_text(encoding="utf-8"))
    for item in gold_items:
        counts["external_gold_questions"] += 1
        mapped_rows = []
        for source in item.get("gold_sources", []):
            pair = pair_from_gold_source(source)
            if pair and pair in pair_map:
                mapped_rows.append(pair_map[pair])
        if not mapped_rows:
            continue
        counts["external_gold_mapped"] += 1
        question = fix_text(item.get("question"))
        answer = strip_external_source(item.get("verified_answer")) or str(mapped_rows[0].get("article_body") or "")[:900]
        llm_rows.append(
            {
                "id": f"external_gold_llm_{counts['external_gold_mapped']:06d}",
                "instruction": (
                    "Verilen resmi mevzuat kaynaklarina dayali Turkce bir hukuk cevabi uret. "
                    "Kaynakta olmayan bilgiyi ekleme ve dayanak maddeleri article_key ile belirt."
                ),
                "input": make_llm_input(question, mapped_rows, max_context_chars=max_context_chars),
                "output": make_llm_output(answer, mapped_rows),
                "source": "external_gold_benchmark_mapped_official_law",
                "gold_article_keys": "; ".join(str(row.get("article_key") or "") for row in mapped_rows),
            }
        )

        used = {str(row.get("article_key") or "") for row in mapped_rows}
        neg = random_negative(corpus, rng, used)
        if neg is not None:
            embedding_rows.append(
                {
                    "id": f"external_gold_emb_{counts['external_gold_mapped']:06d}",
                    "query": question,
                    "positive_passage": str(mapped_rows[0].get("retrieval_text") or ""),
                    "negative_passage": str(neg.get("retrieval_text") or ""),
                    "positive_id": str(mapped_rows[0].get("article_key") or ""),
                    "negative_id": str(neg.get("article_key") or ""),
                    "positive_citation": str(mapped_rows[0].get("citation_label") or ""),
                    "negative_citation": str(neg.get("citation_label") or ""),
                    "negative_type": "sampled_global_negative",
                    "source": "external_gold_benchmark_mapped_official_law",
                }
            )
            reranker_rows.append(
                {
                    "id": f"external_gold_rr_{counts['external_gold_mapped']:06d}_pos",
                    "query_id": fix_text(item.get("question_id")) or f"external_gold_{counts['external_gold_mapped']:06d}",
                    "query": question,
                    "candidate_passage": str(mapped_rows[0].get("retrieval_text") or ""),
                    "label": 1,
                    "candidate_id": str(mapped_rows[0].get("article_key") or ""),
                    "citation_label": str(mapped_rows[0].get("citation_label") or ""),
                    "source": "external_gold_benchmark_mapped_official_law",
                    "negative_type": "",
                }
            )
            reranker_rows.append(
                {
                    "id": f"external_gold_rr_{counts['external_gold_mapped']:06d}_neg",
                    "query_id": fix_text(item.get("question_id")) or f"external_gold_{counts['external_gold_mapped']:06d}",
                    "query": question,
                    "candidate_passage": str(neg.get("retrieval_text") or ""),
                    "label": 0,
                    "candidate_id": str(neg.get("article_key") or ""),
                    "citation_label": str(neg.get("citation_label") or ""),
                    "source": "external_gold_benchmark_mapped_official_law",
                    "negative_type": "sampled_global_negative",
                }
            )

    for row in read_jsonl(external_dir / "llm.jsonl"):
        counts["external_llm_rows"] += 1
        metadata = row.get("metadata", {})
        pair = pair_from_external_id(metadata.get("source_id") or metadata.get("corpus_row_id") or metadata.get("chunk_id"))
        if not pair or pair not in pair_map:
            continue
        official = pair_map[pair]
        messages = row.get("messages", [])
        user = next((msg for msg in messages if msg.get("role") == "user"), {})
        assistant = next((msg for msg in messages if msg.get("role") == "assistant"), {})
        question = extract_question_from_user(user.get("content", ""))
        answer = strip_external_source(assistant.get("content", ""))
        if not question or not answer:
            continue
        counts["external_llm_mapped"] += 1
        llm_rows.append(
            {
                "id": f"external_llm_mapped_{counts['external_llm_mapped']:06d}",
                "instruction": (
                    "Verilen resmi mevzuat kaynagina dayali Turkce bir hukuk cevabi uret. "
                    "Cevabin sonunda dayanak maddeyi article_key ile belirt."
                ),
                "input": make_llm_input(question, [official], max_context_chars=max_context_chars),
                "output": make_llm_output(answer, [official]),
                "source": "external_sft_mapped_official_law",
                "gold_article_keys": str(official.get("article_key") or ""),
            }
        )

    emb_train, emb_val = split_train_val(embedding_rows, rng, val_ratio)
    rr_train, rr_val = split_train_val(reranker_rows, rng, val_ratio)
    llm_train, llm_val = split_train_val(llm_rows, rng, val_ratio)

    paths = {
        "embedding_train": output_dir / "external_mapped_embedding_train.jsonl",
        "embedding_val": output_dir / "external_mapped_embedding_val.jsonl",
        "reranker_train": output_dir / "external_mapped_reranker_train.jsonl",
        "reranker_val": output_dir / "external_mapped_reranker_val.jsonl",
        "llm_train": output_dir / "external_mapped_llm_train.jsonl",
        "llm_val": output_dir / "external_mapped_llm_val.jsonl",
    }
    write_jsonl(paths["embedding_train"], emb_train)
    write_jsonl(paths["embedding_val"], emb_val)
    write_jsonl(paths["reranker_train"], rr_train)
    write_jsonl(paths["reranker_val"], rr_val)
    write_jsonl(paths["llm_train"], llm_train)
    write_jsonl(paths["llm_val"], llm_val)

    report = {
        "corpus_csv": str(corpus_csv),
        "external_dir": str(external_dir),
        "seed": seed,
        "val_ratio": val_ratio,
        **counts,
        "embedding_train_records": len(emb_train),
        "embedding_val_records": len(emb_val),
        "reranker_train_records": len(rr_train),
        "reranker_val_records": len(rr_val),
        "llm_train_records": len(llm_train),
        "llm_val_records": len(llm_val),
        "outputs": {key: str(value) for key, value in paths.items()},
        "leakage_policy": (
            "The locked final benchmark is not used for tuning. External examples are mapped to official "
            "article_key values only when law_no/article_no can be matched to the official corpus."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map external legal tuning data to the official-law schema.")
    parser.add_argument("--corpus-csv", default="data/processed/legal_main_law_corpus_v3.csv")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report-path", default="reports/external_mapped_tuning_data_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_external_mapped_tuning_data(
        corpus_csv=Path(args.corpus_csv),
        external_dir=Path(args.external_dir),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report_path),
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_context_chars=args.max_context_chars,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
