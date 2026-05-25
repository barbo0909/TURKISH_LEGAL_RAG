from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

try:
    from .embeddings import encode_queries, load_embedding_model
    from .retrieval import RetrievalEngine, load_jsonl
except ImportError:
    from embeddings import encode_queries, load_embedding_model
    from retrieval import RetrievalEngine, load_jsonl


LAW_KEYWORDS: dict[str, list[str]] = {
    "1_5_2709": [
        "anayasa",
        "anayasa mahkemesi",
        "bireysel başvuru",
        "tbmm",
        "cumhurbaşkanı",
        "yasama",
        "milletvekili",
        "milletlerarası antlaşma",
        "kanun hükmünde kararname",
        "cumhurbaşkanlığı kararnamesi",
        "şekil bakımından denetim",
        "itiraz yolu",
        "somut norm",
        "egemenlik",
    ],
    "1_5_6098": [
        "borç",
        "borçlar",
        "sözleşme",
        "kira",
        "kiracı",
        "kiraya veren",
        "tahliye",
        "temerrüt",
        "ceza koşulu",
        "hata",
        "yanılma",
        "aldatma",
        "korkutma",
        "vekalet",
        "bağışlama",
        "satış",
        "faiz",
        "zamanaşımı",
    ],
    "1_5_5237": [
        "türk ceza",
        "suç",
        "ceza",
        "kasten",
        "taksir",
        "meşru savunma",
        "haksız tahrik",
        "teşebbüs",
        "iştirak",
        "zincirleme suç",
        "adli para",
        "dava zamanaşımı",
        "şikayet",
    ],
    "1_5_5271": [
        "ceza muhakemesi",
        "cmk",
        "tutuklama",
        "yakalama",
        "gözaltı",
        "arama",
        "elkoyma",
        "müdafi",
        "iddianame",
        "hüküm",
        "mahkeme",
        "koruma tedbiri",
        "iletişimin tespiti",
        "tanık",
        "bilirkişi",
    ],
    "1_5_4721": [
        "medeni",
        "evlilik",
        "boşanma",
        "miras",
        "mirasçı",
        "tereke",
        "velayet",
        "vesayet",
        "soybağı",
        "eşya",
        "tapu",
        "kişilik",
        "fiil ehliyeti",
        "ayırt etme gücü",
    ],
    "1_5_4857": [
        "iş kanunu",
        "işçi",
        "işveren",
        "kıdem",
        "ihbar",
        "fazla çalışma",
        "fazla mesai",
        "hafta tatili",
        "işe iade",
        "geçersiz fesih",
        "haklı fesih",
        "yıllık izin",
        "çalışma süresi",
        "iş sözleşmesi",
    ],
    "1_5_6102": [
        "ticaret",
        "tacir",
        "şirket",
        "anonim",
        "limited",
        "pay",
        "sermaye",
        "yönetim kurulu",
        "genel kurul",
        "çek",
        "bono",
        "poliçe",
        "ticari işletme",
        "haksız rekabet",
    ],
}


SYNONYMS: dict[str, list[str]] = {
    "veto": ["geri gönderme", "cumhurbaşkanı kanunu geri gönderir", "bir daha görüşülmek üzere"],
    "temerrüt faizi": ["borçlunun temerrüdü", "temerrüt", "faiz"],
    "işe iade": ["geçersiz fesih", "iş güvencesi", "fesih bildirimi"],
    "tahliye davası": ["kiraya veren", "kiracı", "kira sözleşmesi", "tahliye", "gereksinim", "yeniden inşa"],
    "olağan kanun yolları": ["başvuru yollarının tüketilmesi", "bireysel başvuru"],
    "milletlerarası antlaşma": ["usulüne göre yürürlüğe konulmuş", "anayasa aykırılık iddiası"],
    "şekil bakımından": ["son oylama", "ivedilikle görüşülemez", "anayasa değişikliği"],
    "somut norm denetimi": ["itiraz yolu", "uygulanacak kural", "anayasa aykırılığı"],
    "cumhurbaşkanlığı kararnamesi": ["temel haklar", "siyasi haklar", "kanunda açıkça düzenlenen konular"],
    "yasama dokunulmazlığı": ["yasama sorumsuzluğu", "milletvekili", "meclis çalışmaları"],
}


def normalize(text: str) -> str:
    return str(text).lower().replace("ı", "i")


def expand_query(query: str) -> str:
    lowered = normalize(query)
    additions: list[str] = []
    for key, values in SYNONYMS.items():
        if key in lowered:
            additions.extend(values)
    for doc_key, keywords in LAW_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            additions.extend(keywords[:8])
    if not additions:
        return query
    unique_additions = []
    seen = set()
    for item in additions:
        if item not in seen and item not in lowered:
            unique_additions.append(item)
            seen.add(item)
    return f"{query} {' '.join(unique_additions)}"


def infer_law_filters(query: str, min_score: int = 1, max_laws: int = 2) -> list[str]:
    lowered = normalize(query)
    scored: list[tuple[str, int]] = []
    for doc_key, keywords in LAW_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score >= min_score:
            scored.append((doc_key, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [doc_key for doc_key, _ in scored[:max_laws]]


class OptimizedRetrievalEngine:
    def __init__(self, index_root: str | Path, device: str | None = None) -> None:
        self.base = RetrievalEngine(index_root=index_root, device=device)
        self.index_root = Path(index_root)
        self.embedding_model_name = self.base.embedding_model_name
        self.device = device
        self._embeddings: np.ndarray | None = None
        self.doc_to_indices: dict[str, list[int]] = {}
        for idx, record in enumerate(self.base.metadata):
            self.doc_to_indices.setdefault(str(record.get("doc_key", "")), []).append(idx)

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = np.load(self.base.manifest["embeddings_path"]).astype("float32")
        return self._embeddings

    @property
    def embedding_model(self):
        return self.base.embedding_model

    def encode_query(self, query: str) -> np.ndarray:
        vector = encode_queries(self.embedding_model, [query], self.embedding_model_name).astype("float32")
        return vector[0]

    def filtered_dense_search(self, query: str, doc_keys: list[str], top_k: int = 10) -> list[dict[str, Any]]:
        indices: list[int] = []
        for doc_key in doc_keys:
            indices.extend(self.doc_to_indices.get(doc_key, []))
        if not indices:
            return []
        query_vector = self.encode_query(query)
        sub_embeddings = self.embeddings[np.array(indices)]
        scores = sub_embeddings @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for pos in order:
            idx = indices[int(pos)]
            record = dict(self.base.metadata[idx])
            record.update(
                {
                    "score": float(scores[int(pos)]),
                    "dense_score": float(scores[int(pos)]),
                    "retriever": "optimized_law_filtered_dense",
                }
            )
            results.append(record)
        return results

    def search(self, query: str, top_k: int = 10, fallback_k: int = 100) -> list[dict[str, Any]]:
        expanded_query = expand_query(query)
        law_filters = infer_law_filters(query)
        if law_filters:
            filtered = self.filtered_dense_search(expanded_query, law_filters, top_k=top_k)
            if filtered:
                return filtered
        return self.base.dense_search(expanded_query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized retrieval smoke test.")
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = OptimizedRetrievalEngine(args.index_root, device=args.device)
    for rank, result in enumerate(engine.search(args.query, top_k=args.top_k), start=1):
        print(rank, result.get("citation_label"), result.get("article_key"), result.get("score"))


if __name__ == "__main__":
    main()
