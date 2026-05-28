from __future__ import annotations

from typing import Any


DEFAULT_PROMPT_VERSION = "citation_aware_tr_v2_strict_context_citations"


SYSTEM_INSTRUCTION = (
    "Sen Türk hukuku alanında çalışan citation-aware bir hukuk asistanısın. "
    "Sadece verilen mevzuat bağlamına dayanarak cevap ver. "
    "Bağlamda cevap yoksa bunu açıkça söyle. "
    "Uydurma kanun, madde, karar veya kaynak üretme. "
    "Dayanak maddeler bölümünde yalnızca bağlamdaki Kaynak satırlarını aynen kullan."
)


def format_context(retrieved_items: list[dict[str, Any]], max_chars: int = 9000) -> str:
    blocks: list[str] = []
    total = 0
    for idx, item in enumerate(retrieved_items, start=1):
        generation_text = str(item.get("generation_text", "")).strip()
        citation_label = str(item.get("citation_label", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        article_key = str(item.get("article_key", "")).strip()
        block = (
            f"[{idx}]\n"
            f"Kaynak: {citation_label}\n"
            f"ArticleKey: {article_key}\n"
            f"URL: {source_url}\n"
            f"{generation_text}"
        ).strip()
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, retrieved_items: list[dict[str, Any]], max_context_chars: int = 9000) -> str:
    context = format_context(retrieved_items, max_chars=max_context_chars)
    return f"""{SYSTEM_INSTRUCTION}

Soru:
{question}

Bağlam:
{context}

Cevap formatı:
1. Kısa cevap
2. Şartlar / açıklama
3. Dayanak maddeler
4. Not / sınırlılık

Cevap:
"""


CONCISE_EVAL_SYSTEM_INSTRUCTION = (
    "Sen Turk hukuku alaninda calisan citation-aware bir hukuk asistanisin. "
    "Sadece verilen mevzuat baglamina dayanarak cevap ver. "
    "Cevabi gold benchmark stiline yakin, kisa ve dogrudan yaz. "
    "Gereksiz aciklama yapma ve uydurma kaynak uretme."
)


def build_concise_eval_rag_prompt(
    question: str,
    retrieved_items: list[dict[str, Any]],
    max_context_chars: int = 9000,
) -> str:
    context = format_context(retrieved_items, max_chars=max_context_chars)
    return f"""{CONCISE_EVAL_SYSTEM_INSTRUCTION}

Soru:
{question}

Baglam:
{context}

Cevap kurallari:
- Ilk cumlede nihai cevabi ver.
- En fazla 2-3 cumle kullan.
- Gold benchmark cevabi gibi kisa, dogrudan ve hukuki sonucu belirten bir cevap yaz.
- Son satirda yalnizca baglamdaki ilgili kaynak veya maddeyi dayanak olarak yaz.
- Baglamda cevap yoksa bunu kisa bicimde belirt.

Cevap:
"""
