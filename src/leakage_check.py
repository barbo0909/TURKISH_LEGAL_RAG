from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


TOKEN_RE = re.compile(r"[0-9A-Za-z\u00c7\u011e\u0130\u00d6\u015e\u00dc\u00e7\u011f\u0131\u00f6\u015f\u00fc]+")


def normalize_for_leakage(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = value.replace("ı", "i")
    return " ".join(TOKEN_RE.findall(value))


def similarity(a: str, b: str) -> float:
    left = normalize_for_leakage(a)
    right = normalize_for_leakage(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def max_similarity(text: str, candidates: list[str]) -> tuple[float, str]:
    best_score = 0.0
    best_candidate = ""
    for candidate in candidates:
        score = similarity(text, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_score, best_candidate
