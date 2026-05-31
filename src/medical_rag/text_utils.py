from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def lexical_overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)
