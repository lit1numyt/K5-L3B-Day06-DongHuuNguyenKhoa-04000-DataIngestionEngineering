"""Lightweight lexical reranking for dense retrieval candidates."""
from __future__ import annotations

import re
from collections.abc import Iterable


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "use", "used", "was", "what", "were", "which", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _lexical_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(text)) / len(query_tokens)


def rerank(query: str, hits: Iterable[dict], lexical_weight: float) -> list[dict]:
    """Combine normalized dense score with query-term coverage.

    The input is intentionally a small dense candidate set. Lexical coverage is
    used as a tie-breaker/precision signal, not as a replacement for embeddings.
    """
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("lexical_weight must be between 0 and 1")
    candidates = [dict(hit) for hit in hits]
    if not candidates:
        return []
    dense_scores = [float(hit.get("score", 0.0)) for hit in candidates]
    low, high = min(dense_scores), max(dense_scores)
    span = high - low
    query_tokens = _tokens(query)
    for hit, dense_score in zip(candidates, dense_scores):
        dense_norm = 1.0 if span == 0 else (dense_score - low) / span
        lexical_score = _lexical_score(query_tokens, hit.get("text", ""))
        hit["dense_score"] = dense_score
        hit["lexical_score"] = round(lexical_score, 6)
        hit["score"] = round(
            (1.0 - lexical_weight) * dense_norm + lexical_weight * lexical_score,
            6,
        )
    return sorted(candidates, key=lambda hit: hit["score"], reverse=True)
