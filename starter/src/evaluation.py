"""Evaluate retrieval evidence and preserve per-query traces."""
import re
from typing import Any


def _normalize_for_match(value: str) -> str:
    value = value.casefold().replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    source_match = hit.get("document_id") == question.get("gold_source_id") or (
        hit.get("metadata", {}).get("document_id") == question.get("gold_source_id")
    )
    page_match = hit.get("page") == question.get("gold_page")
    evidence_options = [question.get("evidence_quote", "")]
    evidence_options.extend(question.get("evidence_quotes", []))
    text = _normalize_for_match(hit.get("text", ""))
    evidence_match = any(
        _normalize_for_match(quote) in text for quote in evidence_options if quote
    )
    return bool(source_match and page_match and evidence_match)


def _filter_match(filters: dict[str, Any] | None, hit: dict[str, Any]) -> bool:
    if not filters:
        return True
    metadata = hit.get("metadata", {})
    return all(str(metadata.get(key)) == str(value) for key, value in filters.items())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check gold source/page/evidence for Hit@1 and Evidence Hit@3."""
    hit_flags = [_hit(question, hit) for hit in hits]
    filters = question.get("filter")
    filter_flags = [_filter_match(filters, hit) for hit in hits]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "gold_source_id": question.get("gold_source_id"),
        "gold_page": question.get("gold_page"),
        "filter": question.get("filter"),
        "hit_at_1": bool(hit_flags and hit_flags[0]),
        "evidence_hit_at_3": any(hit_flags[:3]),
        "filter_compliant_top3": None if not filters else bool(hits and all(filter_flags[:3])),
        "filter_compliant_hits": None if not filters else sum(filter_flags[:3]),
        "hits": [
            {**hit, "gold_hit": hit_flags[index]}
            for index, hit in enumerate(hits)
        ],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval metrics by strategy and dev/test split."""
    summary: dict[str, Any] = {}
    for trace in traces:
        strategy = trace["strategy"]
        split = trace["split"]
        bucket = summary.setdefault(strategy, {}).setdefault(
            split,
            {"count": 0, "hit_at_1": 0, "evidence_hit_at_3": 0},
        )
        bucket["count"] += 1
        bucket["hit_at_1"] += int(trace["hit_at_1"])
        bucket["evidence_hit_at_3"] += int(trace["evidence_hit_at_3"])
        if trace["filter_compliant_top3"] is not None:
            bucket.setdefault("filter_queries", 0)
            bucket.setdefault("filter_compliant_top3", 0)
            bucket["filter_queries"] += 1
            bucket["filter_compliant_top3"] += int(trace["filter_compliant_top3"])
    for strategy_buckets in summary.values():
        for bucket in strategy_buckets.values():
            total = bucket["count"]
            bucket["hit_at_1_fraction"] = f"{bucket['hit_at_1']}/{total}"
            bucket["evidence_hit_at_3_fraction"] = f"{bucket['evidence_hit_at_3']}/{total}"
            if bucket.get("filter_queries"):
                bucket["filter_compliance_fraction"] = (
                    f"{bucket['filter_compliant_top3']}/{bucket['filter_queries']}"
                )
    return summary
