"""Prepare blank manual-audit rows from the current retrieval artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import by_paragraph, fixed_size
from src.models import DocumentPage


def load_pages(path: Path) -> list[DocumentPage]:
    pages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["eligible_content"]:
            pages.append(
                DocumentPage(
                    document_id=record["document_id"],
                    source=record["source"],
                    page=record["page"],
                    text=record["text"],
                    metadata={"section": record["section"]},
                )
            )
    return pages


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    pages = load_pages(repo / "data" / "paper_pages.jsonl")
    run_metadata = json.loads((root / "results" / "run_metadata.json").read_text())
    traces = {
        strategy: [
            json.loads(line)
            for line in (root / "results" / f"traces_{strategy}.jsonl").read_text().splitlines()
        ]
        for strategy in ("A", "B", "C")
    }
    question_by_id = {
        question["id"]: question
        for question in json.loads((repo / "benchmark" / "questions.json").read_text())
    }
    test_ids = [question_id for question_id, question in question_by_id.items() if question["split"] == "test"]
    configs = run_metadata["strategies"]
    chunks_by_strategy = {
        "A": [
            chunk
            for page in pages
            for chunk in fixed_size(page, configs["A"]["size"], configs["A"]["overlap"])
        ],
        "B": [
            chunk
            for page in pages
            for chunk in by_paragraph(page, configs["B"]["max_size"])
        ],
    }
    chunks_by_strategy["C"] = chunks_by_strategy["B"]

    rows = []
    for page in pages[:10]:
        for strategy in ("A", "B", "C"):
            page_chunks = [chunk for chunk in chunks_by_strategy[strategy] if chunk.page == page.page]
            if not page_chunks:
                continue
            chunk = page_chunks[0]
            rows.append(
                {
                    "audit_type": "coherence",
                    "strategy": strategy,
                    "item_id": f"page-{page.page}",
                    "question_id": "",
                    "question": "",
                    "expected_answer": "",
                    "gold_page": page.page,
                    "gold_evidence": "",
                    "rank": "",
                    "source_page": page.page,
                    "chunk_id": chunk.chunk_id,
                    "score": "",
                    "text": chunk.text,
                    "metadata": json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    "requested_filter": "",
                    "filter_compliant": "",
                    "label": "",
                    "reason": "",
                }
            )

    for question_id in test_ids:
        question = question_by_id[question_id]
        for strategy in ("A", "B", "C"):
            trace = next(item for item in traces[strategy] if item["question_id"] == question_id)
            for rank, hit in enumerate(trace["hits"], start=1):
                rows.append(
                    {
                        "audit_type": "metadata",
                        "strategy": strategy,
                        "item_id": question_id,
                        "question_id": question_id,
                        "question": question["question"],
                        "expected_answer": question["expected_answer"],
                        "gold_page": question["gold_page"],
                        "gold_evidence": question["evidence_quote"],
                        "rank": rank,
                        "source_page": hit["page"],
                        "chunk_id": hit["chunk_id"],
                        "score": hit["score"],
                        "text": hit["text"],
                        "metadata": json.dumps(hit["metadata"], ensure_ascii=False, sort_keys=True),
                        "requested_filter": json.dumps(question.get("filter"), ensure_ascii=False),
                        "filter_compliant": trace["filter_compliant_top3"],
                        "label": "",
                        "reason": "Check source, page, section, and filter correctness against the paper.",
                    }
                )
            rows.append(
                {
                    "audit_type": "grounding",
                    "strategy": strategy,
                    "item_id": question_id,
                    "question_id": question_id,
                    "question": question["question"],
                    "expected_answer": question["expected_answer"],
                    "gold_page": question["gold_page"],
                    "gold_evidence": question["evidence_quote"],
                    "rank": "",
                    "source_page": question["gold_page"],
                    "chunk_id": "",
                    "score": "",
                    "text": json.dumps(
                        [
                            {"rank": rank, "chunk_id": hit["chunk_id"], "page": hit["page"], "text": hit["text"]}
                            for rank, hit in enumerate(trace["hits"], start=1)
                        ],
                        ensure_ascii=False,
                    ),
                    "metadata": "",
                    "requested_filter": json.dumps(question.get("filter"), ensure_ascii=False),
                    "filter_compliant": trace["filter_compliant_top3"],
                    "label": "",
                    "reason": "Score 0=no evidence, 1=partial evidence, 2=sufficient evidence and provenance.",
                }
            )

    output = root / "results" / "manual_audit.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} blank audit rows to {output}")


if __name__ == "__main__":
    main()
