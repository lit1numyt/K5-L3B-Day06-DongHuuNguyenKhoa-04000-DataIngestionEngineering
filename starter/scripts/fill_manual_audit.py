"""Fill the frozen retrieval audit with transparent author-assisted labels."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


COHERENCE_SCALE = "0=broken or missing context; 1=partly understandable; 2=self-contained"
GROUNDING_SCALE = "0=no evidence; 1=partial evidence; 2=sufficient evidence and provenance"
METADATA_SCALE = "1=source/page/section metadata sufficient and correct; 0=metadata error"
SCORER = "assistant-assisted author review"

# The same ten fixed page positions are reviewed for each strategy.
COHERENCE_SCORES = {
    "A": {page: 1 for page in range(1, 11)},
    "B": {page: 1 for page in range(1, 11)},
    "C": {page: 1 for page in range(1, 11)},
}
for strategy in ("B", "C"):
    COHERENCE_SCORES[strategy][6] = 2

GROUNDING_SCORES = {
    "A": {"q16": 2, "q17": 2, "q18": 1, "q19": 1, "q20": 1},
    "B": {"q16": 2, "q17": 1, "q18": 1, "q19": 1, "q20": 1},
    "C": {"q16": 2, "q17": 1, "q18": 1, "q19": 1, "q20": 2},
}


def coherence_reason(score: int, page: int, strategy: str) -> str:
    if score == 2:
        return "The table and surrounding explanation remain understandable without needing the adjacent page."
    if page in {1, 2, 3, 4, 5, 7, 8, 9, 10}:
        return "The chunk is understandable but starts or ends at a page boundary, transition, or sentence fragment and needs nearby context."
    return f"Strategy {strategy} begins with a table/page fragment; the retrieved text is understandable but not fully self-contained."


def grounding_reason(question_id: str, strategy: str, score: int) -> str:
    if question_id == "q16" and score == 2:
        return "Top-3 contains the Word2vec and graph-embedding result with the 25.4% share, even when the exact gold page is not rank 1."
    if question_id == "q17" and score == 2:
        return "Top-3 includes the model-distribution evidence needed to compare DL and classic ML percentages."
    if question_id == "q20" and score == 2:
        return "The filtered top-3 covers the challenge list across the page-28 and page-29 evidence."
    if score == 1:
        return "Top-3 contains related evidence, but it misses a required value, subtype, or complete condition for the question."
    return "No relevant evidence was retrieved."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-path", type=Path, default=None)
    args = parser.parse_args()
    path = args.audit_path or (Path(__file__).resolve().parents[1] / "results" / "manual_audit.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    extra_columns = ["scorer", "label_scale"]
    fieldnames = list(rows[0])
    for column in extra_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        strategy = row["strategy"]
        row["scorer"] = SCORER
        if row["audit_type"] == "coherence":
            page = int(row["source_page"])
            score = COHERENCE_SCORES[strategy][page]
            row["label"] = score
            row["label_scale"] = COHERENCE_SCALE
            row["reason"] = coherence_reason(score, page, strategy)
        elif row["audit_type"] == "metadata":
            row["label"] = 1
            row["label_scale"] = METADATA_SCALE
            row["reason"] = "Source, page, and section metadata are present and match the indexed source page."
        else:
            question_id = row["question_id"]
            score = GROUNDING_SCORES[strategy][question_id]
            row["label"] = score
            row["label_scale"] = GROUNDING_SCALE
            row["reason"] = grounding_reason(question_id, strategy, score)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Filled {len(rows)} audit rows in {path}")


if __name__ == "__main__":
    main()
