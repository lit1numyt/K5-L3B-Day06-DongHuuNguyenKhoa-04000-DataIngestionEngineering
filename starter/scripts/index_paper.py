"""Create a page-level, provenance-preserving index for the lab paper."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

from pypdf import PdfReader


DOCUMENT_ID = "automated-software-vulnerability-detection-survey-v1"
TITLE = "A Survey on Automated Software Vulnerability Detection Using Machine Learning and Deep Learning"


def normalize_text(text: str) -> str:
    """Normalize Unicode/whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = re.sub(r"[ \t]+", " ", paragraph)
        paragraph = re.sub(r" *\n+ *", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def section_for_page(page: int) -> str:
    if page == 1:
        return "abstract"
    if page == 2:
        return "introduction"
    if page in {3, 4, 5}:
        return "background-and-related-work"
    if page in {6, 7, 8}:
        return "related-work"
    if page in {9, 10}:
        return "methodology"
    if page in {11, 12}:
        return "results-rq1"
    if page in {13, 14, 15, 16, 17, 18, 19, 20, 21}:
        return "results-rq2"
    if page in {22, 23, 24, 25}:
        return "results-rq3"
    if page in {26, 27}:
        return "results-rq4"
    if page in {28, 29}:
        return "results-rq5"
    if page == 30:
        return "threats-and-conclusion"
    return "references"


def build_index(pdf_path: Path, index_path: Path, manifest_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    records = []
    content_pages = 0

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        normalized = normalize_text(raw_text)
        eligible = page_number <= 30 and bool(normalized)
        if eligible:
            content_pages += 1
        records.append(
            {
                "document_id": DOCUMENT_ID,
                "source": str(pdf_path),
                "page": page_number,
                "section": section_for_page(page_number),
                "eligible_content": eligible,
                "raw_character_count": len(raw_text),
                "character_count": len(normalized),
                "text": normalized,
            }
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source_id": DOCUMENT_ID,
        "title": TITLE,
        "source_path": str(pdf_path),
        "source_snapshot": "arXiv:2306.11673v1",
        "source_url": "https://arxiv.org/abs/2306.11673",
        "license_or_permission": "Verify redistribution and reuse terms for the bundled arXiv source before submission.",
        "accessed_at": date.today().isoformat(),
        "physical_page_count": len(reader.pages),
        "eligible_content_pages": list(range(1, 31)),
        "excluded_pages": list(range(31, len(reader.pages) + 1)),
        "page_count_rule": "Count physical pages 1-30 containing the article; exclude reference-only pages 31-37.",
        "index_path": str(index_path),
        "section_field": {
            "name": "section",
            "source": "Verified from the PDF headings and the bundled TeX section structure.",
            "purpose": "Metadata filter for questions that explicitly constrain retrieval to a paper section.",
        },
        "content_page_count": content_pages,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build_index(args.pdf, args.index, args.manifest)


if __name__ == "__main__":
    main()
