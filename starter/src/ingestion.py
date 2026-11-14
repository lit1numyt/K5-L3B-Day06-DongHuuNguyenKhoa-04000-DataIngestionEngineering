"""Read and normalize source data while preserving page provenance."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


PAPER_TITLE_PREFIX = "A Survey on Automated Software Vulnerability Detection"
AUTHOR_HEADER = "Nima Shiri harzevili, Alvine Boaye Belle, Junjie Wang, Song Wang, Zhen Ming (Jack) Jiang, and Nachiappan Nagappan"
JOURNAL_FOOTER_PREFIX = "J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018."


def clean_page_artifacts(text: str, page_number: int) -> str:
    """Remove repeated PDF running headers/footers without removing article content."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"111", f"111:{page_number}"}:
            continue
        stripped = re.sub(r"^111:\d+\s+", "", stripped)
        if stripped.startswith(JOURNAL_FOOTER_PREFIX):
            continue
        if page_number > 1 and (
            stripped.startswith(PAPER_TITLE_PREFIX)
            or stripped == "Using Machine Learning and Deep Learning"
            or stripped == AUTHOR_HEADER
            or AUTHOR_HEADER in stripped
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page and retain empty-page diagnostics in metadata."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        raw_text = pdf_page.extract_text() or ""
        text = normalize_text(clean_page_artifacts(raw_text, page_number))
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "raw_character_count": len(raw_text),
                    "character_count": len(text),
                    "extraction_status": "ok" if text else "empty",
                },
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = re.sub(r"[ \t]+", " ", paragraph)
        paragraph = re.sub(r" *\n+ *", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)
