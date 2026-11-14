"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into deterministic overlapping character windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if not page.text:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(page.text), step)):
        text = page.text[start : start + size].strip()
        if not text:
            continue
        metadata = dict(page.metadata)
        metadata.update(
            {
                "chunking": "fixed_size",
                "chunk_index": index,
                "start_char": start,
                "end_char": min(start + size, len(page.text)),
            }
        )
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:p{page.page}:fixed:{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text,
                metadata=metadata,
            )
        )
        if start + size >= len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to ``max_size`` and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    grouped: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                grouped.append(current)
                current = ""
            grouped.extend(
                paragraph[start : start + max_size].strip()
                for start in range(0, len(paragraph), max_size)
                if paragraph[start : start + max_size].strip()
            )
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_size:
            current = f"{current}\n\n{paragraph}"
        else:
            grouped.append(current)
            current = paragraph
    if current:
        grouped.append(current)

    merged: list[str] = []
    for text in grouped:
        if (
            merged
            and len(text) < 100
            and len(merged[-1]) + 2 + len(text) <= max_size
        ):
            merged[-1] = f"{merged[-1]}\n\n{text}"
        else:
            merged.append(text)
    grouped = merged

    chunks: list[Chunk] = []
    for index, text in enumerate(grouped):
        metadata = dict(page.metadata)
        metadata.update({"chunking": "by_paragraph", "chunk_index": index})
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:p{page.page}:paragraph:{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text,
                metadata=metadata,
            )
        )
    return chunks
