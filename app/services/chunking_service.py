"""
services/chunking_service.py — Paragraph-aware text chunking.

Splits the extracted PDF text into overlapping chunks that:
  - Respect paragraph boundaries where possible.
  - Stay within the configured character limit.
  - Carry document_id, page_number, and chunk_index metadata.
  - Overlap consecutive chunks to preserve cross-boundary context.

Design rationale:
  Simple paragraph splitting (double-newline) is chosen over sentence-based
  splitting because academic/technical PDFs often have meaningful paragraph
  breaks.  Overlap prevents the model from missing context at chunk edges.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass

from app.config import get_settings
from app.services.pdf_service import PageData

logger = logging.getLogger(__name__)
settings = get_settings()


# ------------------------------------------------------------------ #
# Data class
# ------------------------------------------------------------------ #
@dataclass
class TextChunk:
    """A single chunk of text with associated metadata."""

    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int | None
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #
def _split_into_paragraphs(text: str) -> list[str]:
    """
    Split text on double newlines (paragraph breaks).

    Each returned paragraph is stripped and non-empty.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _build_chunks_from_paragraphs(
    paragraphs: list[str],
    document_id: str,
    page_number: int | None,
    start_index: int,
    chunk_size: int,
    overlap: int,
) -> tuple[list[TextChunk], int]:
    """
    Greedily pack paragraphs into chunks of at most `chunk_size` characters,
    then slide back by `overlap` characters to create the next chunk boundary.

    Returns (list_of_chunks, next_start_index).
    """
    chunks: list[TextChunk] = []
    current_text = ""
    idx = start_index

    for para in paragraphs:
        # If a single paragraph exceeds chunk_size, split it into pieces.
        while len(para) > chunk_size:
            if current_text:
                chunks.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_index=idx,
                        page_number=page_number,
                        text=current_text.strip(),
                    )
                )
                idx += 1
                current_text = ""

            sub_slice = para[:chunk_size]
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=idx,
                    page_number=page_number,
                    text=sub_slice.strip(),
                )
            )
            idx += 1
            step = chunk_size - overlap if (chunk_size - overlap) > 0 else chunk_size
            para = para[step:]

        if not para.strip():
            continue

        # If adding this paragraph would exceed the limit, flush first.
        if current_text and len(current_text) + len(para) + 2 > chunk_size:
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=idx,
                    page_number=page_number,
                    text=current_text.strip(),
                )
            )
            idx += 1
            # Overlap: carry the tail of the previous chunk into the next.
            current_text = current_text[-overlap:] if overlap else ""

        current_text = (current_text + "\n\n" + para).strip() if current_text else para

    # Flush the last partial chunk.
    if current_text.strip():
        chunks.append(
            TextChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                page_number=page_number,
                text=current_text.strip(),
            )
        )
        idx += 1

    return chunks, idx


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def chunk_pages(
    pages: list[PageData],
    document_id: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[TextChunk]:
    """
    Convert a list of PageData objects into overlapping text chunks.

    Parameters
    ----------
    pages:       Extracted page objects from pdf_service.
    document_id: UUID string of the parent document (stored as metadata).
    chunk_size:  Maximum character count per chunk (defaults to settings).
    overlap:     Character overlap between consecutive chunks (defaults to settings).

    Returns
    -------
    Ordered list of TextChunk objects ready for embedding.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP

    all_chunks: list[TextChunk] = []
    global_index = 0

    for page in pages:
        if page.is_empty():
            continue

        paragraphs = _split_into_paragraphs(page.text)
        page_chunks, global_index = _build_chunks_from_paragraphs(
            paragraphs=paragraphs,
            document_id=document_id,
            page_number=page.page_number,
            start_index=global_index,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(page_chunks)

    logger.info(
        "Chunked document %s: %d pages → %d chunks (size=%d, overlap=%d)",
        document_id,
        len(pages),
        len(all_chunks),
        chunk_size,
        overlap,
    )
    return all_chunks


def chunk_text(
    text: str,
    document_id: str,
    page_number: int | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[TextChunk]:
    """
    Chunk a raw string (convenience wrapper, e.g. for tests).

    Splits into paragraphs then applies the same packing logic.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
    paragraphs = _split_into_paragraphs(text)
    chunks, _ = _build_chunks_from_paragraphs(
        paragraphs=paragraphs,
        document_id=document_id,
        page_number=page_number,
        start_index=0,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunks
