"""
tests/test_chunking_service.py — Unit tests for the text chunking service.
"""

from __future__ import annotations

from app.services.chunking_service import (
    TextChunk,
    _split_into_paragraphs,
    chunk_pages,
    chunk_text,
)
from app.services.pdf_service import PageData


# ------------------------------------------------------------------ #
# Paragraph splitting
# ------------------------------------------------------------------ #
def test_split_simple_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    result = _split_into_paragraphs(text)
    assert result == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_split_strips_whitespace():
    text = "  First.  \n\n  Second.  "
    result = _split_into_paragraphs(text)
    assert result == ["First.", "Second."]


def test_split_filters_empty_paragraphs():
    text = "First.\n\n\n\n\nSecond."
    result = _split_into_paragraphs(text)
    assert len(result) == 2


# ------------------------------------------------------------------ #
# chunk_text
# ------------------------------------------------------------------ #
def test_chunk_text_basic():
    text = "First para.\n\nSecond para.\n\nThird para."
    chunks = chunk_text(text, document_id="doc-1", chunk_size=50, overlap=0)
    assert len(chunks) >= 1
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_text_respects_size():
    # Make a very large paragraph and chunk with small limit
    long_text = "word " * 500
    chunks = chunk_text(long_text, document_id="doc-1", chunk_size=100, overlap=0)
    for chunk in chunks:
        assert chunk.char_count <= 200  # Allow some slack for overlap flush


def test_chunk_text_preserves_document_id():
    chunks = chunk_text("Para one.\n\nPara two.", document_id="test-doc-999")
    for c in chunks:
        assert c.document_id == "test-doc-999"


def test_chunk_text_indices_are_sequential():
    text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
    chunks = chunk_text(text, document_id="doc-1", chunk_size=100, overlap=0)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_text_with_overlap_produces_more_chunks():
    text = "\n\n".join([f"Para {i} with some content." for i in range(10)])
    without_overlap = chunk_text(text, document_id="doc-1", chunk_size=80, overlap=0)
    with_overlap = chunk_text(text, document_id="doc-1", chunk_size=80, overlap=40)
    # With overlap we should get at least as many chunks
    assert len(with_overlap) >= len(without_overlap)


def test_chunk_ids_are_unique():
    text = "\n\n".join([f"Para {i}." for i in range(10)])
    chunks = chunk_text(text, document_id="doc-1")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ #
# chunk_pages
# ------------------------------------------------------------------ #
def test_chunk_pages_skips_empty_pages():
    pages = [
        PageData(page_number=1, text=""),
        PageData(page_number=2, text="Real content here."),
    ]
    chunks = chunk_pages(pages, document_id="doc-1")
    # All chunks should come from page 2
    for c in chunks:
        assert c.page_number == 2


def test_chunk_pages_preserves_page_numbers():
    pages = [
        PageData(page_number=1, text="Page one content."),
        PageData(page_number=2, text="Page two content."),
    ]
    chunks = chunk_pages(pages, document_id="doc-1", chunk_size=30, overlap=0)
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
