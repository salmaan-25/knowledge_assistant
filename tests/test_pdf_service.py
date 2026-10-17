"""
tests/test_pdf_service.py — Unit tests for PDF extraction and validation.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.services.pdf_service import (
    ExtractionResult,
    PageData,
    _extract_pages,
    _sanitize_filename,
    _validate_extension,
    _validate_pdf_header,
    _validate_size,
)

settings = get_settings()


# ------------------------------------------------------------------ #
# Unit tests — extension validation
# ------------------------------------------------------------------ #
def test_valid_extension_passes():
    _validate_extension("my_notes.pdf")  # should not raise


def test_invalid_extension_raises():
    with pytest.raises(HTTPException) as exc_info:
        _validate_extension("my_notes.docx")
    assert exc_info.value.status_code == 422


def test_uppercase_extension_rejected():
    with pytest.raises(HTTPException):
        _validate_extension("MY_NOTES.PDF")  # .PDF not .pdf → rejected


# ------------------------------------------------------------------ #
# Unit tests — size validation
# ------------------------------------------------------------------ #
def test_size_within_limit_passes():
    data = b"x" * 1024  # 1 KB — well within limit
    _validate_size(data)  # should not raise


def test_size_over_limit_raises():
    limit_bytes = settings.max_upload_bytes
    data = b"x" * (limit_bytes + 1)
    with pytest.raises(HTTPException) as exc_info:
        _validate_size(data)
    assert exc_info.value.status_code == 413


# ------------------------------------------------------------------ #
# Unit tests — PDF header validation
# ------------------------------------------------------------------ #
def test_valid_pdf_header_passes():
    _validate_pdf_header(b"%PDF-1.4 rest of content")  # should not raise


def test_invalid_header_raises():
    with pytest.raises(HTTPException) as exc_info:
        _validate_pdf_header(b"PK\x03\x04 zip header")
    assert exc_info.value.status_code == 422


# ------------------------------------------------------------------ #
# Unit tests — filename sanitisation
# ------------------------------------------------------------------ #
def test_sanitize_normal_filename():
    result = _sanitize_filename("my notes.pdf")
    assert result.endswith("my_notes.pdf")


def test_sanitize_removes_path_separators():
    result = _sanitize_filename("../../etc/passwd.pdf")
    assert ".." not in result
    assert "/" not in result


def test_sanitize_limits_length():
    long_name = "a" * 300 + ".pdf"
    result = _sanitize_filename(long_name)
    assert len(result) <= 250  # UUID prefix + max 200 chars


# ------------------------------------------------------------------ #
# Unit tests — PageData
# ------------------------------------------------------------------ #
def test_page_data_not_empty_with_text():
    p = PageData(page_number=1, text="  Some text  ")
    assert not p.is_empty()
    assert p.char_count > 0


def test_page_data_empty_with_whitespace():
    p = PageData(page_number=1, text="   \n\n  ")
    assert p.is_empty()
    assert p.char_count == 0


# ------------------------------------------------------------------ #
# Unit tests — ExtractionResult
# ------------------------------------------------------------------ #
def test_extraction_result_total_text():
    pages = [
        PageData(page_number=1, text="First page content."),
        PageData(page_number=2, text="Second page content."),
    ]
    result = ExtractionResult(
        document_id="doc-1",
        filename="test.pdf",
        file_path="/uploads/test.pdf",
        file_size_bytes=1024,
        pages=pages,
    )
    assert result.page_count == 2
    assert not result.is_empty
    assert "First page" in result.total_text
    assert "Second page" in result.total_text


def test_extraction_result_is_empty_when_all_pages_empty():
    pages = [PageData(page_number=1, text="")]
    result = ExtractionResult(
        document_id="doc-2",
        filename="empty.pdf",
        file_path="/uploads/empty.pdf",
        file_size_bytes=100,
        pages=pages,
    )
    assert result.is_empty


# ------------------------------------------------------------------ #
# Integration test — extract from minimal PDF bytes
# ------------------------------------------------------------------ #
def test_extract_pages_from_valid_pdf(sample_pdf_bytes):
    pages = _extract_pages(sample_pdf_bytes)
    assert len(pages) >= 1
    assert isinstance(pages[0], PageData)
    assert pages[0].page_number == 1


def test_extract_pages_from_corrupt_bytes():
    with pytest.raises(HTTPException) as exc_info:
        _extract_pages(b"%PDF-1.4 this is not a valid pdf")
    assert exc_info.value.status_code == 422
