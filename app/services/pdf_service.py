"""
services/pdf_service.py — PDF validation and text extraction.

Responsibilities:
  - Validate uploaded file (extension, MIME type, file size, PDF header).
  - Extract text page by page using pypdf.
  - Preserve page numbers as metadata.
  - Detect and report empty or scanned PDFs rather than silently indexing nothing.
  - Never execute content from the uploaded document.

Security note:
  Uploaded files are treated as untrusted input.  We only read text from them;
  no document content is used as instructions to any system component.
"""

from __future__ import annotations

import io
import logging
import os
import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
PDF_MAGIC_BYTES = b"%PDF"


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #
class PageData:
    """Text and metadata extracted from a single PDF page."""

    __slots__ = ("char_count", "page_number", "text")

    def __init__(self, page_number: int, text: str) -> None:
        self.page_number = page_number
        self.text = text.strip()
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        return self.char_count == 0


class ExtractionResult:
    """Result of a full PDF extraction."""

    def __init__(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        pages: list[PageData],
    ) -> None:
        self.document_id = document_id
        self.filename = filename
        self.file_path = file_path
        self.file_size_bytes = file_size_bytes
        self.pages = pages

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def non_empty_pages(self) -> list[PageData]:
        return [p for p in self.pages if not p.is_empty()]

    @property
    def total_text(self) -> str:
        return "\n\n".join(p.text for p in self.non_empty_pages)

    @property
    def is_empty(self) -> bool:
        return len(self.non_empty_pages) == 0


# ------------------------------------------------------------------ #
# Validation helpers
# ------------------------------------------------------------------ #
def _validate_extension(filename: str) -> None:
    """Raise HTTPException if the filename does not end with .pdf."""
    if not filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only PDF files are accepted (.pdf). Received: '{filename}'.",
        )


def _validate_content_type(content_type: str | None) -> None:
    """Raise HTTPException if the declared MIME type is not a PDF type."""
    if content_type and content_type.split(";")[0].strip() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid content type '{content_type}'. Expected application/pdf.",
        )


def _validate_size(data: bytes) -> None:
    """Raise HTTPException if the file exceeds the configured size limit."""
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(f"File exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB."),
        )


def _validate_pdf_header(data: bytes) -> None:
    """Raise HTTPException if the byte stream does not start with the PDF magic bytes."""
    if not data.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file does not appear to be a valid PDF (bad magic bytes).",
        )


def _sanitize_filename(original: str) -> str:
    """
    Return a safe filename: UUID prefix + sanitized original name.
    Strips path separators and limits length.
    """
    name = Path(original).name  # strip directory traversal
    name = re.sub(r"[^\w.\-]", "_", name)  # replace unsafe chars
    name = name[:200]  # limit length
    return f"{uuid.uuid4().hex[:8]}_{name}"


# ------------------------------------------------------------------ #
# Save file to disk
# ------------------------------------------------------------------ #
def _save_file(data: bytes, filename: str) -> str:
    """Persist the PDF bytes to the uploads directory and return the path."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(data)
    logger.info("Saved PDF to %s (%d bytes)", file_path, len(data))
    return str(file_path)


# ------------------------------------------------------------------ #
# PDF text extraction
# ------------------------------------------------------------------ #
def _extract_pages(data: bytes) -> list[PageData]:
    """
    Extract text from each page using pypdf.

    Returns a list of PageData objects (one per page).  Pages with no
    extractable text are included with empty text so the caller can detect
    scanned / image-only PDFs.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse the PDF file: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error reading PDF")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unexpected error reading PDF: {exc}",
        ) from exc

    pages: list[PageData] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from page %d: %s", i, exc)
            text = ""
        pages.append(PageData(page_number=i, text=text))

    return pages


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
async def process_upload(upload_file: UploadFile) -> ExtractionResult:
    """
    Validate, save, and extract text from an uploaded PDF.

    Steps:
      1. Validate extension and declared content type.
      2. Read the file bytes into memory.
      3. Validate file size and PDF magic bytes.
      4. Save to the uploads directory with a safe filename.
      5. Extract text page by page.
      6. Warn (not error) if the PDF has no extractable text.

    Raises HTTPException for any validation or parsing failure.
    """
    _validate_extension(upload_file.filename or "")
    _validate_content_type(upload_file.content_type)

    data = await upload_file.read()

    _validate_size(data)
    _validate_pdf_header(data)

    safe_name = _sanitize_filename(upload_file.filename or "upload.pdf")
    file_path = _save_file(data, safe_name)
    doc_id = str(uuid.uuid4())

    pages = _extract_pages(data)

    result = ExtractionResult(
        document_id=doc_id,
        filename=safe_name,
        file_path=file_path,
        file_size_bytes=len(data),
        pages=pages,
    )

    if result.is_empty:
        logger.warning(
            "PDF '%s' has no extractable text — it may be scanned or image-only.",
            upload_file.filename,
        )

    logger.info(
        "Processed PDF '%s': %d pages, %d non-empty",
        upload_file.filename,
        result.page_count,
        len(result.non_empty_pages),
    )
    return result


def load_pdf_from_disk(file_path: str) -> ExtractionResult:
    """
    Re-extract text from a previously saved PDF on disk.

    Useful for re-indexing without requiring the user to re-upload.
    """
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {file_path}",
        )
    data = path.read_bytes()
    pages = _extract_pages(data)
    return ExtractionResult(
        document_id="",
        filename=path.name,
        file_path=file_path,
        file_size_bytes=len(data),
        pages=pages,
    )


def delete_pdf_file(file_path: str) -> bool:
    """
    Delete the PDF file from disk.  Returns True if deleted, False if not found.
    Safe to call even if the file no longer exists.
    """
    try:
        os.remove(file_path)
        logger.info("Deleted PDF file: %s", file_path)
        return True
    except FileNotFoundError:
        logger.warning("PDF file not found during deletion: %s", file_path)
        return False
    except OSError as exc:
        logger.error("Error deleting PDF file %s: %s", file_path, exc)
        return False
