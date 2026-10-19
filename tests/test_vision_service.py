"""
tests/test_vision_service.py — Unit tests for multimodal OCR and visual diagram intelligence.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from app.services.pdf_service import PageData
from app.services.vision_service import (
    _prepare_image,
    describe_image,
    enrich_pdf_pages_with_vision,
    ocr_scanned_page,
)


def test_prepare_image_resizes_and_converts():
    # Large grayscale image
    img = Image.new("L", (1600, 1200), color=128)
    processed = _prepare_image(img)

    assert processed.mode == "RGB"
    assert processed.size[0] <= 800
    assert processed.size[1] <= 800


def test_describe_image_with_mock():
    img = Image.new("RGB", (300, 300), color="blue")

    mock_response = MagicMock()
    mock_response.text = "A blue square diagram illustrating network topology."

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        desc = describe_image(img, page_number=1)
        assert desc is not None
        assert "blue square" in desc


def test_describe_image_handles_error():
    img = Image.new("RGB", (200, 200), color="red")

    with patch("google.generativeai.GenerativeModel", side_effect=RuntimeError("API error")):
        desc = describe_image(img, page_number=1)
        assert desc is None


def test_ocr_scanned_page_with_mock():
    img = Image.new("RGB", (400, 600), color="white")

    mock_response = MagicMock()
    mock_response.text = "Chapter 1: Principles of Economics\n1. People face trade-offs."

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        text = ocr_scanned_page(img, page_number=1)
        assert text is not None
        assert "Principles of Economics" in text


def test_enrich_pdf_pages_with_vision_scanned_ocr(tmp_path):
    import pymupdf

    # Create a dummy PDF with 1 blank page
    pdf_path = str(tmp_path / "scanned.pdf")
    doc = pymupdf.open()
    doc.new_page(width=300, height=300)
    doc.save(pdf_path)
    doc.close()

    pages = [PageData(page_number=1, text="")]  # Empty text simulates scan

    with patch(
        "app.services.vision_service.ocr_scanned_page",
        return_value="Extracted scanned text from document page.",
    ):
        enriched = enrich_pdf_pages_with_vision(pdf_path, pages)
        assert len(enriched) == 1
        assert "Extracted scanned text" in enriched[0].text
        assert enriched[0].char_count > 0


def test_enrich_pdf_pages_with_vision_diagram(tmp_path):
    import pymupdf

    pdf_path = str(tmp_path / "diagram.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)

    # Insert an embedded image >= 120x120
    test_img = Image.new("RGB", (200, 200), color="green")
    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 50, 250, 250), stream=buf.getvalue())
    doc.save(pdf_path)
    doc.close()

    pages = [PageData(page_number=1, text="Regular paragraph text explaining concepts.")]

    with patch(
        "app.services.vision_service.describe_image",
        return_value="A flowchart displaying three-tier system architecture.",
    ):
        enriched = enrich_pdf_pages_with_vision(pdf_path, pages)
        assert len(enriched) == 1
        assert "Regular paragraph text" in enriched[0].text
        assert (
            "[Figure on Page 1: A flowchart displaying three-tier system architecture.]"
            in enriched[0].text
        )
