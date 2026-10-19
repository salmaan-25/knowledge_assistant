"""
services/vision_service.py — Multimodal document intelligence and OCR.

Enriches extracted PDF pages with:
1. High-accuracy OCR for scanned or image-only pages using Gemini Vision.
2. Semantic descriptions for embedded charts, diagrams, architectures, and figures.

Ensures that RAG retrieval and Q&A can answer questions about visual content,
charts, and scanned notes without requiring external C++ OCR binaries.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import google.generativeai as genai
from PIL import Image

from app.config import get_settings

if TYPE_CHECKING:
    from app.services.pdf_service import PageData

logger = logging.getLogger(__name__)
settings = get_settings()

# ------------------------------------------------------------------ #
# Visual processing limits (optimizes latency, tokens & cost)
# ------------------------------------------------------------------ #
MIN_IMAGE_DIMENSION = 120  # Filter out tiny icons, bullets, spacers
MAX_IMAGES_PER_PAGE = 2  # At most 2 key figures analyzed per page
MAX_VISUAL_PAGES_PER_DOC = 12  # Cap on total pages analyzed per upload
MAX_IMAGE_SIZE = (800, 800)  # Resize large images to conserve tokens


def _prepare_image(image: Image.Image) -> Image.Image:
    """Convert to RGB and resize large images to save tokens and upload bandwidth."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    # Downscale if larger than limit, maintaining aspect ratio
    image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image


def describe_image(image: Image.Image, page_number: int) -> str | None:
    """
    Generate a concise description of a chart, diagram, or illustration.
    """
    if not settings.GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        processed_img = _prepare_image(image)

        prompt = (
            "Analyze this figure, diagram, or chart from an educational study document. "
            "In 2 to 3 concise, information-dense sentences, summarize what it depicts. "
            "Include any visible title, labels, numbers, axis trends, or key component relationships. "
            "Be direct and factual without conversational filler."
        )

        response = model.generate_content(
            [prompt, processed_img],
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=200,
            ),
        )
        text = response.text.strip() if response and response.text else ""
        return text or None

    except Exception as exc:
        logger.warning("Failed to describe image on page %d: %s", page_number, exc)
        return None


def ocr_scanned_page(image: Image.Image, page_number: int) -> str | None:
    """
    Extract readable text from a scanned document page or photo using Gemini Vision.
    """
    if not settings.GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        processed_img = _prepare_image(image)

        prompt = (
            "You are an expert document OCR engine. Extract all readable text, titles, headings, "
            "bullet points, tables, and notes from this scanned page. "
            "Preserve layout and structure accurately in clean markdown or plain text. "
            "Output only the extracted document text without any introductory commentary."
        )

        response = model.generate_content(
            [prompt, processed_img],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )
        text = response.text.strip() if response and response.text else ""
        return text or None

    except Exception as exc:
        logger.warning("OCR failed for scanned page %d: %s", page_number, exc)
        return None


def enrich_pdf_pages_with_vision(file_path: str, pages: list[PageData]) -> list[PageData]:
    """
    Scan a saved PDF for scanned pages and embedded diagrams, enriching PageData
    with OCR text and visual descriptions.
    """
    try:
        import pymupdf
    except ImportError:
        logger.warning("PyMuPDF not available; skipping vision enrichment.")
        return pages

    try:
        doc = pymupdf.open(file_path)
    except Exception as exc:
        logger.warning("Could not open %s with PyMuPDF: %s", file_path, exc)
        return pages

    total_pages_checked = min(len(pages), len(doc), MAX_VISUAL_PAGES_PER_DOC)

    for i in range(total_pages_checked):
        page_data = pages[i]
        pdf_page = doc[i]
        page_num = i + 1

        # -------------------------------------------------------------
        # 1. Scanned page OCR fallback
        # -------------------------------------------------------------
        if page_data.char_count < 50:
            try:
                pix = pdf_page.get_pixmap(dpi=150)
                page_img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = ocr_scanned_page(page_img, page_number=page_num)
                if ocr_text and len(ocr_text.strip()) > 30:
                    page_data.text = ocr_text.strip()
                    page_data.char_count = len(page_data.text)
                    logger.info(
                        "Page %d transcribed via Gemini OCR (%d chars)", page_num, len(ocr_text)
                    )
            except Exception as exc:
                logger.warning("Error performing OCR on page %d: %s", page_num, exc)

        # -------------------------------------------------------------
        # 2. Embedded figures & diagrams analysis
        # -------------------------------------------------------------
        try:
            image_list = pdf_page.get_images(full=True)
            figures_processed = 0

            for img_info in image_list:
                if figures_processed >= MAX_IMAGES_PER_PAGE:
                    break

                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img.get("image")
                if not img_bytes:
                    continue

                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size

                # Skip decorative elements and icons
                if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                    continue

                description = describe_image(pil_img, page_number=page_num)
                if description:
                    visual_block = f"\n\n[Figure on Page {page_num}: {description}]\n\n"
                    page_data.text += visual_block
                    page_data.char_count = len(page_data.text)
                    figures_processed += 1
                    logger.info("Enriched page %d with figure description", page_num)

        except Exception as exc:
            logger.warning("Error extracting images on page %d: %s", page_num, exc)

    doc.close()
    return pages
