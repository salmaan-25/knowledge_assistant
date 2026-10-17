"""
routes/documents.py — PDF upload, listing, retrieval, deletion, and summarisation.

Endpoints:
  POST   /documents/upload              — Upload and index a PDF.
  GET    /documents                     — List authenticated user's documents.
  GET    /documents/{document_id}       — Get a single document's metadata.
  DELETE /documents/{document_id}       — Delete a document and all its data.
  POST   /documents/{document_id}/summary — Generate a summary via the LangGraph agent.

All endpoints require authentication (Bearer JWT).
Ownership is enforced — a user can only access their own documents.
"""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph_state import INTENT_SUMMARISE
from app.agents.learning_agent import get_learning_graph
from app.database.connection import get_db
from app.database.models import Document, DocumentChunk, DocumentStatus, User
from app.routes.auth import get_current_user
from app.schemas.documents import (
    DocumentListResponse,
    DocumentOut,
    DocumentUploadResponse,
    SummaryRequest,
    SummaryResponse,
)
from app.services.chunking_service import chunk_pages
from app.services.embedding_service import (
    delete_document_vectors,
    index_chunks,
)
from app.services.pdf_service import delete_pdf_file, process_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


# ------------------------------------------------------------------ #
# Ownership guard
# ------------------------------------------------------------------ #
async def _get_owned_document(
    document_id: str,
    current_user: User,
    db: AsyncSession,
) -> Document:
    """
    Fetch a document and verify ownership.  Raises HTTP 404 on not-found
    and HTTP 403 on ownership violation.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if doc.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this document.",
        )
    return doc


# ------------------------------------------------------------------ #
# Background indexing task
# ------------------------------------------------------------------ #
async def _index_document_background(
    document_id: str,
    file_path: str,
) -> None:
    """
    Background task: chunk the PDF and index chunks into ChromaDB and PostgreSQL.

    Runs after the upload response has been returned to the client so the user
    is not blocked waiting for embedding generation.
    """
    from app.database.connection import AsyncSessionLocal
    from app.services.pdf_service import load_pdf_from_disk

    async with AsyncSessionLocal() as db:
        try:
            # Mark document as processing
            result = await db.execute(select(Document).where(Document.id == document_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return

            doc.status = DocumentStatus.PROCESSING
            await db.commit()

            # Re-extract from disk (avoids keeping file bytes in memory)
            extraction = load_pdf_from_disk(file_path)
            chunks = chunk_pages(extraction.pages, document_id=document_id)

            # Store chunk metadata in PostgreSQL
            chunk_orm_objects = [
                DocumentChunk(
                    document_id=document_id,
                    chroma_chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    text=c.text,
                    char_count=c.char_count,
                )
                for c in chunks
            ]
            db.add_all(chunk_orm_objects)

            # Index embeddings in ChromaDB
            index_chunks(chunks)

            # Update document record
            doc.status = DocumentStatus.READY
            doc.total_chunks = len(chunks)
            doc.page_count = extraction.page_count
            await db.commit()

            logger.info(
                "Document %s indexed: %d pages, %d chunks",
                document_id,
                extraction.page_count,
                len(chunks),
            )

        except Exception as exc:
            logger.exception("Background indexing failed for document %s: %s", document_id, exc)
            async with AsyncSessionLocal() as err_db:
                result = await err_db.execute(select(Document).where(Document.id == document_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = DocumentStatus.FAILED
                    doc.error_message = str(exc)[:500]
                    await err_db.commit()


# ================================================================== #
# POST /documents/upload
# ================================================================== #
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF and begin asynchronous indexing",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """
    Upload a PDF.

    1. Validates extension, MIME type, size, and PDF magic bytes.
    2. Saves the file to disk with a safe filename.
    3. Extracts page text synchronously to get a page count.
    4. Creates a Document record in PostgreSQL (status=PENDING).
    5. Schedules background indexing (chunking + embedding + ChromaDB).
    6. Returns immediately with HTTP 202 so the client is not blocked.
    """
    extraction = await process_upload(file)

    doc = Document(
        id=extraction.document_id,
        owner_id=current_user.id,
        filename=extraction.filename,
        original_filename=file.filename or "upload.pdf",
        file_path=extraction.file_path,
        file_size_bytes=extraction.file_size_bytes,
        mime_type=file.content_type or "application/pdf",
        page_count=extraction.page_count,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.commit()

    if extraction.is_empty:
        doc.status = DocumentStatus.FAILED
        doc.error_message = "No extractable text found. The PDF may be scanned or image-only."
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=doc.error_message,
        )

    background_tasks.add_task(
        _index_document_background,
        document_id=extraction.document_id,
        file_path=extraction.file_path,
    )

    return DocumentUploadResponse(
        document_id=extraction.document_id,
        filename=extraction.filename,
        page_count=extraction.page_count,
        total_chunks=0,  # will be updated after background indexing
        status=DocumentStatus.PENDING.value,
        message=(
            f"PDF uploaded successfully ({extraction.page_count} pages). "
            "Indexing is in progress; the document will be ready shortly."
        ),
    )


# ================================================================== #
# GET /documents
# ================================================================== #
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents belonging to the authenticated user",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    """Return all documents owned by the current user, ordered newest first."""
    result = await db.execute(
        select(Document)
        .where(Document.owner_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[DocumentOut.model_validate(d) for d in docs],
        total=len(docs),
    )


# ================================================================== #
# GET /documents/{document_id}
# ================================================================== #
@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Get metadata for a single document",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentOut:
    """Return metadata for a single owned document."""
    doc = await _get_owned_document(document_id, current_user, db)
    return DocumentOut.model_validate(doc)


# ================================================================== #
# DELETE /documents/{document_id}
# ================================================================== #
@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a document and all its indexed data",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    """
    Permanently delete a document.

    - Removes ChromaDB vectors.
    - Removes the file from disk.
    - Deletes the PostgreSQL record (cascades to chunks, quiz attempts, answers).
    """
    doc = await _get_owned_document(document_id, current_user, db)

    # Delete vectors from ChromaDB
    deleted_vectors = delete_document_vectors(document_id)
    logger.info("Deleted %d vectors for document %s", deleted_vectors, document_id)

    # Delete file from disk
    delete_pdf_file(doc.file_path)

    # Delete from PostgreSQL (cascade handles chunks, attempts, answers)
    await db.delete(doc)
    await db.commit()

    logger.info("Document %s deleted by user %s", document_id, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ================================================================== #
# POST /documents/{document_id}/summary
# ================================================================== #
@router.post(
    "/{document_id}/summary",
    response_model=SummaryResponse,
    summary="Generate a summary of a document or specific section",
)
async def summarise_document(
    document_id: str,
    body: SummaryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SummaryResponse:
    """
    Generate a summary using the LangGraph learning agent (summarisation branch).

    Accepts an optional section_hint to focus the summary.
    """
    doc = await _get_owned_document(document_id, current_user, db)

    if doc.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is not ready for queries. Current status: {doc.status.value}.",
        )

    graph = get_learning_graph()
    initial_state = {
        "document_id": document_id,
        "user_id": current_user.id,
        "question": body.section_hint or "Summarise the document",
        "intent": INTENT_SUMMARISE,  # bypass intent classification
        "section_hint": body.section_hint,
        "max_summary_length": body.max_length,
        "retry_count": 0,
        "difficulty": "medium",
        "question_count": 10,
    }

    # Route directly to retrieval (skip intent node by setting intent)
    # We re-use the QA retrieval path (retrieve_for_qa) for summaries.
    from app.agents.learning_agent import generate_summary, retrieve_for_qa

    retrieved = retrieve_for_qa(initial_state)
    initial_state.update(retrieved)
    final = generate_summary(initial_state)

    summary_text = final.get("answer", "Unable to generate summary.")
    sources = final.get("sources", [])
    source_pages = sorted({s["page_number"] for s in sources if s.get("page_number")})

    return SummaryResponse(
        document_id=document_id,
        summary=summary_text,
        source_pages=source_pages,
        section_hint=body.section_hint,
    )
