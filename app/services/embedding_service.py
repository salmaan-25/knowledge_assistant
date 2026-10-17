"""
services/embedding_service.py — Gemini embedding generation and ChromaDB management.

Responsibilities:
  - Generate text embeddings via the Gemini text-embedding-004 model.
  - Store chunk embeddings in a persisted ChromaDB collection.
  - Provide a retrieval function that returns the top-K relevant chunks
    for a query, with full metadata (document_id, page_number).
  - Delete all vectors for a document on document deletion.

Design notes:
  - We use ChromaDB's built-in cosine distance.
  - Embeddings are generated in batches to stay within API rate limits.
  - The ChromaDB collection is created once and reused across requests.
  - A document is never re-embedded if its chunks already exist in the
    collection (idempotent upsert).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from google import generativeai as genai

from app.config import get_settings
from app.services.chunking_service import TextChunk

logger = logging.getLogger(__name__)
settings = get_settings()

# ------------------------------------------------------------------ #
# Gemini client setup
# ------------------------------------------------------------------ #
genai.configure(api_key=settings.GEMINI_API_KEY)

# ------------------------------------------------------------------ #
# ChromaDB client (lazy singleton)
# ------------------------------------------------------------------ #
_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return (or create) the persistent ChromaDB client singleton."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialised at %s", settings.CHROMA_PERSIST_DIR)
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    """Return (or create) the ChromaDB collection for knowledge chunks."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ------------------------------------------------------------------ #
# Embedding generation
# ------------------------------------------------------------------ #
_EMBED_BATCH_SIZE = 100  # Gemini embedding API batch limit
_RETRY_SLEEP_S = 2  # Seconds to wait before retrying on rate limit


def _embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the Gemini embedding model.

    Splits into batches of _EMBED_BATCH_SIZE and retries once on API error.

    task_type: "RETRIEVAL_DOCUMENT" for indexing, "RETRIEVAL_QUERY" for queries.
    """
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
        attempt = 0
        while attempt < settings.MAX_LLM_RETRIES:
            try:
                result = genai.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                all_embeddings.extend(result["embedding"])
                break
            except Exception as exc:
                attempt += 1
                logger.warning(
                    "Embedding attempt %d/%d failed: %s", attempt, settings.MAX_LLM_RETRIES, exc
                )
                if attempt < settings.MAX_LLM_RETRIES:
                    time.sleep(_RETRY_SLEEP_S * attempt)
                else:
                    raise RuntimeError(
                        f"Failed to generate embeddings after {settings.MAX_LLM_RETRIES} attempts: {exc}"
                    ) from exc

    return all_embeddings


# ------------------------------------------------------------------ #
# Indexing
# ------------------------------------------------------------------ #
def index_chunks(chunks: list[TextChunk]) -> int:
    """
    Generate embeddings for `chunks` and upsert them into ChromaDB.

    Returns the number of chunks successfully indexed.
    Uses upsert so re-indexing the same document is idempotent.
    """
    if not chunks:
        logger.warning("index_chunks called with an empty chunk list.")
        return 0

    collection = _get_collection()

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas: list[dict[str, Any]] = [
        {
            "document_id": c.document_id,
            "page_number": c.page_number if c.page_number is not None else -1,
            "chunk_index": c.chunk_index,
            "char_count": c.char_count,
        }
        for c in chunks
    ]

    logger.info("Generating embeddings for %d chunks…", len(chunks))
    embeddings = _embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("Indexed %d chunks for document %s", len(chunks), chunks[0].document_id)
    return len(chunks)


# ------------------------------------------------------------------ #
# Retrieval
# ------------------------------------------------------------------ #
class RetrievedChunk:
    """A chunk returned from a similarity search."""

    __slots__ = ("chunk_id", "distance", "document_id", "page_number", "text")

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        page_number: int | None,
        text: str,
        distance: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page_number = page_number if page_number != -1 else None
        self.text = text
        self.distance = distance

    @property
    def relevance_score(self) -> float:
        """Convert cosine distance to a 0–1 relevance score (1 = most relevant)."""
        return max(0.0, 1.0 - self.distance)


def retrieve_chunks(
    query: str,
    document_id: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Embed the query and return the top-K most relevant chunks for the given document.

    Parameters
    ----------
    query:       Natural language query string.
    document_id: Filter results to this document only.
    top_k:       Number of chunks to return (defaults to settings.RETRIEVAL_TOP_K).
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    collection = _get_collection()

    query_embedding = _embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=False,
    ):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=metadata.get("document_id", document_id),
                page_number=metadata.get("page_number"),
                text=text,
                distance=distance,
            )
        )

    logger.info(
        "Retrieved %d chunks for document %s (query: '%.80s…')",
        len(chunks),
        document_id,
        query,
    )
    return chunks


# ------------------------------------------------------------------ #
# Deletion
# ------------------------------------------------------------------ #
def delete_document_vectors(document_id: str) -> int:
    """
    Remove all vectors belonging to a document from ChromaDB.

    Returns the number of vectors deleted.
    """
    collection = _get_collection()

    # First, find all chunk IDs for this document.
    existing = collection.get(
        where={"document_id": document_id},
        include=[],  # only IDs needed
    )
    ids_to_delete = existing["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d vectors for document %s", len(ids_to_delete), document_id)

    return len(ids_to_delete)


def get_document_chunk_count(document_id: str) -> int:
    """Return how many chunks are indexed for a given document."""
    collection = _get_collection()
    result = collection.get(where={"document_id": document_id}, include=[])
    return len(result["ids"])
