"""
agents/graph_state.py — Typed state shared across all LangGraph nodes.

The GraphState dataclass is the single source of truth for what has been
computed at each step of the learning workflow.  Every node reads from and
writes to a copy of this state; LangGraph merges updates between nodes.

Design notes:
  - Using TypedDict for LangGraph compatibility (LangGraph requires a dict-like
    state, not a plain dataclass).
  - All fields are Optional so nodes can be entered with partial state and
    fill in their own fields.
  - The 'intent' field drives conditional routing between workflow branches.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from app.services.embedding_service import RetrievedChunk


class GraphState(TypedDict, total=False):
    """
    Shared mutable state passed between LangGraph nodes.

    Fields
    ------
    document_id : str
        UUID of the document being worked with.
    user_id : str
        UUID of the requesting user.
    question : str | None
        The user's raw input (question, quiz request, etc.).
    intent : str | None
        Classified intent: "question_answering" | "summarization" | "quiz_generation"
        | "progress_lookup" | "unknown".
    retrieved_chunks : list[RetrievedChunk] | None
        Passages retrieved from ChromaDB relevant to the query.
    answer : str | None
        Generated answer text (question answering and summarization branch).
    sources : list[dict] | None
        Source references (chunk_id, page_number, excerpt) for the answer.
    has_sufficient_context : bool
        True when retrieved evidence is strong enough to ground an answer.
    generated_quiz : Any | None
        The validated GeneratedQuiz object (quiz generation branch).
    error : str | None
        Set by any node that encounters a non-fatal error.
    retry_count : int
        Tracks how many LLM retries have been used in the current invocation.
    topic_filter : str | None
        Optional topic hint for quiz generation or retrieval filtering.
    difficulty : str
        Quiz difficulty: "easy" | "medium" | "hard".
    question_count : int
        Number of quiz questions requested.
    section_hint : str | None
        Optional section/chapter hint for summarisation.
    max_summary_length : int | None
        Target summary length in words.
    """

    document_id: str
    user_id: str
    question: str | None
    intent: str | None
    retrieved_chunks: list[RetrievedChunk] | None
    answer: str | None
    sources: list[dict[str, Any]] | None
    has_sufficient_context: bool
    generated_quiz: Any | None
    error: str | None
    retry_count: int
    topic_filter: str | None
    difficulty: str
    question_count: int
    section_hint: str | None
    max_summary_length: int | None
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


# ------------------------------------------------------------------ #
# Routing constants — referenced both here and in learning_agent.py
# ------------------------------------------------------------------ #
INTENT_QA = "question_answering"
INTENT_SUMMARISE = "summarization"
INTENT_QUIZ = "quiz_generation"
INTENT_PROGRESS = "progress_lookup"
INTENT_UNKNOWN = "unknown"

VALID_INTENTS = {INTENT_QA, INTENT_SUMMARISE, INTENT_QUIZ, INTENT_PROGRESS, INTENT_UNKNOWN}
