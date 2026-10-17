"""
routes/quizzes.py — Quiz generation, delivery (no answer leakage), and submission.

Endpoints:
  POST /quizzes/generate           — Generate a quiz from a document.
  GET  /quizzes/{quiz_id}          — Get a quiz's questions (no correct answers).
  POST /quizzes/{quiz_id}/submit   — Submit answers and receive a scored result.

Security:
  - correct_answer fields are NEVER included in GET /quizzes/{quiz_id} or
    the /generate response.
  - The answer key is stored server-side in quiz_attempts.questions_json.
  - Scoring is 100% deterministic Python — no LLM involvement in grading.
  - Ownership is enforced — users can only access their own quiz attempts.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import (
    DocumentStatus,
    QuizAnswer,
    QuizAttempt,
    TopicProgress,
    User,
)
from app.routes.auth import get_current_user
from app.routes.documents import _get_owned_document
from app.schemas.quiz import (
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizQuestionPublic,
    QuizSubmitRequest,
    QuizSubmitResponse,
)
from app.services.embedding_service import retrieve_chunks
from app.services.quiz_service import (
    deserialise_questions,
    generate_quiz,
    score_quiz,
    serialise_questions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quizzes", tags=["Quizzes"])


# ------------------------------------------------------------------ #
# Ownership guard for quiz attempts
# ------------------------------------------------------------------ #
async def _get_owned_attempt(
    quiz_id: str,
    current_user: User,
    db: AsyncSession,
) -> QuizAttempt:
    result = await db.execute(select(QuizAttempt).where(QuizAttempt.id == quiz_id))
    attempt = result.scalar_one_or_none()

    if attempt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found.")
    if attempt.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this quiz.",
        )
    return attempt


# ================================================================== #
# POST /quizzes/generate
# ================================================================== #
@router.post(
    "/generate",
    response_model=QuizGenerateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a quiz from a document",
)
async def generate_quiz_endpoint(
    body: QuizGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QuizGenerateResponse:
    """
    Generate a quiz from the user's document.

    Steps:
      1. Verify document ownership and READY status.
      2. Retrieve relevant chunks from ChromaDB.
      3. Call quiz_service.generate_quiz() (Gemini + Pydantic validation).
      4. Store the quiz attempt (including answer key) in PostgreSQL.
      5. Return the questions WITHOUT correct_answer — only the public view.
    """
    doc = await _get_owned_document(body.document_id, current_user, db)

    if doc.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is not ready. Status: {doc.status.value}.",
        )

    # Retrieve relevant chunks
    topic_query = body.topic_filter or "key learning concepts important topics"
    chunks = retrieve_chunks(
        query=topic_query,
        document_id=body.document_id,
        top_k=min(body.question_count * 2, 20),
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No relevant content found in the document to generate quiz questions.",
        )

    # Generate and validate quiz
    try:
        quiz = generate_quiz(
            chunks=chunks,
            document_id=body.document_id,
            question_count=body.question_count,
            difficulty=body.difficulty,
            topic_filter=body.topic_filter,
            question_types=body.question_types,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    # Persist quiz attempt with server-side answer key
    attempt = QuizAttempt(
        user_id=current_user.id,
        document_id=body.document_id,
        topic_filter=body.topic_filter,
        difficulty=body.difficulty,
        question_count=len(quiz.questions),
        questions_json=serialise_questions(quiz.questions),
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)

    # Build public (no-answer) questions for the response
    public_questions = [
        QuizQuestionPublic(
            question_index=q.question_index,
            question_text=q.question_text,
            question_type=q.question_type,
            options=q.options,
            topic=q.topic,
        )
        for q in quiz.questions
    ]

    logger.info(
        "Quiz %s generated: %d questions (difficulty=%s) for document %s",
        attempt.id,
        len(public_questions),
        body.difficulty,
        body.document_id,
    )

    return QuizGenerateResponse(
        quiz_id=attempt.id,
        document_id=body.document_id,
        questions=public_questions,
        question_count=len(public_questions),
        difficulty=body.difficulty,
        topic_filter=body.topic_filter,
    )


# ================================================================== #
# GET /quizzes/{quiz_id}
# ================================================================== #
@router.get(
    "/{quiz_id}",
    response_model=QuizGenerateResponse,
    summary="Get a quiz's questions (no correct answers exposed)",
)
async def get_quiz(
    quiz_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QuizGenerateResponse:
    """Return the quiz questions without revealing correct answers."""
    attempt = await _get_owned_attempt(quiz_id, current_user, db)
    questions = deserialise_questions(attempt.questions_json)

    public_questions = [
        QuizQuestionPublic(
            question_index=q.question_index,
            question_text=q.question_text,
            question_type=q.question_type,
            options=q.options,
            topic=q.topic,
        )
        for q in questions
    ]

    return QuizGenerateResponse(
        quiz_id=attempt.id,
        document_id=attempt.document_id,
        questions=public_questions,
        question_count=len(public_questions),
        difficulty=attempt.difficulty.value,
        topic_filter=attempt.topic_filter,
    )


# ================================================================== #
# POST /quizzes/{quiz_id}/submit
# ================================================================== #
@router.post(
    "/{quiz_id}/submit",
    response_model=QuizSubmitResponse,
    summary="Submit answers and receive a deterministically scored result",
)
async def submit_quiz(
    quiz_id: str,
    body: QuizSubmitRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QuizSubmitResponse:
    """
    Submit quiz answers and receive the score and explanations.

    - Prevents double-submission (idempotent check on submitted_at).
    - Scores are calculated 100% in Python — no LLM scoring.
    - Stores individual answer records in quiz_answers.
    - Updates topic_progress for weak-topic tracking.
    """
    attempt = await _get_owned_attempt(quiz_id, current_user, db)

    # Prevent double submission
    if attempt.submitted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This quiz has already been submitted.",
        )

    questions = deserialise_questions(attempt.questions_json)

    # Deterministic scoring
    result = score_quiz(
        quiz_id=quiz_id,
        document_id=attempt.document_id,
        questions=questions,
        submission=body,
    )

    # Persist individual answer records
    answer_records = [
        QuizAnswer(
            attempt_id=quiz_id,
            question_index=r.question_index,
            submitted_answer=r.submitted_answer,
            correct_answer=r.correct_answer,
            is_correct=r.is_correct,
            explanation=r.explanation,
            source_pages=json.dumps(r.source_pages),
        )
        for r in result.results
    ]
    db.add_all(answer_records)

    # Update the attempt record with final scores
    attempt.score = result.score
    attempt.max_score = result.max_score
    attempt.percentage = result.percentage
    attempt.submitted_at = datetime.now(UTC)

    # Update topic progress for weak-topic tracking
    for r in result.results:
        question = next((q for q in questions if q.question_index == r.question_index), None)
        topic = question.topic if question else None
        if not topic:
            continue

        tp_result = await db.execute(
            select(TopicProgress).where(
                TopicProgress.user_id == current_user.id,
                TopicProgress.document_id == attempt.document_id,
                TopicProgress.topic == topic,
            )
        )
        tp = tp_result.scalar_one_or_none()

        if tp is None:
            tp = TopicProgress(
                user_id=current_user.id,
                document_id=attempt.document_id,
                topic=topic,
                attempts=0,
                correct=0,
            )
            db.add(tp)

        tp.attempts += 1
        if r.is_correct:
            tp.correct += 1
        tp.last_attempted_at = datetime.now(UTC)

    await db.commit()

    logger.info(
        "Quiz %s submitted: score=%.1f/%d (%.1f%%)",
        quiz_id,
        result.score,
        result.max_score,
        result.percentage,
    )
    return result
