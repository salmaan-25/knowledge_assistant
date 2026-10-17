"""
routes/progress.py — Learning progress and weak-topic tracking.

Endpoints:
  GET /progress                      — Get the current user's overall progress.
  GET /progress/{document_id}        — Get progress for a specific document.
  GET /progress/{document_id}/weak   — Get weak topics for a document.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import QuizAttempt, TopicProgress, User
from app.routes.auth import get_current_user
from app.routes.documents import _get_owned_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/progress", tags=["Progress"])


@router.get(
    "",
    summary="Get overall learning progress for the authenticated user",
)
async def get_overall_progress(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return aggregate learning statistics across all documents."""
    # Total quiz attempts
    attempts_result = await db.execute(
        select(func.count(QuizAttempt.id)).where(
            QuizAttempt.user_id == current_user.id,
            QuizAttempt.submitted_at.is_not(None),
        )
    )
    total_attempts = attempts_result.scalar() or 0

    # Average score across all submitted attempts
    avg_result = await db.execute(
        select(func.avg(QuizAttempt.percentage)).where(
            QuizAttempt.user_id == current_user.id,
            QuizAttempt.submitted_at.is_not(None),
        )
    )
    avg_score = round(avg_result.scalar() or 0.0, 2)

    # Topic progress records
    tp_result = await db.execute(
        select(TopicProgress).where(TopicProgress.user_id == current_user.id)
    )
    topic_records = tp_result.scalars().all()

    # Identify weak topics (accuracy < 60%)
    weak_topics = [
        {"topic": tp.topic, "document_id": tp.document_id, "accuracy": round(tp.accuracy * 100, 1)}
        for tp in topic_records
        if tp.accuracy < 0.60 and tp.attempts >= 2
    ]

    return {
        "user_id": current_user.id,
        "total_quiz_attempts": total_attempts,
        "average_score_percentage": avg_score,
        "topics_tracked": len(topic_records),
        "weak_topics": weak_topics,
    }


@router.get(
    "/{document_id}",
    summary="Get learning progress for a specific document",
)
async def get_document_progress(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return quiz history and topic performance for a single document."""
    # Verify ownership
    await _get_owned_document(document_id, current_user, db)

    # Quiz attempts for this document
    attempts_result = await db.execute(
        select(QuizAttempt)
        .where(
            QuizAttempt.user_id == current_user.id,
            QuizAttempt.document_id == document_id,
            QuizAttempt.submitted_at.is_not(None),
        )
        .order_by(QuizAttempt.submitted_at.desc())
    )
    attempts = attempts_result.scalars().all()

    # Topic progress for this document
    tp_result = await db.execute(
        select(TopicProgress).where(
            TopicProgress.user_id == current_user.id,
            TopicProgress.document_id == document_id,
        )
    )
    topic_records = tp_result.scalars().all()

    quiz_history = [
        {
            "quiz_id": a.id,
            "score": a.score,
            "max_score": a.max_score,
            "percentage": a.percentage,
            "difficulty": a.difficulty.value,
            "submitted_at": a.submitted_at.isoformat() if a.submitted_at else None,
        }
        for a in attempts
    ]

    topic_performance = [
        {
            "topic": tp.topic,
            "attempts": tp.attempts,
            "correct": tp.correct,
            "accuracy_percentage": round(tp.accuracy * 100, 1),
            "last_attempted_at": tp.last_attempted_at.isoformat() if tp.last_attempted_at else None,
        }
        for tp in topic_records
    ]

    return {
        "document_id": document_id,
        "total_quiz_attempts": len(quiz_history),
        "quiz_history": quiz_history,
        "topic_performance": topic_performance,
    }


@router.get(
    "/{document_id}/weak",
    summary="Get weak topics for a document (accuracy < 60%)",
)
async def get_weak_topics(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return topics where the user's accuracy is below 60%."""
    await _get_owned_document(document_id, current_user, db)

    tp_result = await db.execute(
        select(TopicProgress).where(
            TopicProgress.user_id == current_user.id,
            TopicProgress.document_id == document_id,
        )
    )
    topic_records = tp_result.scalars().all()

    weak = [
        {
            "topic": tp.topic,
            "attempts": tp.attempts,
            "correct": tp.correct,
            "accuracy_percentage": round(tp.accuracy * 100, 1),
        }
        for tp in topic_records
        if tp.accuracy < 0.60 and tp.attempts >= 1
    ]

    # Sort by accuracy ascending (worst topics first)
    weak.sort(key=lambda x: x["accuracy_percentage"])

    return {
        "document_id": document_id,
        "weak_topics": weak,
        "total_weak": len(weak),
        "recommendation": (
            "Focus on the topics listed above. Consider generating a targeted quiz "
            "with topic_filter set to each weak topic."
            if weak
            else "Great work! No weak topics identified."
        ),
    }
