"""
services/quiz_service.py — Quiz generation, validation, and deterministic scoring.

Responsibilities:
  - Build a structured prompt asking Gemini to generate quiz questions from
    retrieved document chunks.
  - Parse and validate the JSON response against QuizQuestion Pydantic models.
  - Retry on invalid output (bounded retries from settings).
  - Store quiz questions (including answer key) in PostgreSQL via the route handler.
  - Evaluate submitted answers deterministically in Python — never ask the LLM
    to calculate a score.
  - Compute per-question explanations from the stored question data.
  - Identify weak topics for progress tracking.

Security:
  - The correct_answer field is NEVER returned to the client before submission.
  - The answer key is stored server-side in the quiz_attempts.questions_json column.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import google.generativeai as genai

from app.config import get_settings
from app.schemas.quiz import (
    AnswerResult,
    GeneratedQuiz,
    QuizQuestion,
    QuizSubmitRequest,
    QuizSubmitResponse,
    SubmittedAnswer,
)
from app.services.embedding_service import RetrievedChunk

logger = logging.getLogger(__name__)
settings = get_settings()

genai.configure(api_key=settings.GEMINI_API_KEY)


# ------------------------------------------------------------------ #
# Prompt builder
# ------------------------------------------------------------------ #
def _build_quiz_prompt(
    chunks: list[RetrievedChunk],
    question_count: int,
    difficulty: str,
    topic_filter: str | None,
    question_types: list[str],
) -> str:
    """Build the quiz generation prompt sent to Gemini."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        page_ref = f"(page {chunk.page_number})" if chunk.page_number else ""
        context_blocks.append(f"[Passage {i} {page_ref}]\n{chunk.text}")

    context_text = "\n\n---\n\n".join(context_blocks)
    types_str = ", ".join(question_types)
    topic_line = f"Focus on the topic: {topic_filter}." if topic_filter else ""

    prompt = f"""You are an expert educator creating a quiz for a student.

CONTEXT (extracted from the student's uploaded document):
{context_text}

TASK:
Generate exactly {question_count} quiz questions based ONLY on the context above.
{topic_line}

Requirements:
- Difficulty level: {difficulty}
- Question types allowed: {types_str}
- For multiple_choice questions: provide exactly 4 options labelled A, B, C, D.
  Each option must be a complete string starting with the letter (e.g. "A. The mitochondria").
- For true_false questions: options must be exactly ["True", "False"].
- correct_answer must be the letter only for multiple_choice (e.g. "A") or "True"/"False".
- Do NOT invent facts not present in the context.
- Include a brief explanation (2-3 sentences) citing which passage supports the answer.
- Include source_pages as a list of integers (page numbers from the passages).

Return a JSON object with this exact structure (no markdown, no code fences):
{{
  "questions": [
    {{
      "question_index": 0,
      "question_text": "...",
      "question_type": "multiple_choice",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "A",
      "explanation": "...",
      "source_pages": [1, 2],
      "topic": "optional topic label"
    }}
  ]
}}"""
    return prompt


# ------------------------------------------------------------------ #
# JSON extraction and validation
# ------------------------------------------------------------------ #
def _extract_json_from_response(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from the model response.

    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}\nRaw text: {text[:500]}") from exc


def _validate_quiz_json(
    raw: dict[str, Any], document_id: str, difficulty: str, topic_filter: str | None
) -> GeneratedQuiz:
    """Validate the parsed JSON against the GeneratedQuiz Pydantic model."""
    try:
        raw["document_id"] = document_id
        raw["difficulty"] = difficulty
        raw["topic_filter"] = topic_filter
        return GeneratedQuiz(**raw)
    except Exception as exc:
        raise ValueError(f"Quiz schema validation failed: {exc}") from exc


# ------------------------------------------------------------------ #
# Public: generate quiz
# ------------------------------------------------------------------ #
def generate_quiz(
    chunks: list[RetrievedChunk],
    document_id: str,
    question_count: int = 10,
    difficulty: str = "medium",
    topic_filter: str | None = None,
    question_types: list[str] | None = None,
) -> GeneratedQuiz:
    """
    Generate a validated quiz from the provided retrieved chunks.

    Retries up to settings.MAX_LLM_RETRIES times on invalid model output.
    Raises RuntimeError if all retries are exhausted.
    """
    question_types = question_types or ["multiple_choice"]
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    prompt = _build_quiz_prompt(
        chunks=chunks,
        question_count=question_count,
        difficulty=difficulty,
        topic_filter=topic_filter,
        question_types=question_types,
    )

    last_error: Exception | None = None
    for attempt in range(1, settings.MAX_LLM_RETRIES + 1):
        try:
            logger.info("Quiz generation attempt %d/%d", attempt, settings.MAX_LLM_RETRIES)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=4096,
                ),
            )
            raw_dict = _extract_json_from_response(response.text)
            quiz = _validate_quiz_json(raw_dict, document_id, difficulty, topic_filter)

            # Normalise question indices
            for i, q in enumerate(quiz.questions):
                q.question_index = i

            logger.info(
                "Successfully generated %d questions (difficulty=%s)",
                len(quiz.questions),
                difficulty,
            )
            return quiz

        except (ValueError, RuntimeError) as exc:
            last_error = exc
            logger.warning("Quiz generation attempt %d failed: %s", attempt, exc)
            if attempt < settings.MAX_LLM_RETRIES:
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Quiz generation failed after {settings.MAX_LLM_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ------------------------------------------------------------------ #
# Public: score quiz
# ------------------------------------------------------------------ #
def score_quiz(
    quiz_id: str,
    document_id: str,
    questions: list[QuizQuestion],
    submission: QuizSubmitRequest,
) -> QuizSubmitResponse:
    """
    Deterministically score the submitted answers against the stored answer key.

    Steps:
      1. Build a lookup from question_index → QuizQuestion.
      2. For each submitted answer, compare with the correct_answer (case-insensitive
         strip).
      3. Aggregate results, score, and weak topics.
      4. Never call the LLM for scoring.

    Raises ValueError if a submitted question_index is out of range.
    """
    question_map: dict[int, QuizQuestion] = {q.question_index: q for q in questions}
    results: list[AnswerResult] = []
    weak_topics: list[str] = []
    correct_count = 0

    # Build a submitted-answer lookup for O(1) access
    answer_map: dict[int, SubmittedAnswer] = {a.question_index: a for a in submission.answers}

    for idx, question in sorted(question_map.items()):
        submitted = answer_map.get(idx)
        submitted_text = submitted.answer.strip() if submitted else ""
        correct_text = question.correct_answer.strip()

        is_correct = submitted_text.upper() == correct_text.upper()
        if is_correct:
            correct_count += 1
        elif question.topic:
            weak_topics.append(question.topic)

        results.append(
            AnswerResult(
                question_index=idx,
                question_text=question.question_text,
                submitted_answer=submitted_text or "(no answer)",
                correct_answer=correct_text,
                is_correct=is_correct,
                explanation=question.explanation,
                source_pages=question.source_pages,
            )
        )

    max_score = len(questions)
    percentage = round((correct_count / max_score) * 100, 2) if max_score > 0 else 0.0
    passed = percentage >= 60.0

    # Deduplicate weak topics while preserving order
    seen: set[str] = set()
    unique_weak_topics = [t for t in weak_topics if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    return QuizSubmitResponse(
        quiz_id=quiz_id,
        document_id=document_id,
        score=float(correct_count),
        max_score=max_score,
        percentage=percentage,
        passed=passed,
        results=results,
        weak_topics=unique_weak_topics,
    )


# ------------------------------------------------------------------ #
# Serialisation helpers (for storing/loading questions_json)
# ------------------------------------------------------------------ #
def serialise_questions(questions: list[QuizQuestion]) -> str:
    """Serialise quiz questions (including answers) to JSON for DB storage."""
    return json.dumps([q.model_dump() for q in questions])


def deserialise_questions(questions_json: str) -> list[QuizQuestion]:
    """Deserialise quiz questions from DB JSON storage."""
    raw = json.loads(questions_json)
    return [QuizQuestion(**q) for q in raw]
