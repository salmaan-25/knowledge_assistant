"""
tests/test_quiz_service.py — Unit tests for quiz scoring and serialisation.

Note: quiz generation tests (which call Gemini) are integration tests and
require GEMINI_API_KEY to be set.  Scoring is fully deterministic and is
tested here without any external calls.
"""

from __future__ import annotations

import json

import pytest

from app.schemas.quiz import (
    QuizQuestion,
    QuizSubmitRequest,
    SubmittedAnswer,
)
from app.services.quiz_service import (
    deserialise_questions,
    score_quiz,
    serialise_questions,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def sample_questions() -> list[QuizQuestion]:
    return [
        QuizQuestion(
            question_index=0,
            question_text="What is photosynthesis?",
            question_type="multiple_choice",
            options=[
                "A. Cellular respiration",
                "B. Light energy to glucose",
                "C. DNA replication",
                "D. Protein synthesis",
            ],
            correct_answer="B",
            explanation="Photosynthesis converts light energy into glucose.",
            source_pages=[3],
            topic="Biology",
        ),
        QuizQuestion(
            question_index=1,
            question_text="Is the Earth flat?",
            question_type="true_false",
            options=["True", "False"],
            correct_answer="False",
            explanation="The Earth is an oblate spheroid.",
            source_pages=[7],
            topic="Geography",
        ),
        QuizQuestion(
            question_index=2,
            question_text="What is 2 + 2?",
            question_type="multiple_choice",
            options=["A. 3", "B. 4", "C. 5", "D. 6"],
            correct_answer="B",
            explanation="Basic arithmetic: 2 + 2 = 4.",
            source_pages=[],
            topic="Mathematics",
        ),
    ]


# ------------------------------------------------------------------ #
# Scoring tests
# ------------------------------------------------------------------ #
def test_score_all_correct(sample_questions):
    submission = QuizSubmitRequest(
        answers=[
            SubmittedAnswer(question_index=0, answer="B"),
            SubmittedAnswer(question_index=1, answer="False"),
            SubmittedAnswer(question_index=2, answer="B"),
        ]
    )
    result = score_quiz("quiz-1", "doc-1", sample_questions, submission)
    assert result.score == 3.0
    assert result.max_score == 3
    assert result.percentage == 100.0
    assert result.passed is True
    assert result.weak_topics == []


def test_score_all_wrong(sample_questions):
    submission = QuizSubmitRequest(
        answers=[
            SubmittedAnswer(question_index=0, answer="A"),
            SubmittedAnswer(question_index=1, answer="True"),
            SubmittedAnswer(question_index=2, answer="A"),
        ]
    )
    result = score_quiz("quiz-1", "doc-1", sample_questions, submission)
    assert result.score == 0.0
    assert result.percentage == 0.0
    assert result.passed is False
    # All wrong → all topics become weak
    assert "Biology" in result.weak_topics
    assert "Geography" in result.weak_topics
    assert "Mathematics" in result.weak_topics


def test_score_partial(sample_questions):
    submission = QuizSubmitRequest(
        answers=[
            SubmittedAnswer(question_index=0, answer="B"),  # correct
            SubmittedAnswer(question_index=1, answer="True"),  # wrong
            SubmittedAnswer(question_index=2, answer="A"),  # wrong
        ]
    )
    result = score_quiz("quiz-1", "doc-1", sample_questions, submission)
    assert result.score == 1.0
    assert result.max_score == 3
    assert abs(result.percentage - 33.33) < 0.5


def test_score_case_insensitive(sample_questions):
    """Scoring should be case-insensitive."""
    submission = QuizSubmitRequest(
        answers=[
            SubmittedAnswer(question_index=0, answer="b"),  # lowercase b
            SubmittedAnswer(question_index=1, answer="false"),  # lowercase
            SubmittedAnswer(question_index=2, answer="B"),
        ]
    )
    result = score_quiz("quiz-1", "doc-1", sample_questions, submission)
    assert result.score == 3.0


def test_score_missing_answer_treated_as_wrong(sample_questions):
    """If a question index is not in the submission, it counts as wrong."""
    submission = QuizSubmitRequest(
        answers=[
            SubmittedAnswer(question_index=0, answer="B"),
            # question 1 and 2 not submitted
        ]
    )
    result = score_quiz("quiz-1", "doc-1", sample_questions, submission)
    assert result.score == 1.0


def test_results_include_explanations(sample_questions):
    submission = QuizSubmitRequest(
        answers=[SubmittedAnswer(question_index=i, answer="A") for i in range(3)]
    )
    result = score_quiz("quiz-1", "doc-1", sample_questions, submission)
    for r in result.results:
        assert r.explanation  # should never be empty


# ------------------------------------------------------------------ #
# Serialisation round-trip
# ------------------------------------------------------------------ #
def test_serialise_deserialise_round_trip(sample_questions):
    serialised = serialise_questions(sample_questions)
    # Must be valid JSON
    parsed = json.loads(serialised)
    assert isinstance(parsed, list)

    deserialised = deserialise_questions(serialised)
    assert len(deserialised) == len(sample_questions)
    for orig, restored in zip(sample_questions, deserialised, strict=False):
        assert orig.question_text == restored.question_text
        assert orig.correct_answer == restored.correct_answer


# ------------------------------------------------------------------ #
# QuizQuestion schema validation
# ------------------------------------------------------------------ #
def test_quiz_question_requires_options_for_multiple_choice():
    with pytest.raises(Exception):
        QuizQuestion(
            question_index=0,
            question_text="What is X?",
            question_type="multiple_choice",
            options=["A. Only one option"],  # less than 2 → invalid
            correct_answer="A",
            explanation="Explanation.",
            source_pages=[],
        )


def test_quiz_question_rejects_invalid_type():
    with pytest.raises(Exception):
        QuizQuestion(
            question_index=0,
            question_text="What is X?",
            question_type="essay",  # invalid type
            options=["A. Option"],
            correct_answer="A",
            explanation="Explanation.",
            source_pages=[],
        )
