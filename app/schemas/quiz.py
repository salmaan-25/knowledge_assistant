"""
schemas/quiz.py — Pydantic models for quiz generation, delivery, and evaluation.

Design principle (from the PRD):
  - The server stores the answer key; the client never receives correct_answer
    before submission.
  - Scoring is deterministic Python — not LLM-based.
  - Structured output from Gemini is validated against QuizQuestion before
    being stored, so invalid model output is caught early.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

# ------------------------------------------------------------------ #
# Question types and difficulty
# ------------------------------------------------------------------ #
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUESTION_TYPES = {"multiple_choice", "true_false", "short_answer"}


# ------------------------------------------------------------------ #
# Internal models (server-side — include correct answer)
# ------------------------------------------------------------------ #
class QuizQuestion(BaseModel):
    """
    Full question model stored server-side.
    correct_answer must NEVER be sent to the client before submission.
    """

    question_index: int
    question_text: str = Field(min_length=5)
    question_type: str = Field(default="multiple_choice")
    options: list[str] = Field(
        default_factory=list,
        description="For multiple_choice: list of option strings (e.g. ['A. ...', 'B. ...']).",
    )
    correct_answer: str = Field(
        description="The correct option letter or answer text.",
    )
    explanation: str = Field(
        description="Explanation shown after the user submits their answer.",
    )
    source_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers that support this question.",
    )
    topic: str | None = None

    @field_validator("question_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {VALID_QUESTION_TYPES}")
        return v

    @model_validator(mode="after")
    def validate_options(self) -> QuizQuestion:
        if self.question_type == "multiple_choice" and len(self.options) < 2:
            raise ValueError("multiple_choice questions must have at least 2 options.")
        return self


class GeneratedQuiz(BaseModel):
    """The full quiz as generated and validated by the server."""

    questions: list[QuizQuestion]
    document_id: str
    difficulty: str
    topic_filter: str | None = None

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        if v not in VALID_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {VALID_DIFFICULTIES}")
        return v


# ------------------------------------------------------------------ #
# Client-facing models (no correct_answer exposed pre-submission)
# ------------------------------------------------------------------ #
class QuizQuestionPublic(BaseModel):
    """Question sent to the client — correct_answer is withheld."""

    question_index: int
    question_text: str
    question_type: str
    options: list[str]
    topic: str | None = None


class QuizGenerateRequest(BaseModel):
    """Request body for POST /quizzes/generate."""

    document_id: str
    question_count: int = Field(default=10, ge=1, le=30)
    difficulty: str = Field(default="medium")
    topic_filter: str | None = Field(
        default=None,
        max_length=300,
        description="Optional topic or chapter to focus the quiz on.",
    )
    question_types: list[str] = Field(
        default=["multiple_choice"],
        description="List of allowed question types.",
    )

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        if v not in VALID_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {VALID_DIFFICULTIES}")
        return v


class QuizGenerateResponse(BaseModel):
    """Response returned after quiz generation — no answers exposed."""

    quiz_id: str
    document_id: str
    questions: list[QuizQuestionPublic]
    question_count: int
    difficulty: str
    topic_filter: str | None = None
    expires_at: str | None = None


# ------------------------------------------------------------------ #
# Submission and results
# ------------------------------------------------------------------ #
class SubmittedAnswer(BaseModel):
    """A single answer submitted by the user."""

    question_index: int
    answer: str = Field(default="", max_length=500)
    selected_option: str | None = None

    @model_validator(mode="after")
    def validate_answer(self) -> SubmittedAnswer:
        if not self.answer and self.selected_option:
            self.answer = self.selected_option
        return self


class QuizSubmitRequest(BaseModel):
    """Request body for POST /quizzes/{quiz_id}/submit."""

    answers: list[SubmittedAnswer]


class AnswerResult(BaseModel):
    """Evaluation result for a single submitted answer."""

    question_index: int
    question_text: str
    submitted_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str
    source_pages: list[int]


class QuizSubmitResponse(BaseModel):
    """Full quiz result returned after submission."""

    quiz_id: str
    document_id: str
    score: float
    max_score: int
    percentage: float
    passed: bool
    results: list[AnswerResult]
    weak_topics: list[str] = Field(
        default_factory=list,
        description="Topics where the user answered incorrectly.",
    )
