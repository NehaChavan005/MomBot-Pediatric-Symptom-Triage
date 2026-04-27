"""Pydantic schemas used across the MomBot application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ConversationTurn(BaseModel):
    """Represents a single conversation turn between the user and assistant."""

    role: Literal["user", "assistant"]
    content: str | None

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str | None) -> str | None:
        """Normalize blank conversation content to None to avoid empty required strings."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SymptomInput(BaseModel):
    """Incoming payload for a triage request."""

    message: str
    child_age_months: int | None = Field(default=None, ge=0)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        """Preserve the message but coerce None-like blanks into an empty string."""
        return value.strip() if isinstance(value, str) else ""


class TriageResponse(BaseModel):
    """Structured triage response returned to clients and validated from model output."""

    symptom_summary: str
    severity: Literal["low", "medium", "high", "emergency"]
    possible_causes: list[str] = Field(default_factory=list, max_length=4)
    home_care_steps: list[str] = Field(default_factory=list)
    escalate_to_doctor: bool
    escalate_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    disclaimer: str
    response_language: Literal["en", "ar"]
    needs_clarification: bool
    clarifying_question: str | None = None

    @field_validator(
        "symptom_summary",
        "disclaimer",
        "escalate_reason",
        "clarifying_question",
        mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: str | None) -> str | None:
        """Convert blank strings to None and trim non-empty strings."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("possible_causes", "home_care_steps")
    @classmethod
    def strip_list_items(cls, values: list[str]) -> list[str]:
        """Remove blank list items while preserving order."""
        cleaned_values = [item.strip() for item in values if isinstance(item, str) and item.strip()]
        return cleaned_values

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "TriageResponse":
        """Enforce safety and completeness constraints across fields."""
        if not self.symptom_summary:
            raise ValueError("symptom_summary is required")
        if not self.disclaimer:
            raise ValueError("disclaimer is required")
        if self.escalate_to_doctor and not self.escalate_reason:
            raise ValueError("escalate_reason is required when escalate_to_doctor is True")
        if self.needs_clarification and not self.clarifying_question:
            raise ValueError("clarifying_question is required when needs_clarification is True")
        if self.severity == "emergency" and self.home_care_steps:
            raise ValueError("home_care_steps must be empty for emergency responses")
        return self
