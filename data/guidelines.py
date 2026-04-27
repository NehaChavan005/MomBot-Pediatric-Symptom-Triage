"""Hardcoded pediatric triage safety rules used before any model call."""

from __future__ import annotations


EMERGENCY_SYMPTOMS: dict[str, bool] = {
    "difficulty breathing": True,
    "not breathing": True,
    "blue lips": True,
    "seizure": True,
    "unconscious": True,
    "severe allergic reaction": True,
    "anaphylaxis": True,
    "fever under 3 months": True,
    "meningitis rash": True,
    "choking": True,
}


def check_emergency_keywords(text: str) -> bool:
    """Return True when the text contains any hardcoded emergency keyword."""
    normalized_text = (text or "").lower()
    return any(keyword in normalized_text for keyword in EMERGENCY_SYMPTOMS)
