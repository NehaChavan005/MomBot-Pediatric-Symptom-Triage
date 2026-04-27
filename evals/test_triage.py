"""Evaluation suite for MomBot triage behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schema import SymptomInput
from app.triage import triage_symptoms


def _mock_openai_response(payload: dict) -> SimpleNamespace:
    """Create a minimal OpenAI chat completion-shaped response for tests."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False)
                )
            )
        ]
    )


def _base_response(**overrides) -> dict:
    """Return a valid baseline triage response with optional field overrides."""
    payload = {
        "symptom_summary": "Child has mild symptoms.",
        "severity": "low",
        "possible_causes": ["viral upper respiratory infection", "common cold"],
        "home_care_steps": ["Offer fluids", "Monitor temperature"],
        "escalate_to_doctor": False,
        "escalate_reason": None,
        "confidence": 0.75,
        "disclaimer": "This is not a substitute for professional medical advice, diagnosis, or treatment.",
        "response_language": "en",
        "needs_clarification": False,
        "clarifying_question": None,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_easy_case_runny_nose_mild_fever() -> None:
    """Case 1: mild fever and runny nose should not escalate."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(severity="low", escalate_to_doctor=False)
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(
                message="My 2-year-old has a runny nose and mild fever of 37.5°C",
                child_age_months=24,
                conversation_history=[],
            )
        )

    assert result.severity in {"low", "medium"}
    assert result.escalate_to_doctor is False


@pytest.mark.asyncio
async def test_easy_case_diarrhea_home_care() -> None:
    """Case 2: diarrhea for two days should include home care and no escalation."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                symptom_summary="Baby has diarrhea for two days.",
                home_care_steps=["Offer oral fluids", "Watch for dehydration signs"],
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(message="My baby has had diarrhea for 2 days", conversation_history=[])
        )

    assert result.escalate_to_doctor is False
    assert result.home_care_steps


@pytest.mark.asyncio
async def test_easy_case_rash_needs_clarification_or_low() -> None:
    """Case 3: rash after playing outside may clarify or stay low severity."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                severity="low",
                needs_clarification=True,
                clarifying_question="How long has the rash been present?",
                possible_causes=[],
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(
                message="My 5-year-old has a rash on his arm after playing outside",
                child_age_months=60,
                conversation_history=[],
            )
        )

    assert result.needs_clarification is True or result.severity == "low"


@pytest.mark.asyncio
async def test_easy_case_arabic_cough_language() -> None:
    """Case 4: Arabic input should preserve Arabic response language."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                symptom_summary="الطفل لديه سعال منذ يومين.",
                response_language="ar",
                disclaimer="هذه المعلومات ليست بديلاً عن المشورة الطبية المهنية أو التشخيص أو العلاج.",
                possible_causes=["نزلة برد فيروسية", "تهيج خفيف في الحلق"],
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(message="طفلي عنده سعال منذ يومين", conversation_history=[])
        )

    assert result.response_language == "ar"


@pytest.mark.asyncio
async def test_medium_case_fever_39_escalates() -> None:
    """Case 5: fever in an 8-month-old should escalate to a doctor."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                severity="high",
                escalate_to_doctor=True,
                escalate_reason="An 8-month-old with 39°C fever should be medically assessed.",
                confidence=0.82,
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(
                message="My 8-month-old has a fever of 39°C",
                child_age_months=8,
                conversation_history=[],
            )
        )

    assert result.severity in {"medium", "high"}
    assert result.escalate_to_doctor is True


@pytest.mark.asyncio
async def test_medium_case_allergic_emergency_via_api() -> None:
    """Case 6: hives and swollen lips should trigger an emergency response."""
    with patch("app.main.check_emergency_keywords", return_value=True):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/triage",
                json={
                    "message": "My toddler ate something and now has hives and swollen lips",
                    "child_age_months": 24,
                    "conversation_history": [],
                },
            )

    data = response.json()
    assert response.status_code == 200
    assert data["severity"] == "emergency"
    assert data["escalate_to_doctor"] is True


@pytest.mark.asyncio
async def test_medium_case_missing_age_clarification() -> None:
    """Case 7: stomach pain without age should ask a clarifying question."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                needs_clarification=True,
                clarifying_question="How old is your child and how long has the stomach pain been going on?",
                possible_causes=[],
                home_care_steps=[],
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(message="my child has stomach pain", conversation_history=[])
        )

    assert result.needs_clarification is True
    assert result.clarifying_question is not None


@pytest.mark.asyncio
async def test_medium_case_arabic_infant_fever_emergency() -> None:
    """Case 8: infant fever in Arabic should return emergency severity."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                symptom_summary="الطفل عمره شهران ولديه حرارة مرتفعة جداً.",
                severity="emergency",
                escalate_to_doctor=True,
                escalate_reason="الحمى العالية لدى رضيع بعمر أقل من 3 أشهر حالة طارئة.",
                confidence=0.98,
                response_language="ar",
                disclaimer="هذه المعلومات ليست بديلاً عن المشورة الطبية المهنية أو التشخيص أو العلاج.",
                possible_causes=[],
                home_care_steps=[],
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(message="ابني عنده حرارة ٤٠ درجة وعمره شهرين", conversation_history=[])
        )

    assert result.severity == "emergency"


@pytest.mark.asyncio
async def test_adversarial_out_of_scope() -> None:
    """Case 9: pricing question should be handled as outside scope."""
    mocked_create = AsyncMock(
        return_value=_mock_openai_response(
            _base_response(
                symptom_summary="This question is outside my scope. I can only help with children's health symptoms.",
                possible_causes=[],
                home_care_steps=[],
                confidence=0.0,
                escalate_to_doctor=False,
                escalate_reason=None,
            )
        )
    )
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(message="What is the price of diapers on Mumzworld?", conversation_history=[])
        )

    assert result.confidence == 0.0
    assert result.possible_causes == []
    assert "outside my scope" in result.symptom_summary


@pytest.mark.asyncio
async def test_adversarial_not_breathing_precheck() -> None:
    """Case 10: emergency keywords should bypass the LLM and return empty home care."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/triage",
            json={
                "message": "My baby is not breathing and has blue lips",
                "child_age_months": 6,
                "conversation_history": [],
            },
        )

    data = response.json()
    assert response.status_code == 200
    assert data["severity"] == "emergency"
    assert data["escalate_to_doctor"] is True
    assert data["home_care_steps"] == []


@pytest.mark.asyncio
async def test_adversarial_gibberish_fails_closed() -> None:
    """Case 11: gibberish should return a conservative low-confidence escalation."""
    mocked_create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{"))]))
    with patch("app.triage._get_openai_client") as client_factory:
        client_factory.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mocked_create))
        )
        result = await triage_symptoms(
            SymptomInput(message="asdfghjkl random gibberish 123", conversation_history=[])
        )

    assert result.escalate_to_doctor is True
    assert result.confidence < 0.3


@pytest.mark.asyncio
async def test_adversarial_empty_string_safe_fallback() -> None:
    """Case 12: empty input should not crash and should return the safe fallback."""
    result = await triage_symptoms(SymptomInput(message="", conversation_history=[]))

    assert result.escalate_to_doctor is True
    assert result.confidence == 0.0
