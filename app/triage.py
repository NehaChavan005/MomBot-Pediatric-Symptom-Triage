"""Core triage workflow for MomBot."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.language import detect_language
from app.prompts import get_system_prompt
from app.schema import SymptomInput, TriageResponse

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def _get_openai_client() -> AsyncOpenAI:
    """Build and return the shared OpenAI async client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)


def _build_disclaimer(language: str) -> str:
    """Return the required medical disclaimer in the target language."""
    if language == "ar":
        return "هذه المعلومات ليست بديلاً عن المشورة الطبية المهنية أو التشخيص أو العلاج."
    return "This is not a substitute for professional medical advice, diagnosis, or treatment."


def safe_fallback_response(language: str, message: str | None = None) -> TriageResponse:
    """Return a conservative fallback response when safe structured triage is not possible."""
    fallback_message = (
        "لم أتمكن من تحليل الحالة بشكل آمن. يرجى استشارة طبيب فوراً."
        if language == "ar"
        else "I could not safely analyze this - please consult a doctor."
    )

    return TriageResponse(
        symptom_summary=fallback_message,
        severity="high",
        possible_causes=[],
        home_care_steps=[],
        escalate_to_doctor=True,
        escalate_reason=fallback_message,
        confidence=0.0,
        disclaimer=_build_disclaimer(language),
        response_language=language,
        needs_clarification=False,
        clarifying_question=None,
    )


def _out_of_scope_response(language: str) -> TriageResponse:
    """Return a structured out-of-scope response for non-health questions."""
    summary = (
        "هذا السؤال خارج نطاقي. أستطيع المساعدة فقط في أعراض صحة الأطفال."
        if language == "ar"
        else "This question is outside my scope. I can only help with children's health symptoms."
    )
    return TriageResponse(
        symptom_summary=summary,
        severity="low",
        possible_causes=[],
        home_care_steps=[],
        escalate_to_doctor=False,
        escalate_reason=None,
        confidence=0.0,
        disclaimer=_build_disclaimer(language),
        response_language=language,
        needs_clarification=False,
        clarifying_question=None,
    )


def emergency_response(language: str, message: str) -> TriageResponse:
    """Return an immediate emergency response without calling the language model."""
    if language == "ar":
        summary = "تم رصد أعراض طارئة تتطلب تقييماً طبياً فورياً."
        reason = "قد تكون هناك علامات خطر طبية طارئة. توجهي إلى الطبيب أو الطوارئ فوراً."
    else:
        summary = "I identified emergency warning signs in the symptoms you described."
        reason = "These symptoms may indicate a medical emergency. Please seek immediate medical care now."

    return TriageResponse(
        symptom_summary=summary,
        severity="emergency",
        possible_causes=[],
        home_care_steps=[],
        escalate_to_doctor=True,
        escalate_reason=reason,
        confidence=1.0,
        disclaimer=_build_disclaimer(language),
        response_language=language,
        needs_clarification=False,
        clarifying_question=None,
    )


def _build_messages(user_input: SymptomInput, language: str) -> list[dict[str, str]]:
    """Build the chat history sent to the model."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": get_system_prompt(language)},
    ]

    if user_input.child_age_months is not None:
        age_context = (
            f"Child age: {user_input.child_age_months} months."
            if language == "en"
            else f"عمر الطفل: {user_input.child_age_months} شهراً."
        )
        messages.append({"role": "system", "content": age_context})

    for turn in user_input.conversation_history:
        if turn.content:
            messages.append({"role": turn.role, "content": turn.content})

    if user_input.message.strip():
        messages.append({"role": "user", "content": user_input.message.strip()})

    return messages


def _contains_any(text: str, phrases: list[str]) -> bool:
    """Return True when any phrase is present in the normalized text."""
    return any(phrase in text for phrase in phrases)


def _extract_age_months(message: str) -> int | None:
    """Infer child age in months from common English and Arabic phrasing."""
    lowered = message.lower()

    month_match = re.search(r"(\d+)\s*[- ]?\s*(?:month|months|month-old)", lowered)
    if month_match:
        return int(month_match.group(1))

    year_match = re.search(r"(\d+)\s*[- ]?\s*(?:year|years|year-old)", lowered)
    if year_match:
        return int(year_match.group(1)) * 12

    arabic_month_match = re.search(r"(\d+|[٠-٩]+)\s*(?:شهر|شهور|أشهر|شهراً|شهرين)", message)
    if arabic_month_match:
        return _to_western_digits(arabic_month_match.group(1))

    arabic_year_match = re.search(r"(\d+|[٠-٩]+)\s*(?:سنة|سنوات)", message)
    if arabic_year_match:
        return _to_western_digits(arabic_year_match.group(1)) * 12

    return None


def _to_western_digits(value: str) -> int:
    """Convert Arabic numerals to an integer."""
    translation = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return int(value.translate(translation))


def _extract_temperature_c(message: str) -> float | None:
    """Extract a temperature in Celsius from English or Arabic text."""
    normalized = message.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    match = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*c", normalized.lower())
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+(?:\.\d+)?)\s*درجة", normalized)
    if match:
        return float(match.group(1))

    return None


def _is_gibberish(message: str) -> bool:
    """Flag obviously low-information input that should fail closed."""
    cleaned = re.sub(r"[^A-Za-z\u0600-\u06FF]", "", message)
    if len(cleaned) < 4:
        return True
    vowel_count = sum(1 for char in cleaned.lower() if char in "aeiou")
    return " " not in message and vowel_count == 0 and not re.search(r"[\u0600-\u06FF]", cleaned)


def heuristic_triage_response(user_input: SymptomInput, language: str) -> TriageResponse:
    """Provide a conservative local triage result when the model is unavailable."""
    message = user_input.message.strip()
    lowered = message.lower()
    age_months = user_input.child_age_months if user_input.child_age_months is not None else _extract_age_months(message)
    temperature_c = _extract_temperature_c(message)

    if not message:
        return safe_fallback_response(language)

    if _is_gibberish(message):
        return safe_fallback_response(language)

    if _contains_any(lowered, ["price", "product", "diapers", "baby wipes", "mumzworld"]):
        return _out_of_scope_response(language)

    if _contains_any(lowered, ["not breathing", "blue lips", "difficulty breathing", "seizure", "unconscious", "choking"]):
        return emergency_response(language, message)

    if _contains_any(lowered, ["swollen lips", "hives"]) or _contains_any(message, ["تورم الشفاه", "شرى", "حساسية شديدة"]):
        return emergency_response(language, message)

    if age_months is not None and age_months < 3 and temperature_c is not None and temperature_c >= 38.0:
        return emergency_response(language, message)

    if _contains_any(lowered, ["stomach pain", "abdominal pain"]) or _contains_any(message, ["ألم في البطن", "وجع بطن"]):
        if age_months is None:
            question = (
                "How old is your child and how long has the stomach pain been going on?"
                if language == "en"
                else "كم عمر طفلك، ومنذ متى بدأ ألم البطن؟"
            )
            summary = (
                "I need one more detail before giving safe triage."
                if language == "en"
                else "أحتاج إلى معلومة إضافية واحدة قبل إعطاء فرز آمن."
            )
            return TriageResponse(
                symptom_summary=summary,
                severity="medium",
                possible_causes=[],
                home_care_steps=[],
                escalate_to_doctor=False,
                escalate_reason=None,
                confidence=0.45,
                disclaimer=_build_disclaimer(language),
                response_language=language,
                needs_clarification=True,
                clarifying_question=question,
            )

    if _contains_any(lowered, ["runny nose", "sneez", "mild cough"]) or _contains_any(message, ["رشح", "سعال خفيف", "عطاس"]):
        summary = (
            "This sounds like a mild upper respiratory symptom pattern."
            if language == "en"
            else "يبدو هذا نمطاً خفيفاً من أعراض الجهاز التنفسي العلوي."
        )
        causes = (
            ["common cold", "mild viral infection"]
            if language == "en"
            else ["نزلة برد", "عدوى فيروسية خفيفة"]
        )
        steps = (
            ["Offer fluids regularly", "Let your child rest", "Monitor for fever or breathing changes"]
            if language == "en"
            else ["قدمي السوائل بانتظام", "دعي طفلك يرتاح", "راقبي الحمى أو تغيرات التنفس"]
        )
        return TriageResponse(
            symptom_summary=summary,
            severity="low",
            possible_causes=causes,
            home_care_steps=steps,
            escalate_to_doctor=False,
            escalate_reason=None,
            confidence=0.72,
            disclaimer=_build_disclaimer(language),
            response_language=language,
            needs_clarification=False,
            clarifying_question=None,
        )

    if _contains_any(lowered, ["diarrhea"]) or _contains_any(message, ["إسهال"]):
        summary = (
            "This may be a mild gastrointestinal illness if your child is still drinking."
            if language == "en"
            else "قد يكون هذا مرضاً هضمياً خفيفاً إذا كان الطفل ما زال يشرب السوائل."
        )
        causes = (
            ["viral stomach bug", "mild food-related irritation"]
            if language == "en"
            else ["عدوى فيروسية بالمعدة", "تهيج خفيف مرتبط بالطعام"]
        )
        steps = (
            ["Offer oral fluids often", "Watch for dehydration", "Seek care if symptoms worsen"]
            if language == "en"
            else ["قدمي السوائل الفموية بشكل متكرر", "راقبي علامات الجفاف", "اطلبي الرعاية إذا ساءت الأعراض"]
        )
        return TriageResponse(
            symptom_summary=summary,
            severity="low",
            possible_causes=causes,
            home_care_steps=steps,
            escalate_to_doctor=False,
            escalate_reason=None,
            confidence=0.7,
            disclaimer=_build_disclaimer(language),
            response_language=language,
            needs_clarification=False,
            clarifying_question=None,
        )

    if _contains_any(lowered, ["rash"]) or _contains_any(message, ["طفح", "حساسية جلدية"]):
        question = (
            "How long has the rash been there, and does your child have fever or itching?"
            if language == "en"
            else "منذ متى ظهر الطفح، وهل لدى الطفل حمى أو حكة؟"
        )
        summary = (
            "A localized rash may be mild, but I need one more detail."
            if language == "en"
            else "قد يكون الطفح الموضعي خفيفاً، لكنني أحتاج إلى معلومة إضافية واحدة."
        )
        return TriageResponse(
            symptom_summary=summary,
            severity="low",
            possible_causes=["skin irritation", "insect bite"] if language == "en" else ["تهيج جلدي", "لدغة حشرة"],
            home_care_steps=["Keep the area clean", "Monitor for spread or fever"] if language == "en" else ["حافظي على نظافة المنطقة", "راقبي انتشار الطفح أو ظهور الحمى"],
            escalate_to_doctor=False,
            escalate_reason=None,
            confidence=0.6,
            disclaimer=_build_disclaimer(language),
            response_language=language,
            needs_clarification=True,
            clarifying_question=question,
        )

    if temperature_c is not None and age_months is not None and age_months <= 12 and temperature_c >= 39.0:
        summary = (
            "A high fever in an infant should be medically assessed."
            if language == "en"
            else "ينبغي تقييم الحمى المرتفعة عند الرضيع طبياً."
        )
        reason = (
            "Your child is young and has a high fever, so a doctor review is recommended."
            if language == "en"
            else "عمر الطفل صغير ولديه حمى مرتفعة، لذلك يُنصح بمراجعة الطبيب."
        )
        return TriageResponse(
            symptom_summary=summary,
            severity="high",
            possible_causes=["viral infection", "other febrile illness"] if language == "en" else ["عدوى فيروسية", "مرض آخر مصحوب بالحمى"],
            home_care_steps=["Offer fluids", "Keep monitoring temperature"] if language == "en" else ["قدمي السوائل", "استمري في مراقبة الحرارة"],
            escalate_to_doctor=True,
            escalate_reason=reason,
            confidence=0.78,
            disclaimer=_build_disclaimer(language),
            response_language=language,
            needs_clarification=False,
            clarifying_question=None,
        )

    return safe_fallback_response(language)


def _triage_response_json_schema() -> dict[str, Any]:
    """Return the JSON schema used for OpenAI structured output."""
    return {
        "name": "triage_response",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "symptom_summary": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high", "emergency"]},
                "possible_causes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
                "home_care_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "escalate_to_doctor": {"type": "boolean"},
                "escalate_reason": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
                "disclaimer": {"type": "string"},
                "response_language": {"type": "string", "enum": ["en", "ar"]},
                "needs_clarification": {"type": "boolean"},
                "clarifying_question": {"type": ["string", "null"]},
            },
            "required": [
                "symptom_summary",
                "severity",
                "possible_causes",
                "home_care_steps",
                "escalate_to_doctor",
                "escalate_reason",
                "confidence",
                "disclaimer",
                "response_language",
                "needs_clarification",
                "clarifying_question",
            ],
        },
        "strict": True,
    }


async def _call_openai_for_triage(client: AsyncOpenAI, messages: list[dict[str, str]]) -> TriageResponse:
    """Call OpenAI using the most reliable structured-output path available."""
    if hasattr(client, "beta") and hasattr(client.beta, "chat") and hasattr(client.beta.chat, "completions"):
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=TriageResponse,
            temperature=0.2,
        )
        parsed = response.choices[0].message.parsed
        if isinstance(parsed, TriageResponse):
            return parsed
        if parsed is not None:
            return TriageResponse.model_validate(parsed)
        raise ValueError("OpenAI parse response did not contain parsed structured data")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": _triage_response_json_schema(),
        },
        temperature=0.2,
    )
    raw_content = _extract_message_content(response)
    parsed = json.loads(raw_content)
    return TriageResponse.model_validate(parsed)


def _extract_message_content(response: Any) -> str:
    """Extract the JSON string content from an OpenAI chat completion response."""
    try:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif hasattr(item, "type") and getattr(item, "type", None) == "text":
                    text_parts.append(getattr(item, "text", ""))
            combined = "".join(part for part in text_parts if part)
            if combined:
                return combined
    except Exception:
        LOGGER.exception("Failed to extract content from OpenAI response.")
        raise
    raise ValueError("OpenAI response did not contain message content")


async def triage_symptoms(input: SymptomInput) -> TriageResponse:
    """Run the full triage workflow with language detection, prompting, and safe validation."""
    language = detect_language(input.message)

    if not input.message.strip():
        LOGGER.warning("Received empty symptom input; returning safe fallback.")
        return safe_fallback_response(language)

    messages = _build_messages(input, language)

    try:
        client = _get_openai_client()
        return await _call_openai_for_triage(client, messages)
    except ValidationError:
        LOGGER.exception("Structured response validation failed; using heuristic fallback.")
        return heuristic_triage_response(input, language)
    except Exception:
        LOGGER.exception("OpenAI triage failed; using heuristic fallback.")
        return heuristic_triage_response(input, language)
