"""Streamlit frontend for the MomBot pediatric symptom triage assistant."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import os
from typing import Any
from pathlib import Path
from html import escape, unescape

import requests
import streamlit as st

# Ensure the project root is importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.language import detect_language
from app.schema import SymptomInput
from app.triage import emergency_response, safe_fallback_response, triage_symptoms
from data.guidelines import check_emergency_keywords

LOGGER = logging.getLogger(__name__)
API_URL = os.getenv("MOMBOT_API_URL", "http://localhost:8000/triage")
MUMZWORLD_PINK = "#E91E8C"
MUMZWORLD_SOFT_PINK = "#FCE4F1"
MUMZWORLD_BLUSH = "#FFF6FA"
MUMZWORLD_TEXT = "#5F4B57"
MUMZWORLD_MUTED = "#8F7A86"
MUMZWORLD_BORDER = "#F3D7E6"
MUMZWORLD_PEACH = "#FFE7D6"
MUMZWORLD_MINT = "#EAF8F2"
MUMZWORLD_LAVENDER = "#F6ECFF"
MUMZWORLD_ROSE_BG = "#F7E6EF"
MUMZWORLD_DUSTY = "#EFD6E3"


def _initialize_state() -> None:
    """Initialize required Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "Auto"
    if "child_age_months_text" not in st.session_state:
        st.session_state.child_age_months_text = ""


def _clean_display_text(value: Any) -> str:
    """Convert any stored value into safe plain text for UI rendering."""
    text = "" if value is None else str(value)
    text = unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _get_ui_direction(ui_language: str, latest_user_text: str | None) -> tuple[str, str]:
    """Return the HTML direction and language code for the UI."""
    if ui_language == "AR":
        return "rtl", "ar"
    if ui_language == "EN":
        return "ltr", "en"

    detected = detect_language(latest_user_text or "")
    return ("rtl", "ar") if detected == "ar" else ("ltr", "en")


def _inject_styles(direction: str) -> None:
    """Inject custom styling for Mumzworld branding and Arabic RTL support."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, #f6cddd 0%, transparent 26%),
                    radial-gradient(circle at top right, #fbd7c8 0%, transparent 24%),
                    radial-gradient(circle at bottom right, #e7d7fb 0%, transparent 22%),
                    linear-gradient(180deg, #faeef4 0%, #f7e8f0 45%, #f9f0ea 100%);
                color: {MUMZWORLD_TEXT};
            }}
            [data-testid="stAppViewContainer"] {{
                background: transparent;
            }}
            [data-testid="stSidebar"] {{
                background:
                    radial-gradient(circle at top right, #ffd6e8 0%, transparent 26%),
                    linear-gradient(180deg, #fdf5f9 0%, #f6e7ef 100%);
                border-right: 1px solid {MUMZWORLD_BORDER};
            }}
            [data-testid="stSidebar"] > div:first-child {{
                background: transparent;
            }}
            [data-testid="stSidebar"] * {{
                color: {MUMZWORLD_TEXT};
            }}
            [data-testid="stHeader"] {{
                background: rgba(249, 237, 243, 0.92);
                backdrop-filter: blur(10px);
            }}
            [data-testid="stChatMessage"] {{
                background: transparent;
                border: none;
            }}
            [data-testid="stChatMessage"][data-testid*="user"] {{
                background: linear-gradient(135deg, #fff1f8 0%, #ffe7f2 100%);
                border: 1px solid #f7d4e7;
                border-radius: 20px;
                padding: 0.35rem 0.7rem;
                box-shadow: 0 10px 24px rgba(217, 138, 177, 0.08);
            }}
            [data-testid="stChatMessage"][data-testid*="assistant"] {{
                background: linear-gradient(135deg, #ffffff 0%, #fffafc 100%);
                border: 1px solid #f1dbe7;
                border-radius: 20px;
                padding: 0.35rem 0.7rem;
                box-shadow: 0 10px 24px rgba(190, 160, 176, 0.06);
            }}
            [data-testid="stChatInput"] {{
                background: linear-gradient(180deg, #fff9fc 0%, #f9eef4 100%);
                border: 1px solid {MUMZWORLD_BORDER};
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(190, 112, 153, 0.08);
            }}
            [data-testid="stChatInput"] textarea {{
                color: {MUMZWORLD_TEXT};
            }}
            [data-testid="stTextInputRootElement"] > div,
            [data-testid="stNumberInput"] > div,
            [data-testid="stRadio"] {{
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid {MUMZWORLD_BORDER};
                border-radius: 16px;
                padding: 0.35rem;
            }}
            .stButton > button {{
                background: linear-gradient(135deg, #f8b4d4 0%, {MUMZWORLD_PINK} 100%);
                color: white;
                border: none;
                border-radius: 999px;
                box-shadow: 0 10px 24px rgba(218, 77, 151, 0.22);
            }}
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 980px;
            }}
            .mom-title {{
                color: {MUMZWORLD_PINK};
                font-weight: 800;
                margin-bottom: 0.15rem;
                letter-spacing: -0.02em;
            }}
            .mom-subtitle {{
                color: {MUMZWORLD_MUTED};
                margin-bottom: 1.25rem;
            }}
            .mom-card {{
                border: 1px solid {MUMZWORLD_BORDER};
                border-radius: 22px;
                padding: 1.1rem 1.2rem;
                background: linear-gradient(180deg, #ffffff 0%, {MUMZWORLD_BLUSH} 100%);
                box-shadow: 0 14px 34px rgba(194, 133, 164, 0.10);
                direction: {direction};
                text-align: {"right" if direction == "rtl" else "left"};
            }}
            .mom-banner {{
                background: linear-gradient(180deg, #fff2f8 0%, #ffe8f3 100%);
                border-left: 5px solid {MUMZWORLD_PINK};
                padding: 0.9rem 1rem;
                border-radius: 14px;
                margin: 0.75rem 0;
                color: #8a0d52;
                direction: {direction};
                box-shadow: 0 10px 24px rgba(206, 99, 150, 0.08);
            }}
            .mom-disclaimer {{
                color: {MUMZWORLD_MUTED};
                font-style: italic;
                margin-top: 0.8rem;
            }}
            .severity-badge {{
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                color: white;
                font-size: 0.85rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
                letter-spacing: 0.03em;
            }}
            .mom-section-label {{
                color: {MUMZWORLD_TEXT};
                font-weight: 700;
                margin-top: 1rem;
                margin-bottom: 0.45rem;
            }}
            .mom-shell {{
                background:
                    linear-gradient(180deg, rgba(255, 248, 252, 0.82) 0%, rgba(246, 233, 240, 0.94) 100%);
                border: 1px solid rgba(232, 200, 218, 0.98);
                border-radius: 28px;
                padding: 1.15rem;
                box-shadow: 0 18px 40px rgba(177, 119, 149, 0.14);
            }}
            .mom-hero {{
                display: grid;
                grid-template-columns: 1.2fr 0.8fr;
                gap: 1rem;
                margin-bottom: 1rem;
            }}
            .mom-hero-panel {{
                background: linear-gradient(135deg, rgba(247, 214, 231, 0.98) 0%, rgba(255, 246, 250, 0.92) 100%);
                border: 1px solid {MUMZWORLD_DUSTY};
                border-radius: 22px;
                padding: 1rem 1.1rem;
                box-shadow: 0 14px 28px rgba(181, 125, 154, 0.12);
            }}
            .mom-hero-highlights {{
                display: flex;
                gap: 0.6rem;
                flex-wrap: wrap;
                margin-top: 0.85rem;
            }}
            .mom-pill {{
                padding: 0.4rem 0.7rem;
                border-radius: 999px;
                font-size: 0.86rem;
                font-weight: 700;
                color: {MUMZWORLD_TEXT};
            }}
            .mom-pill-pink {{
                background: {MUMZWORLD_SOFT_PINK};
            }}
            .mom-pill-peach {{
                background: {MUMZWORLD_PEACH};
            }}
            .mom-pill-mint {{
                background: {MUMZWORLD_MINT};
            }}
            .mom-pill-lavender {{
                background: {MUMZWORLD_LAVENDER};
            }}
            .mom-accent-note {{
                background: linear-gradient(135deg, #ffe9c8 0%, #fddcaa 100%);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                color: #765213;
                border: 1px solid #eecb84;
                font-weight: 600;
            }}
            .stProgress > div > div > div > div {{
                background: linear-gradient(90deg, #f7b6d2 0%, {MUMZWORLD_PINK} 100%);
            }}
            @media (max-width: 768px) {{
                .mom-hero {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _severity_color(severity: str) -> str:
    """Map severity labels to UI colors."""
    mapping = {
        "low": "#2E7D32",
        "medium": "#F9A825",
        "high": "#EF6C00",
        "emergency": "#C62828",
    }
    return mapping.get(severity, "#6B7280")


def _ui_copy(language: str) -> dict[str, str]:
    """Return localized UI labels for the structured response card."""
    if language == "ar":
        return {
            "possible_causes": "الأسباب المحتملة",
            "home_care_steps": "خطوات العناية المنزلية",
            "doctor_escalation": "ضرورة مراجعة الطبيب",
            "confidence": "مستوى الثقة",
        }

    return {
        "possible_causes": "Possible causes",
        "home_care_steps": "Home care steps",
        "doctor_escalation": "Doctor escalation",
        "confidence": "Confidence",
    }


def _render_structured_response(response: dict[str, Any]) -> None:
    """Render a structured triage card or clarifying question."""
    language = response.get("response_language", "en")
    labels = _ui_copy(language)
    is_out_of_scope = not response.get("possible_causes") and not response.get("home_care_steps") and not response.get("escalate_to_doctor")
    summary_text = _clean_display_text(response.get("symptom_summary", ""))

    if response.get("needs_clarification") and response.get("clarifying_question"):
        question_text = _clean_display_text(response.get("clarifying_question"))
        st.markdown(
            f"""
            <div class="mom-card">
                <strong>{escape(question_text)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if is_out_of_scope:
        st.info(summary_text)
        return

    severity = response.get("severity", "high")
    badge_color = _severity_color(severity)
    possible_causes = response.get("possible_causes") or []
    home_care_steps = response.get("home_care_steps") or []
    escalate = response.get("escalate_to_doctor", False)
    escalate_reason = response.get("escalate_reason")
    confidence = float(response.get("confidence", 0.0))
    badge_html = "" if is_out_of_scope else f'<div class="severity-badge" style="background:{badge_color};">{severity.upper()}</div>'

    st.markdown(
        f"""
        <div class="mom-card">
            {badge_html}
            <p><strong>{escape(summary_text)}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if possible_causes:
        st.markdown(
            f'<div class="mom-section-label">{labels["possible_causes"]}</div>',
            unsafe_allow_html=True,
        )
        for cause in possible_causes:
            st.markdown(f"- {_clean_display_text(cause)}")

    if home_care_steps:
        st.markdown(
            f'<div class="mom-section-label">{labels["home_care_steps"]}</div>',
            unsafe_allow_html=True,
        )
        for index, step in enumerate(home_care_steps, start=1):
            st.markdown(f"{index}. {_clean_display_text(step)}")

    if escalate and escalate_reason:
        st.markdown(
            f'<div class="mom-banner"><strong>{labels["doctor_escalation"]}:</strong> {escape(_clean_display_text(escalate_reason))}</div>',
            unsafe_allow_html=True,
        )

    if not is_out_of_scope:
        st.markdown(
            f'<div class="mom-section-label">{labels["confidence"]}</div>',
            unsafe_allow_html=True,
        )
        st.progress(max(0.0, min(confidence, 1.0)))


def _build_api_payload(message: str, child_age_months: int | None) -> dict[str, Any]:
    """Build the API payload from the current Streamlit session state."""
    history = []
    for item in st.session_state.messages:
        history.append(
            {
                "role": item["role"],
                "content": item.get("api_content"),
            }
        )

    return {
        "message": message,
        "child_age_months": child_age_months,
        "conversation_history": history,
    }


def _parse_child_age_months(raw_value: str) -> int | None:
    """Parse the optional child age field from the sidebar."""
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None

    try:
        parsed = int(cleaned)
    except ValueError:
        LOGGER.warning("Invalid child age input provided: %s", raw_value)
        return None

    if parsed < 0:
        LOGGER.warning("Negative child age input provided: %s", raw_value)
        return None

    return parsed


def _call_backend(payload: dict[str, Any]) -> dict[str, Any]:
    """Call the FastAPI backend and return the JSON response."""
    try:
        response = requests.post(API_URL, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        LOGGER.exception("Backend request failed. Falling back to in-process triage.")
        return _call_local_triage(payload)


def _call_local_triage(payload: dict[str, Any]) -> dict[str, Any]:
    """Run triage directly inside the Streamlit process when the API is unavailable."""
    language = detect_language(payload.get("message", ""))

    try:
        symptom_input = SymptomInput.model_validate(payload)

        if check_emergency_keywords(symptom_input.message):
            return emergency_response(language, symptom_input.message).model_dump()

        return asyncio.run(triage_symptoms(symptom_input)).model_dump()
    except Exception:
        LOGGER.exception("Local triage fallback failed.")
        return safe_fallback_response(language).model_dump()


def main() -> None:
    """Run the Streamlit chat application."""
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="MomBot", page_icon="👶", layout="wide")
    _initialize_state()

    st.sidebar.title("Settings")
    st.session_state.child_age_months_text = st.sidebar.text_input(
        "Child age in months (optional)",
        value=st.session_state.child_age_months_text,
        placeholder="e.g. 24",
    )
    child_age_months = _parse_child_age_months(st.session_state.child_age_months_text)
    if st.session_state.child_age_months_text.strip() and child_age_months is None:
        st.sidebar.warning("Please enter a valid whole number for age in months.")

    st.session_state.ui_language = st.sidebar.radio(
        "UI Language / لغة الواجهة",
        options=["Auto", "EN", "AR"],
        index=["Auto", "EN", "AR"].index(st.session_state.ui_language),
        horizontal=True,
    )

    latest_user_text = None
    for item in reversed(st.session_state.messages):
        if item["role"] == "user":
            latest_user_text = item["content"]
            break

    direction, _ = _get_ui_direction(st.session_state.ui_language, latest_user_text)
    _inject_styles(direction)

    st.markdown('<div class="mom-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mom-hero">
            <div class="mom-hero-panel">
                <h1 class="mom-title">MomBot 👶 — Pediatric Symptom Triage</h1>
                <div class="mom-subtitle">Gentle symptom guidance for mothers in English and Arabic.</div>
                <div class="mom-hero-highlights">
                    <span class="mom-pill mom-pill-pink">Bilingual support</span>
                    <span class="mom-pill mom-pill-peach">Structured triage</span>
                    <span class="mom-pill mom-pill-mint">Emergency aware</span>
                </div>
            </div>
            <div class="mom-accent-note">
                Share symptoms naturally. MomBot will ask a follow-up question when needed and highlight when urgent care matters.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("triage_response"):
                _render_structured_response(message["triage_response"])
            else:
                st.write(_clean_display_text(message.get("content")))

    user_prompt = st.chat_input("Type symptoms here / اكتبي الأعراض هنا")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt, "api_content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    payload = _build_api_payload(user_prompt, child_age_months)
    triage_response = _call_backend(payload)

    assistant_api_content = (
        triage_response.get("clarifying_question")
        if triage_response.get("needs_clarification")
        else triage_response.get("symptom_summary")
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": _clean_display_text(assistant_api_content),
            "api_content": _clean_display_text(assistant_api_content),
            "triage_response": triage_response,
        }
    )

    with st.chat_message("assistant"):
        _render_structured_response(triage_response)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
