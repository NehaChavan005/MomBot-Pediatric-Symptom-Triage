"""Language detection helpers for English and Arabic user messages."""

from __future__ import annotations

import logging
import re

from langdetect import DetectorFactory, LangDetectException, detect

LOGGER = logging.getLogger(__name__)
ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")

# langdetect is non-deterministic without a fixed seed.
DetectorFactory.seed = 0


def contains_arabic(text: str) -> bool:
    """Return True when the text contains Arabic script characters."""
    return bool(text and ARABIC_CHAR_PATTERN.search(text))


def detect_language(text: str) -> str:
    """Detect whether the text is English or Arabic, defaulting safely to English."""
    if not text or not text.strip():
        return "en"

    if contains_arabic(text):
        return "ar"

    try:
        detected = detect(text)
    except LangDetectException:
        LOGGER.warning("Language detection failed; defaulting to English.")
        return "en"
    except Exception:
        LOGGER.exception("Unexpected language detection error; defaulting to English.")
        return "en"

    return "ar" if detected == "ar" else "en"
