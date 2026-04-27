"""FastAPI application entrypoint for the MomBot API."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.language import detect_language
from app.schema import SymptomInput
from app.triage import emergency_response, safe_fallback_response, triage_symptoms
from data.guidelines import check_emergency_keywords

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="MomBot", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a simple health status for the backend."""
    return {"status": "ok", "model": "gpt-4o"}


@app.post("/triage")
async def triage(input: SymptomInput):
    """Triage pediatric symptoms with an emergency shortcut and safe fallback handling."""
    language = detect_language(input.message)

    try:
        if check_emergency_keywords(input.message):
            LOGGER.info("Emergency keyword pre-check triggered.")
            return emergency_response(language, input.message)

        return await triage_symptoms(input)
    except ValidationError:
        LOGGER.exception("Request validation failed inside triage handler.")
        fallback = safe_fallback_response(language)
        return JSONResponse(status_code=422, content=fallback.model_dump())
    except Exception:
        LOGGER.exception("Unhandled triage error; returning safe fallback.")
        fallback = safe_fallback_response(language)
        return JSONResponse(status_code=500, content=fallback.model_dump())
