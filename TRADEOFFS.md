# TRADEOFFS

## 1. Why This Problem

Pediatric symptom triage is the highest-leverage AI problem for Mumzworld because it sits directly at the intersection of urgency, trust, and audience relevance. Mumzworld serves mothers, and one of the most emotionally charged moments in that customer journey is uncertainty about a child's symptoms. A triage assistant can create immediate value by helping mothers decide whether a situation is likely low risk, needs home monitoring, or requires prompt medical attention. That makes it both mission-aligned and sticky in a way a generic shopping assistant is not.

## 2. Why GPT-4o

GPT-4o is a strong fit because it handles natural language well across English and Arabic, follows structured-output instructions reliably, and performs well on messy real-world health descriptions without requiring rigid form inputs. For this assessment, structured JSON support matters as much as language quality because the output has to validate cleanly into a safety-oriented schema. GPT-4o also offers a practical balance between quality, latency, and implementation simplicity.

## 3. Architecture Decisions

FastAPI was chosen for the backend because it makes request validation, async I/O, and typed API contracts straightforward. Streamlit was chosen for the frontend because it is fast to build, easy to demo, and well suited to a conversational assessment project. Splitting the system into API and UI instead of writing a single script keeps responsibilities clean, makes testing easier, and better reflects production design. Pydantic validation is central because it turns the LLM response into a contract: either it fits the schema or the system falls back safely.

## 4. Uncertainty Handling

The system uses a layered safety net instead of trusting the model blindly. First, a rule-based emergency keyword pre-check short-circuits obvious emergencies before any model call. Second, the model is asked to produce an explicit confidence score and escalation rationale. Third, if parsing or validation fails, the application returns a conservative fallback response that escalates to a doctor. Together, those layers reduce the chance that a malformed or overconfident model response becomes unsafe user guidance.

## 5. Multilingual Approach

Using `langdetect` plus direct prompt control is stronger than translate-then-process for this use case. Translation introduces an avoidable failure point, can flatten nuance, and may degrade emergency cues or colloquial symptom wording. By keeping the original message intact and asking the model to respond in the same language, the system preserves context and avoids unnecessary linguistic drift. Prompting Arabic responses in Modern Standard Arabic also improves consistency and readability.

## 6. What Was Cut

- RAG over real WHO or CDC documents with citation grounding was intentionally skipped to keep the project self-contained and offline-friendly.
- Voice input and speech-to-text were omitted even though they could be useful for stressed caregivers on mobile.
- User authentication and conversation persistence were not added because they are not necessary for the core assessment.
- Region-specific care pathways, provider lookup, and emergency hotline localization were left out to avoid pretending to have clinical operations integration.

## 7. What's Next

- Add retrieval over curated pediatric guidance documents with explicit citations and versioning.
- Improve Arabic performance with dialect-aware normalization and test sets from Gulf Arabic phrasing.
- Add a clinician-reviewed policy layer for symptom-specific escalation thresholds beyond generic prompting.
