# MomBot

MomBot is a bilingual pediatric symptom triage assistant for Mumzworld, designed for mothers who want fast, structured guidance when describing a child's symptoms in either English or Arabic. It does not diagnose disease, but it can ask a clarifying follow-up question, highlight emergency signals, suggest home-care steps when appropriate, and escalate clearly when a doctor is needed.

## Quick Start

Set up the project in under 5 minutes:

```bash
git clone <your-repo-url>
cd mombot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your `OPENAI_API_KEY` to `.env`, then start the backend and UI in two terminals:

```bash
uvicorn app.main:app --reload
```

```bash
streamlit run ui/streamlit_app.py
```

## Example Inputs

English:

- `My 2-year-old has a runny nose and mild fever of 37.5°C`
- `My baby has had diarrhea for 2 days`
- `My toddler ate something and now has hives and swollen lips`

Arabic:

- `طفلي عنده سعال منذ يومين`
- `ابني عنده حرارة ٤٠ درجة وعمره شهرين`

## Architecture

```text
┌──────────────────────┐
│   Streamlit Frontend │
│ bilingual chat UI    │
└──────────┬───────────┘
           │ HTTP POST /triage
           ▼
┌──────────────────────┐
│    FastAPI Backend   │
│ input validation     │
│ emergency pre-check  │
└──────────┬───────────┘
           │
           ├──────────────► data/guidelines.py
           │               keyword emergency rules
           │
           ├──────────────► app/language.py
           │               EN/AR detection
           │
           └──────────────► app/triage.py
                           prompt + GPT-4o call
                           JSON schema validation
                           safe fallback response
```

## Evals

The evaluation suite covers easy, medium, adversarial, multilingual, emergency, and malformed-input scenarios.

See [EVALS.md](/e:/P/MomBot/EVALS.md).

Run:

```bash
pytest evals/test_triage.py
```

## Tooling

Built with GPT-4o via OpenAI API, FastAPI, Streamlit, Pydantic v2. Claude used for architecture review and README drafting. pytest for evals.
