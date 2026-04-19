---
title: Cesar Assistant
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

Cesar Assistant

Gradio app with two modules:

- Discovery form to collect Cesar Mora's goals, routines, integrations, financial questions, and constraints.
- Analytical and conversational chatbot powered by Gemini, pandas, and matplotlib for reasoning, personality-aware NLP, tables, and charts.

## Requirements

- Python 3.9+
- A `.env` file with `GEMINI_API_KEY`
- Optional `GEMINI_MODEL` to force a preferred Gemini model
- Supabase configured with the tables defined in `supabase_schema.sql`

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The app launches with `share=True`, so Gradio prints a temporary public URL in the terminal.

## Flask example

If you want a minimal Supabase + Flask example alongside the Gradio app, run:

```bash
python supabase_flask_app.py
```

This example reads `SUPABASE_URL` and `SUPABASE_KEY` from `.env` and renders rows from a `todos` table.

## What the app does

- Builds a master prompt from discovery answers including financial profile fields.
- Uses `google.genai` with fallback across compatible Gemini models.
- Adds a conversational NLP mode that estimates communication style and adapts the answer tone.
- Stores profile memory and conversation history in Supabase when the database is configured.
- Applies business-response rules from detected personality categories such as urgency, risk posture, business focus, and detail preference.
- Lets you select the Gemini model from the UI or through `GEMINI_MODEL` in the environment.
- Renders returned table data with pandas and charts with matplotlib.

## Supabase memory

1. Apply the SQL in `supabase_schema.sql` to your Supabase project.
2. Set `SUPABASE_URL` and `SUPABASE_KEY` if you want the Flask example and the main chatbot to share the same access path. `SUPABASE_SERVICE_KEY` or `SUPABASE_ANON_KEY` also work as fallbacks for the chatbot.
3. Use the same `Profile ID` in Discovery and Analitica to persist and reload the same user profile.

The app stores:

- `assistant_profiles`: profile snapshot, master prompt, tone, and personality baseline.
- `assistant_conversations`: user prompts, assistant replies, detected personality summary, and row count for structured outputs.

If the tables or permissions are missing, the app continues working without persistence and shows the memory error in the UI.

## Example prompts for the analytical tab

- `Explain the pros and cons of RAG architectures step-by-step.`
- `Generate a table of the top 5 programming languages by popularity in 2024 and plot them as a bar chart.`
- `Create a monthly cash-flow summary table for a small retail business and chart net cash flow.`
