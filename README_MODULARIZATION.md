# ResuMate v2 Modularization

This refactor extracts the monolithic `app_v2.py` into a production-grade, modular package under `resumate/v2`. The top-level `app_v2.py` is now a thin bootstrap that invokes the modular app.

## New structure

- `resumate/v2/config.py` — configuration, env, paths, Groq client
- `resumate/v2/core/evaluation.py` — text extraction, keywords, similarity, ATS scoring, sections, word count
- `resumate/v2/services/nlp_service.py` — spaCy and SentenceTransformer loaders, NER skills, action verbs
- `resumate/v2/services/grammar_service.py` — LanguageTool loader and safe `check_grammar`
- `resumate/v2/services/llm_service.py` — Groq-based recommendation and chat
- `resumate/v2/services/db_service.py` — SQLite init, register, authenticate, history insert/query
- `resumate/v2/services/plot_service.py` — chart generation and saving
- `resumate/v2/services/pdf_service.py` — ReportLab PDF report builder
- `resumate/v2/ui/main.py` — Streamlit UI orchestrating all features
- `resumate/assets/style.css` — central stylesheet for Streamlit UI

## How to run

From the project root:

```pwsh
# (optional) activate your venv
# & .venv\Scripts\Activate.ps1

# run the v2 app
streamlit run app_v2.py
```

## Environment

- Set `GROQ_API_KEY` in a `.env` file to enable AI features.
- LanguageTool cache defaults to `LanguageTool/` at repo root; falls back to `%USERPROFILE%\.resumate\languagetool` if not writable.

## Notes

- Heavy NLP models (spaCy en_core_web_sm, sentence-transformers) are cached by Streamlit. First run may take a while.
- Database: uses `resumate.db` at repo root. Existing history will be preserved.

## Next steps (optional)

- Move CSS injection to read directly from `resumate/assets/style.css` in the UI (currently loaded; ensure it's applied early in the page lifecycle).
- Add logging and error reporting.
- Add unit tests for core scoring and services.
