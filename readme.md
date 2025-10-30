# ResuMate

AI-powered resume evaluator and visualizer built with Streamlit.

ResuMate helps you compare resumes against a job description using semantic similarity, ATS-style scoring, grammar checks and tailored recommendations. It generates a leaderboard, charts and a downloadable PDF report to help users understand which resumes are the best match.

---

## Key features

- Upload a job description (PDF) and multiple resumes (PDF).
- Semantic similarity ranking (resume vs JD).
- ATS-like keyword matching and scoring.
- Grammar checking via LanguageTool.
- Action-verb counts, section detection, and keyword/missed keyword reporting.
- Resume leaderboard and visual analysis (plots exported as `plot.png`).
- Export a consolidated PDF report with results.

## Tech stack

- Python 3.8+ (recommended)
- Streamlit (UI)
- SpaCy (NLP)
- sentence-transformers (embeddings / similarity)
- language-tool-python (grammar)
- reportlab, PyPDF2 (PDF generation)
- SQLite (simple local storage - `resumate.db`)

See `requirements.txt` for the full dependency list.

## Quickstart (Windows / PowerShell)

1. Clone the repo and change directory:

```powershell
cd C:\Users\<you>\Documents\Resumate\v2
```

2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The `requirements.txt` file lists packages used by the project. You may need to correct or install a couple of packages manually if the file contains typos (for example `skilearn` → `scikit-learn`).
- Install spaCy model if not already present:

```powershell
python -m spacy download en_core_web_sm
```

4. (Optional) Create a `.env` file to configure API keys and other secrets. The app will read `GROQ_API_KEY` from environment variables when present. Example `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the Streamlit app:

```powershell
streamlit run app.py
```

If you prefer, you can run `python app.py`—the Streamlit invocation is the usual approach for the interactive UI.

## Configuration

- `config.py` centralises repo paths and loader logic. It reads `.env` via python-dotenv.
- Database: a lightweight SQLite DB (`resumate.db`) stores evaluation history.
- LanguageTool cache: `config.get_languagetool_cache_dir()` will prefer a repo-local `LanguageTool/` folder or fall back to a user-local cache.

## Project layout

Top-level files and important directories:

- `app.py` - app entrypoint that imports and runs the UI.
- `config.py` - configuration helpers and environment variable loading.
- `requirements.txt` - Python dependencies used by the project.
- `resumate.db` - local SQLite DB (auto-created/used by the app).
- `ui/` - Streamlit UI components. `ui/main.py` contains the primary UI and leaderboard rendering.
- `services/` - modular services (PDF, NLP, LLM, grammar, DB, plotting).
- `core/` - core evaluation and analysis functions.
- `LanguageTool/` - (optional) local LanguageTool files and cache.

## Usage notes

- Upload a job description (PDF) and one or more resumes (PDF) on the Home page, then click `🔍 Evaluate Resumes` to run the full analysis.
- Evaluation results are displayed with detailed per-resume breakdowns. A leaderboard shows ranked similarity percentages.
- Use the Download button to export a consolidated PDF report.

## Troubleshooting

- If Streamlit reports missing packages, double-check `requirements.txt` and install packages manually.
- If spaCy raises a model error, run: `python -m spacy download en_core_web_sm`.
- If LanguageTool fails to start or download models, check `LANGUAGE_TOOL_CACHE_DIR` environment variable or permissions for `LanguageTool/` in the repo.
- If resume names or text are hard to read in the UI, check `ui/main.py` or the injected CSS under `resumate/assets/style.css` (the app injects custom CSS via `_inject_css()` in `ui/main.py`).

## Development notes

- The project is structured to keep services small and testable. If adding features, add unit tests alongside the changed modules.
- To persist DB changes across runs, commit or back up `resumate.db` carefully. The DB contains evaluation history.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repo and create a feature branch.
2. Add/modify code and tests.
3. Run the app locally and verify changes.
4. Open a pull request with a clear summary of changes.

Please follow Python packaging and style conventions (PEP8). If adding third-party dependencies, prefer well-maintained packages and update `requirements.txt`.

## License & credits

Add your preferred license here (e.g., MIT) or keep repo license consistent with your org's policy.

---

If you'd like, I can:

- Move the inline leaderboard CSS into `resumate/assets/style.css` and standardize UI theming.
- Clean up or pin the `requirements.txt` entries (fix typos like `skilearn`) and produce a `requirements-dev.txt` for development tools.

Tell me which follow-up you'd like and I'll implement it.
