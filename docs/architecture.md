# ResuMate — Architecture Diagram and Explanation

## High level

ResuMate is a Streamlit-based single-process web app that evaluates PDF resumes against a job description. Key runtime components:

- UI: Streamlit (`ui/main.py`) — collects JD and resume PDF uploads, shows results and charts, drives evaluation workflow.
- Core evaluation: `core/evaluation.py` — orchestrates text extraction, NLP feature extraction, similarity and ATS score computation.
- Services (in `services/`): small modules handling specialized tasks:
  - `pdf_service.py` — PDF text extraction and PDF report generation (ReportLab).
  - `nlp_service.py` — spaCy and sentence-transformers (SBERT) utilities.
  - `grammar_service.py` — LanguageTool wrapper (language-tool-python) with local cache directory.
  - `llm_service.py` — Groq AI (optional) for recommendations/chat (GROQ_API_KEY).
  - `plot_service.py` — Plotly rendering & static export (kaleido fallback) and helpers.
  - `db_service.py` — SQLite thin wrapper for user and history storage.

External dependencies and resources

- Groq API (optional) — for advanced AI feedback and chat (configured by `GROQ_API_KEY`).
- LanguageTool (Java-backed) — used by `language_tool_python`; the code supports a dummy fallback.
- spaCy model: `en_core_web_sm` and `sentence-transformers` SBERT model `all-MiniLM-L6-v2`.
- SQLite file in repo root (`resumate.db`) used for history.

## Diagram (Mermaid)

```mermaid
flowchart LR
  subgraph UI[Streamlit UI]
    A[User Uploads JD (PDF)]
    B[User Uploads Resumes (PDFs)]
    C[Evaluate Button]
    D[Leaderboard / Charts / PDF Download]
    E[Chat / Recommendation Panel]
  end

  subgraph App[Application Logic]
    F[core.evaluation]
    G[pdf_service]
    H[nlp_service]
    I[grammar_service]
    J[llm_service]
    K[plot_service]
    L[db_service(SQLite)]
  end

  subgraph Ext[External Services]
    M[Groq API]
    N[LanguageTool JVM]
  end

  A -->|uploaded file| G
  B -->|uploaded files| G
  G -->|text| F
  F -->|call| H
  F -->|call| I
  F -->|store metrics| L
  F -->|request recommendation| J
  H -->|embeddings & NER| F
  I -->|grammar issues| F
  J -->|AI recommendation| F
  F -->|metrics| K
  K -->|chart bytes| D
  G -->|report bytes| D
  E -->|chat messages| J
  J -->|calls| M
  I -->|loads/uses| N
  L -->|history queries| D

  style UI fill:#f3f4f6,stroke:#111827
  style App fill:#ffffff,stroke:#111827
  style Ext fill:#eef2ff

  click M "https://groq.ai" "Groq API"
  click N "https://languagetool.org" "LanguageTool"
```

## Component responsibilities & interactions

- Streamlit UI (`ui/main.py`)
  - Handles auth (simple SQLite users), file uploads, session state, and the evaluation workflow.
  - Uses `@st.cache_resource` to cache heavy objects (spaCy model, SBERT model, LanguageTool instance).
  - Drives the evaluation loop: for each resume file it calls `core.evaluation.extract_text`, then computes similarity and ATS metrics, calls `llm_service.generate_recommendation`, and persists results via `db_service.insert_history`.

- core.evaluation
  - Orchestrates resume text extraction, TF-IDF keyword extraction, SBERT similarity, NER-based skill extraction, grammar checks, and ATS score composition.
  - Returns detailed `details` for each resume (similarity, matched/missing skills/keywords, grammar data).

- pdf_service
  - Uses `PyPDF2` for reading PDF text and ReportLab for generating a final PDF report with optional embedded chart bytes from `plot_service`.

- nlp_service
  - Loads spaCy and SBERT (SentenceTransformer). Provides `extract_skills_ner`, `sbert_model.encode`, and `count_action_verbs`.

- grammar_service
  - Wraps `language_tool_python` with a cached executor and returns (count, detail) for grammar issues. On initialization it attempts to set `LANGUAGE_TOOL_CACHE_DIR` (from `config.get_languagetool_cache_dir`). If LanguageTool fails it returns a dummy object and the app continues with grammar disabled.

- llm_service
  - If `GROQ_API_KEY` is provided, contacts Groq chat/completions (synchronous) to produce recommendations and chat replies. If not configured, returns helpful placeholder text.

- plot_service
  - Renders interactive visualizations using Plotly in Streamlit and tries to export static PNG bytes via Kaleido for embedding into PDF; stores any export error details in Streamlit session state.

- db_service
  - Uses SQLite file (`resumate.db`) with `check_same_thread=False` to tolerate Streamlit's threaded environment. Stores `users` and `history` rows; returns history for the UI.

## Data flow summary (happy path)

1. User uploads JD PDF and one or more resume PDFs in the Streamlit UI.
2. UI passes files to `pdf_service.extract_text`, which returns plain text per PDF.
3. `core.evaluation` calls `nlp_service` to compute embeddings (SBERT) and NER; computes TF-IDF keywords.
4. `core.evaluation` calls `grammar_service.check_grammar` to collect grammar issues.
5. Metrics are combined into an ATS score and similarity percentage. Missing keywords/skills are computed.
6. For each resume, `llm_service.generate_recommendation` (Groq) may be called to generate a short paragraph of feedback.
7. Results are inserted into SQLite history by `db_service` and returned to UI.
8. UI asks `plot_service` to generate chart bytes (or file), then calls `pdf_service.generate_pdf_report` to produce a downloadable PDF report.
9. User views leaderboard, charts, expands details, or downloads the PDF. The chat panel can send messages to `llm_service.chat_reply` with current context.

## Deployment & scaling notes

- Current setup is ideal for local/small single-user use (Streamlit single-process). For multi-user or production:
  - Move to a server-friendly DB (Postgres) and a dedicated API server to avoid concurrent SQLite issues.
  - Split heavy tasks (embedding, report generation, LLM calls) into background workers (Celery/RQ) to keep UI responsive; store job results in DB and surface progress via Polling/WebSocket.
  - Containerize (Docker) and deploy behind a web server. Expose only the Streamlit app or wrap with a lightweight FastAPI/Flask API if needed.
  - Protect LLM API keys (do not embed in client-side code); centralize LLM usage behind a server or rate-limiter.

## Observations & quick wins

- `db_service` already sets `check_same_thread=False`, which helps but is not a full concurrency solution — consider Postgres when multiple users are expected.
- `language_tool_python` may require Java; the code already includes a fallback dummy. Document Java pre-reqs in README.
- Plot export relies on `kaleido`; ensure it is installed in `requirements.txt` (already present).
- Consider extracting heavy dependencies and models into a separate `services/ai_worker.py` to allow horizontal scaling.

## Next steps (suggested)

- Add `docs/README-deploy.md` with Dockerfile + `docker-compose.yml` to demonstrate a recommended deployment (Streamlit + Postgres + a worker).
- Add integration tests for the core flow using small PDF fixtures (happy path + missing-text PDF).
- Add health checks for external dependencies (Groq reachable, LanguageTool status) and surfacing them in the UI.

---

*File generated programmatically: `docs/architecture.md`*
