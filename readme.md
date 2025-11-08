# ResuMate — NLP Resume Assistant

ResuMate is an NLP project focused on helping users analyze, improve and visualize resumes using natural language processing and LLMs. This repository contains the code, services, and UI for the project.

## Table of contents

- Project overview
- Key features
- Repository structure
- Quick start (development)
- Usage
- Contributing
- License & Copyright

## Project overview

ResuMate is a personal NLP project by Sam T James that combines classical NLP techniques and large language models to analyze resumes, check grammar, extract structured data, produce visualizations, and generate suggestions for improvement.

This repository contains services for parsing PDFs, grammar checks, LLM interactions, plotting utilities, and a small UI for interacting with the tools.

## Key features

- Resume parsing from PDF
- Grammar checking and suggestions
- LLM-backed recommendations and rewrites
- Basic visualization of resume metrics
- Simple local UI for experimenting with the pipeline

## Repository structure

Top-level files
- `app.py` — application entry (examples / runner)
- `config.py` — configuration and settings
- `requirements.txt` — Python dependencies
- `readme.md` — this file

Packages and folders
- `core/` — core processing logic (e.g., evaluation)
- `services/` — service implementations: `db_service.py`, `grammar_service.py`, `llm_service.py`, `nlp_service.py`, `pdf_service.py`, `plot_service.py`
- `ui/` — UI code (Streamlit app in `main.py`)
- `LanguageTool/` — language tool related resources
- `docs/` — architecture and documentation

## Quick start (development)

Prerequisites

- Python 3.10+ (or compatible 3.x)
- Git

Create a virtual environment and install dependencies (PowerShell):

```pwsh
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer to run the UI (Streamlit):

```pwsh
# from repository root
streamlit run .\ui\main.py
```

Or run the main app (if provided):

```pwsh
python app.py
```

## Usage

1. Start the UI with Streamlit (recommended for experiments).
2. Upload a resume (PDF) or paste text.
3. Use grammar checks, run LLM suggestions, and view visualizations.

For programmatic usage, import the services from `services` and call them directly. Each service exposes a small API; see inline docstrings and `services/` modules for details.

## Contributing

This project is primarily a personal project by Sam T James. Contributions are welcome — open an issue or pull request and I will review them. If you plan to contribute large features, please open an issue first to discuss the design.

When contributing, please:

- Keep changes focused and small
- Add tests for new functionality where applicable
- Follow the existing code style

## License & Copyright

Copyright (c) 2025 Sam T James

This project is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

If you prefer a different license or want the repository to be private / restricted, tell me and I will update this README and the license text accordingly.

Contact: Sam T James

