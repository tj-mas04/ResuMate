# ResuMate: AI-Powered Resume Evaluation Assistant

ResuMate is an AI-powered application for evaluating resumes against job descriptions using modern NLP and machine learning techniques. Built with Streamlit, it provides instant ATS scoring, keyword and skills gap analysis, grammar checking, and personalized recommendations.

***

### Features

- **AI-Powered ATS Scoring:** Quantifies how well resumes match job descriptions using semantic similarity and keyword analysis.
- **Keyword and Skills Gap Analysis:** Identifies missing keywords and skills using NER and TF-IDF techniques.
- **Grammar Checking:** Integrates grammar checking and error reporting using LanguageTool.
- **Personalized Recommendations:** Leverages LLMs to generate tailored resume improvement suggestions.
- **PDF Report Export:** Generates comprehensive PDF reports for detailed analysis and comparison.
- **Leaderboard & History:** Tracks performance over time, visualizes metrics, and displays a ranked history of evaluations.

***

### Installation

1. **Clone the repository:**
   ```
   git clone <repo-url>
   cd <project-folder>
   ```

2. **Set up a virtual environment (recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not present, see "Dependencies" below for manual installation.)*

4. **Set environment variables:**
   - `GROQAPIKEY` (for LLM-powered recommendations)
   - `LANGUAGETOOLCACHEDIR` (optional, for LanguageTool cache directory)

   These can be set in a `.env` file in the project root.

***

### Usage

Run the main Streamlit app:
```sh
streamlit run app_v2.py
```
- Upload a Job Description PDF and one or multiple resume PDFs
- Review instant AI-powered scores, skill/keyword gaps, grammar issues, and tailored advice in the dashboard
- Download the full evaluation report as PDF

***

### Dependencies

Major packages:
- `streamlit`
- `spacy` (and English model)
- `languagetool-python`
- `PyPDF2`
- `sentence-transformers`
- `matplotlib`, `pandas`, `numpy`
- `scikit-learn`
- `reportlab`
- `dotenv`
- LLM integration (Groq API)

Install via pip, for example:
```sh
pip install streamlit spacy languagetool-python PyPDF2 sentence-transformers matplotlib pandas numpy scikit-learn reportlab python-dotenv groq
python -m spacy download en_core_web_sm
```

***

### File Structure

| File              | Description                        |
|-------------------|------------------------------------|
| app_v2.py         | Main Streamlit app logic           |
| resumate.db       | SQLite database for user/history    |
| .env              | API keys and configuration         |
| requirements.txt  | Python dependencies                |

***

### Notes

- Requires Python 3.8+
- Make sure `groq` API key is valid for LLM features.
- LanguageTool integration requires Java 17+ installed system-wide.

***

### License

This project is for academic/demo purposes. Adapt licensing as appropriate.

***

### Credits

Developed as a modern, minimalistic, extensible tool for resume evaluation and improvement.

***

Feel free to adapt and extend this README as your project evolves to include further modules and usage scenarios.


## Copyright

Copyright (c) 2025 Sam T James.  
All rights reserved.  
Unauthorized copying, distribution, or use of this project or any part of it is strictly prohibited.
