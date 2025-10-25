# Standard library imports
import datetime
import hashlib
import io
import os
from pathlib import Path
import re
import sqlite3
import time
from math import pi

# Third-party imports
import language_tool_python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import spacy
import streamlit as st
import textstat
from dotenv import load_dotenv
from groq import Groq
from langchain.memory import ConversationBufferMemory
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config("ResuMate", "📄", layout="wide")


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


nlp = load_spacy_model()

@st.cache_resource
def load_grammar_tool():
    # Determine LanguageTool cache directory.
    # Priority: existing LANGUAGE_TOOL_CACHE_DIR env var -> project-local LanguageTool folder -> user home folder
    env_dir = os.getenv("LANGUAGE_TOOL_CACHE_DIR")
    if env_dir:
        lt_cache_path = Path(env_dir)
    else:
        # place the cache inside the repository (one level up from this file)
        repo_root = Path(__file__).resolve().parents[1]
        lt_cache_path = repo_root / "LanguageTool"

    try:
        lt_cache_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to user home directory if project path is not writable
        lt_cache_path = Path.home() / ".resumate" / "languagetool"
        lt_cache_path.mkdir(parents=True, exist_ok=True)

    lt_cache_dir = str(lt_cache_path)

    # Set env variable for cache (LanguageTool will download models here)
    os.environ["LANGUAGE_TOOL_CACHE_DIR"] = lt_cache_dir

    # Initialize tool normally
    try:
        tool = language_tool_python.LanguageTool("en-US")
        # mark available
        tool._available = True
        return tool
    except SystemError as se:
        # Java version incompatible (LanguageTool requires Java >= 17)
        msg = f"LanguageTool initialization failed: {se}. Grammar checks will be disabled."
        print(msg)

        class DummyGrammarTool:
            def __init__(self, message):
                self._message = message
                self._available = False

            def check(self, text):
                # Return empty list of matches so rest of app works
                return []

        return DummyGrammarTool(msg)
    except Exception as e:
        # Any other exception: fallback to dummy tool but log the error
        msg = f"LanguageTool initialization error: {e}. Grammar checks will be disabled."
        print(msg)

        class DummyGrammarTool2:
            def __init__(self, message):
                self._message = message
                self._available = False

            def check(self, text):
                return []

        return DummyGrammarTool2(msg)

grammar_tool = load_grammar_tool()




def extract_skills_ner(text):
    """
    Extract potential skills, technologies, degrees, orgs, job titles
    using NER from spaCy.
    """
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in [
            "ORG",
            "PRODUCT",
            "WORK_OF_ART",
            "EDUCATION",
            "GPE",
            "NORP",
            "PERSON",
            "LANGUAGE",
        ]:
            skills.add(ent.text.strip())
    return list(skills)


@st.cache_resource
def load_sentence_model():
    # using a compact, good-quality model
    return SentenceTransformer("all-MiniLM-L6-v2")


sbert_model = load_sentence_model()


# --- Database Setup ---
conn = sqlite3.connect("resumate.db", check_same_thread=False)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password_hash TEXT)"""
)
c.execute(
    """CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, username TEXT, timestamp TEXT, resume_name TEXT, similarity REAL, ats_score REAL)"""
)
conn.commit()


def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register(u, pw):
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (u, hash_pw(pw)))
        conn.commit()
        return True
    except:
        return False


def authenticate(u, pw):
    c.execute("SELECT password_hash FROM users WHERE username=?", (u,))
    row = c.fetchone()
    return row and row[0] == hash_pw(pw)


# --- Evaluation Logic (Your Original Code) ---
def extract_text(f):
    r = PyPDF2.PdfReader(f)
    return (
        "".join(p.extract_text() or "" for p in r.pages).strip() or "No readable text."
    )


def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()


def compute_similarity(a, b):
    """
    Returns semantic similarity between texts a and b as percentage (0-100).
    Uses sentence-transformers embeddings and cosine similarity.
    """
    # encode -> returns numpy arrays (or tensors). Convert to numpy to be safe.
    emb = sbert_model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    a_emb, b_emb = emb[0], emb[1]

    # cosine similarity
    cos = np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-10)
    score = float(cos * 100.0)

    # Clamp between 0 and 100
    return max(0.0, min(100.0, score))


def compute_ats(resume_text, jd_text, jd_keywords=None):
    if jd_keywords is None:
        jd_keywords = extract_keywords(jd_text, top_n=20)

    # Semantic similarity
    sim_score = compute_similarity(jd_text, resume_text)

    # Keyword match
    res_keywords = extract_keywords(resume_text, top_n=20)
    matched_kw = len(set(jd_keywords) & set(res_keywords))

    # NER-based skills
    resume_skills = extract_skills_ner(resume_text)
    jd_skills = extract_skills_ner(jd_text)
    matched_skills = set(jd_skills) & set(resume_skills)
    missing_skills = set(jd_skills) - set(resume_skills)
    skill_match_score = (len(matched_skills) / max(1, len(jd_skills))) * 100

    # Section completeness
    sections = sections_found(resume_text)
    section_score = (sum(sections.values()) / len(sections)) * 100

    # Grammar score
    grammar_count, _ = check_grammar(resume_text)
    max_errors = max(1, word_count(resume_text) // 50)
    grammar_score = max(0, 100 - min(grammar_count / max_errors * 100, 100))

    # Weighted ATS
    ats = (
        0.4 * sim_score
        + 0.2 * skill_match_score
        + 0.2 * section_score
        + 0.2 * grammar_score
    )

    return round(ats, 2), list(matched_skills), list(missing_skills)


def check_grammar(t):
    matches = grammar_tool.check(t)
    return len(matches), [{"Error": m.message, "Sentence": m.context} for m in matches]



def missing_kw(jd, res):
    return list(set(jd) - set(res))


def sections_found(t):
    return {
        s: bool(re.search(s, t, re.I))
        for s in ["Education", "Experience", "Skills", "Projects", "Certifications"]
    }


def count_action_verbs(text):
    """
    Detect all verbs dynamically using spaCy POS tagging.
    Returns:
        - total verb count
        - list of verbs found
    """
    doc = nlp(text)
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    return len(verbs), verbs


def generate_recommendation(d):
    """
    Generate tailored recommendations using Groq LLM.
    Combines missing keywords, skills, and grammar insights
    to produce a short, actionable summary.
    """

    prompt = f"""
    You are an expert resume reviewer for ATS systems.

    Analyze the following resume evaluation details and give a short, actionable paragraph
    (3-5 sentences) suggesting how to improve the resume to better match the job description.

    Resume Details:
    - Similarity Score: {d.get('similarity', 0)}%
    - ATS Score: {d.get('ats_score', 0)}
    - Missing Keywords: {', '.join(d.get('missing_keywords', []))}
    - Missing Skills: {', '.join(d.get('missing_skills', []))}
    - Grammar Errors: {d.get('grammar_errors', 0)}
    - Action Verbs Used: {d.get('action_verbs_count', 0)}

    Give clear and specific feedback, e.g.:
    “Add more mentions of cloud frameworks and deployment tools.”
    Avoid bullet points, give one cohesive paragraph.
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=180,
        )
        recommendation = response.choices[0].message.content.strip()
    except Exception as e:
        recommendation = f"(Groq API Error) Could not generate feedback: {e}"

    return recommendation


def word_count(t):
    return len(t.split())


def save_plot(scores, ats):
    labels = list(scores.keys())
    n = len(labels)

    # if no data, show a friendly message and return
    if n == 0:
        st.info("No resumes to plot.")
        return None

    sim_values = [float(x) for x in scores.values()]
    ats_values = [float(x) for x in ats.values()]

    # If fewer than 3 resumes, radar is awkward — use grouped bar chart
    if n < 3:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(n)
        width = 0.35
        ax.bar(x - width / 2, sim_values, width, label="Similarity (%)")
        ax.bar(x + width / 2, ats_values, width, label="ATS Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.savefig("plot.png")
        return "plot.png"

    # Radar chart path for n >= 3
    sim_vals = sim_values + sim_values[:1]
    ats_vals = ats_values + ats_values[:1]
    angles = [i * 2 * pi / n for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, sim_vals, "o-", linewidth=2, label="Similarity (%)")
    ax.fill(angles, sim_vals, alpha=0.25)
    ax.plot(angles, ats_vals, "o-", linewidth=2, label="ATS Score")
    ax.fill(angles, ats_vals, alpha=0.25)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], labels)
    ax.set_ylim(0, 100)
    ax.set_title("Resume Evaluation Radar Chart", y=1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    st.pyplot(fig)
    plt.savefig("plot.png")
    return "plot.png"


def generate_pdf_report(details, plot_path=None):
    """
    Generate a clean, organized PDF report using ReportLab Platypus.

    Includes per-resume metrics, matched/missing skills, missing keywords,
    grammar issues table, sections found, action verbs preview and an overview chart.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    h2 = styles['Heading2']
    normal = styles['Normal']
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=10)

    story = []
    story.append(Paragraph('ResuMate - Resume Evaluation Report', title_style))
    story.append(Spacer(1, 12))

    for name, d in details.items():
        story.append(Paragraph(f'📄 Resume: {name}', h2))
        story.append(Spacer(1, 6))

        # Metrics table (two columns)
        metrics = [
            ['Similarity', f"{d.get('similarity', 0):.2f}%"],
            ['ATS Score', f"{d.get('ats_score', 0):.2f}"],
            ['Grammar Errors', str(d.get('grammar_errors', 0))],
            ['Action Verbs', str(d.get('action_verbs_count', 0))],
            ['Word Count', str(d.get('word_count', 0))],
        ]
        mtable = Table(metrics, colWidths=[2.2*inch, 3.8*inch])
        mtable.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#1a202c')),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
        ]))
        story.append(mtable)
        story.append(Spacer(1, 8))

        # Matched Skills
        story.append(Paragraph('✅ Matched Skills', styles['Heading3']))
        if d.get('matched_skills'):
            skills_tbl = [[s] for s in d['matched_skills']]
            stbl = Table([['Skill']] + skills_tbl, colWidths=[6*inch])
            stbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(stbl)
        else:
            story.append(Paragraph('No matched skills found.', small))
        story.append(Spacer(1, 6))

        # Missing Skills
        story.append(Paragraph('❌ Missing Skills', styles['Heading3']))
        if d.get('missing_skills'):
            mskills_tbl = [[s] for s in d['missing_skills']]
            mst = Table([['Skill']] + mskills_tbl, colWidths=[6*inch])
            mst.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#c53030')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(mst)
        else:
            story.append(Paragraph('No missing skills.', small))
        story.append(Spacer(1, 6))

        # Missing Keywords
        story.append(Paragraph('🔍 Missing Keywords', styles['Heading3']))
        if d.get('missing_keywords'):
            kw_tbl = [[k] for k in d['missing_keywords']]
            kt = Table([['Keyword']] + kw_tbl, colWidths=[6*inch])
            kt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(kt)
        else:
            story.append(Paragraph('No missing keywords.', small))
        story.append(Spacer(1, 6))

        # Grammar Issues (table)
        story.append(Paragraph('🔠 Grammar Issues', styles['Heading3']))
        if d.get('grammar_details'):
            grows = [[gd.get('Error', '')[:80], gd.get('Sentence', '')[:140]] for gd in d['grammar_details']]
            gtbl = Table([['Error', 'Context']] + grows, colWidths=[2.5*inch, 3.5*inch])
            gtbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(gtbl)
        else:
            story.append(Paragraph('No grammar issues detected.', small))
        story.append(Spacer(1, 6))

        # Sections Found
        story.append(Paragraph('📑 Sections Found', styles['Heading3']))
        sec_rows = [[k, 'Found' if v else 'Missing'] for k, v in d.get('sections_found', {}).items()]
        sec_tbl = Table([['Section', 'Status']] + sec_rows, colWidths=[3*inch, 3*inch])
        sec_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(sec_tbl)
        story.append(Spacer(1, 6))

        # Action verbs preview
        av = ', '.join(d.get('action_verbs_list', [])[:30])
        if av:
            story.append(Paragraph('📝 Action Verbs Preview', styles['Heading3']))
            story.append(Paragraph(av, small))
            story.append(Spacer(1, 12))

        # Add a page break between resumes
        story.append(PageBreak())

    # Add plot at end if available
    if plot_path:
        try:
            story.append(Paragraph('📊 Overview Chart', h2))
            story.append(Spacer(1, 8))
            img = RLImage(plot_path, width=6*inch, height=3*inch)
            story.append(img)
        except Exception:
            story.append(Paragraph('⚠ Failed to embed chart.', small))

    doc.build(story)
    buffer.seek(0)
    return buffer


# --- UI Setup ---
st.markdown(
    """<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: #f8f9fa;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding: 2rem 3rem;
        background: #ffffff;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 2rem auto;
        max-width: 1400px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Sidebar Text Color */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Fix sidebar text input color */
    [data-testid="stSidebar"] input[type="text"] {
        color: #1a202c !important;
        background: #ffffff !important;
    }
    [data-testid="stSidebar"] input[type="text"]::placeholder {
        color: #a0aec0 !important;
        opacity: 1 !important;
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Title Styling */
    h1 {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    h2 {
        color: #2d3748 !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #3182ce !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: #3182ce;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(49, 130, 206, 0.3);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.5);
        background: #2c5282;
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Download Button Styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.6);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        color: #ffffff !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #3182ce;
        box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #1a202c;
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"] p {
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    [data-testid="stFileUploader"] .uploaded-file {
        color: #2d3748 !important;
        font-weight: 600;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3182ce;
        background: #edf2f7;
        box-shadow: 0 2px 8px rgba(49, 130, 206, 0.15);
    }
    
    [data-testid="stFileUploader"]:hover p,
    [data-testid="stFileUploader"]:hover .uploaded-file {
        color: #1a202c !important;
    }
    
    /* Metrics and Evaluation Results */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #3182ce !important;
        line-height: 1.2 !important;
        letter-spacing: -0.02em !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #718096 !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Make evaluation results more visible and structured */
    .element-container, .stMarkdown {
        color: #ffffff !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Metric Containers */
    [data-testid="column"] > div > div > div {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    [data-testid="column"] > div > div > div:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Tables in evaluation results */
    .dataframe tbody tr td {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    /* Section Headers */
    .stMarkdown h3 {
        font-size: 1.25rem !important;
        color: #ffffff !important;
        margin: 2rem 0 1rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #e2e8f0 !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        color: #2d3748 !important;
        padding: 1.5rem !important;
        background: white !important;
    }
    
    /* Skills and Keywords sections */
    .element-container table {
        margin: 1rem 0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07) !important;
    }
    
    /* Table Containers */
    div[data-testid="stTable"], 
    div[data-testid="stDataFrame"] {
        background: white !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
        margin: 1rem 0 !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Section Headers above Tables */
    .stMarkdown h3 {
        font-size: 1.25rem !important;
        color: #2d3748 !important;
        margin: 2rem 0 1rem 0 !important;
        padding: 0.75rem 1rem !important;
        border-left: 4px solid #3182ce !important;
        background: #ffffff !important;
        border-radius: 8px !important;
    }
    
    /* Status indicators */
    .stSuccess, .stInfo {
        margin: 1rem 0 !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Metric delta colors */
    [data-testid="stMetricDelta"] {
        color: #2d3748 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        padding: 1rem 1.25rem !important;
        transition: all 0.2s ease !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #edf2f7 !important;
        border-color: #3182ce !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Add icon to indicate expandable sections */
    .streamlit-expanderHeader::before {
        content: "▶" !important;
        margin-right: 8px !important;
        font-size: 0.8em !important;
        transition: transform 0.2s ease !important;
    }
    
    .streamlit-expanderHeader[data-expanded="true"]::before {
        transform: rotate(90deg) !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
        background: #ffffff !important;
        margin: 0.5rem 0 1rem 0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    /* Resume Results Grid Layout */
    .streamlit-expanderContent [data-testid="column"] {
        background: white !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }
    
    /* Skills and Keywords Lists */
    .streamlit-expanderContent ul {
        list-style: none !important;
        padding: 0 !important;
        margin: 1rem 0 !important;
    }
    
    .streamlit-expanderContent ul li {
        padding: 0.5rem 0 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        color: #4a5568 !important;
    }
    
    .streamlit-expanderContent ul li:last-child {
        border-bottom: none !important;
    }
    
    /* Tables */
    .dataframe {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07) !important;
        margin: 1rem 0 !important;
    }
    
    .dataframe thead tr th {
        background: #2d3748 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border: none !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* Index column styling */
    .dataframe thead tr th:first-child,
    .dataframe tbody tr td:first-child {
        color: #2d3748 !important;
        font-weight: 600 !important;
        background-color: #f8fafc !important;
        border-right: 2px solid #e2e8f0 !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8fafc !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #edf2f7 !important;
        transition: all 0.2s ease !important;
    }
    
    .dataframe tbody tr td {
        padding: 1rem !important;
        border: none !important;
        color: #1a202c !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    
    /* Special styling for skills and analysis tables */
    div[data-testid="stTable"] table,
    div[data-testid="stDataFrame"] table {
        width: 100% !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07) !important;
        margin: 1rem 0 !important;
    }

    /* Headers for analysis tables */
    div[data-testid="stTable"] thead tr th,
    div[data-testid="stDataFrame"] thead tr th {
        background: #2d3748 !important;
        color: white !important;
        padding: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-size: 0.9rem !important;
    }

    /* Cells for analysis tables */
    div[data-testid="stTable"] tbody tr td,
    div[data-testid="stDataFrame"] tbody tr td {
        padding: 1rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        color: #1a202c !important;
        font-weight: 500 !important;
        background: white !important;
    }

    /* Hover effect for table rows */
    div[data-testid="stTable"] tbody tr:hover,
    div[data-testid="stDataFrame"] tbody tr:hover {
        background-color: #f7fafc !important;
    }

    /* Last row styling */
    div[data-testid="stTable"] tbody tr:last-child td,
    div[data-testid="stDataFrame"] tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    /* Remove bottom border from last row */
    .dataframe tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    /* Ensure index numbers are visible */
    .row_heading {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Style for all text in tables */
    table tbody tr td {
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    /* Style for metric values in tables */
    table tbody tr td strong {
        color: #2d3748 !important;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: #3182ce;
        border-radius: 10px;
    }
    
    /* Success/Error/Info Messages */
    .stSuccess {
        background-color: #c6f6d5;
        color: #22543d;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #48bb78;
    }
    
    .stError {
        background-color: #fed7d7;
        color: #742a2a;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f56565;
    }
    
    .stInfo {
        background-color: #bee3f8;
        color: #2c5282;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4299e1;
    }
    
    /* Dataframe Container */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Radio Buttons */
    .stRadio > label {
        font-weight: 600;
        color: #ffffff !important;
    }
    
    /* Form Container */
    [data-testid="stForm"] {
        background: #1a202ceb;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > label {
        font-weight: 600;
        color: #ffffff !important;
    }
    
    /* Leaderboard Styling */
    .leaderboard-item {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3182ce;
        transition: all 0.3s ease;
    }
    
    .leaderboard-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(49, 130, 206, 0.15);
        background: #edf2f7;
    }
    
    /* Chat Interface */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main .block-container > div {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3182ce;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2c5282;
    }
</style>""",
    unsafe_allow_html=True,
)

# --- Session State ---
if "user" not in st.session_state:
    st.session_state.user = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = None
if "resume_details" not in st.session_state:
    st.session_state.resume_details = {}
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = None

# --- Authentication UI ---
if not st.session_state.user:
    # Welcome Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>📄 ResuMate</h1>
            <p style='font-size: 1.2rem; color: #718096;'>Your AI-Powered Resume Evaluation Assistant</p>
            <p style='font-size: 1rem; color: #a0aec0;'>Analyze resumes • Get ATS Scores • Improve Your Career</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Authentication Section
    auth_mode = st.sidebar.radio("🔐 Account Options", ["Login", "Register"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 1rem;'>
            <h4 style='margin-top: 0;'>✨ Features</h4>
            <ul style='font-size: 0.9rem; line-height: 1.8;'>
                <li>📊 AI-Powered ATS Scoring</li>
                <li>🎯 Keyword Matching</li>
                <li>🔍 Skills Gap Analysis</li>
                <li>✍️ Grammar Checking</li>
                <li>💬 AI Resume Assistant</li>
                <li>📈 Performance Tracking</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("auth"):
        st.markdown(f"### {auth_mode} to Continue")
        u = st.text_input("👤 Username", placeholder="Enter your username")
        pw = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        ok = st.form_submit_button(f"{'🚀 Login' if auth_mode == 'Login' else '✨ Create Account'}")
    
    if ok:
        if auth_mode == "Login" and authenticate(u, pw):
            st.session_state.user = u
            st.success(f"✅ Welcome back, {u}!")
            st.balloons()
            time.sleep(1)
            st.rerun()
        elif auth_mode == "Register" and register(u, pw):
            st.success("🎉 Registration successful! Please login.")
            st.balloons()
        else:
            st.error("❌ Authentication failed. Please try again.")
else:
    # User Welcome Section
    st.sidebar.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👤</div>
            <h3 style='margin: 0; color: #ffffff;'>Welcome {st.session_state.user} 👋</h3>
            <p style='font-size: 0.8rem; margin: 0.5rem 0 0 0; color: #a0aec0;'>Your personalized resume evaluation dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("📋 Navigation", ["🏠 Home", "📊 History", "🚪 Logout"])

    if page == "🚪 Logout":
        st.session_state.user = None
        st.session_state.jd_text = None
        st.session_state.resume_details = {}
        st.session_state.chat_memory = None
        st.success("👋 Logged out successfully!")
        time.sleep(1)
        st.rerun()
    elif page == "📊 History":
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1>📊 Evaluation History</h1>
                <p style='font-size: 1.1rem; color: #718096;'>Track your resume evaluation progress over time</p>
            </div>
        """, unsafe_allow_html=True)
        
        df = pd.read_sql(
            "SELECT timestamp, resume_name, similarity, ats_score FROM history WHERE username=? ORDER BY timestamp DESC",
            conn,
            params=(st.session_state.user,),
        )
        
        if not df.empty:
            # Summary Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("� Total Evaluations", len(df))
            with col2:
                st.metric("⭐ Avg Similarity", f"{df['similarity'].mean():.2f}%")
            with col3:
                st.metric("🎯 Avg ATS Score", f"{df['ats_score'].mean():.2f}")
            with col4:
                st.metric("🏆 Best Score", f"{df['ats_score'].max():.2f}")
            
            st.markdown("---")
            st.subheader("📋 Detailed History")
            
            # Format the dataframe
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            df.columns = ['📅 Date', '📄 Resume', '🔗 Similarity %', '🎯 ATS Score']
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No evaluation history yet. Start by uploading resumes on the Home page!")
    else:
        # Main Resume Evaluation
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1>📄 AI Resume Evaluator</h1>
                <p style='font-size: 1.1rem; color: #718096;'>Upload your job description and resumes to get instant AI-powered insights</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Upload Section with Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📑 Job Description")
            jd = st.file_uploader(
                "Upload JD (PDF)", 
                type=["pdf"],
                help="Upload the job description you want to match against",
                label_visibility="collapsed"
            )
            if jd:
                st.success(f"✅ Loaded: {jd.name}")
        
        with col2:
            st.markdown("### 📄 Resumes")
            rs = st.file_uploader(
                "Upload Resumes (PDF)", 
                type=["pdf"], 
                accept_multiple_files=True,
                help="Upload one or multiple resumes to evaluate",
                label_visibility="collapsed"
            )
            if rs:
                st.success(f"✅ Loaded {len(rs)} resume(s)")
        
        st.markdown("---")

        if st.button("🔍 Evaluate Resumes") and jd and rs:
            jd_txt = extract_text(jd)
            # Store JD text in session state for chatbot
            st.session_state.jd_text = jd_txt
            scores, ats, details = {}, {}, {}

            # Animated background shown during evaluation
            anim_placeholder = st.empty()
            anim_placeholder.markdown(
                """
                <style>
                /* Subtle animated gradient background */
                .resumate-anim-bg {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: -1; /* behind Streamlit content */
                    pointer-events: none;
                    background: linear-gradient(270deg, #0f172a, #1e293b, #0ea5e9, #6ee7b7);
                    background-size: 600% 600%;
                    opacity: 0.14;
                    animation: resumateGradient 18s ease infinite;
                }

                @keyframes resumateGradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                </style>
                <div class="resumate-anim-bg"></div>
                """,
                unsafe_allow_html=True,
            )

            progress = st.progress(0)

            for i, r in enumerate(rs):
                t = extract_text(r)

                sim = compute_similarity(jd_txt, t)
                ats_score, matched_skills, missing_skills = compute_ats(t, jd_txt)

                # Store for radar chart
                scores[r.name] = sim
                ats[r.name] = ats_score

                verbs_count, verbs_list = count_action_verbs(t)

                details[r.name] = {
                    "similarity": sim,
                    "ats_score": ats_score,
                    "grammar_errors": check_grammar(t)[0],
                    "grammar_details": check_grammar(t)[1],
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "missing_keywords": list(
                        set(extract_keywords(jd_txt)) - set(extract_keywords(t))
                    ),
                    "sections_found": sections_found(t),
                    "action_verbs_count": verbs_count,
                    "action_verbs_list": verbs_list,
                    "word_count": word_count(t),
                }

                details[r.name]["recommendation"] = generate_recommendation(
                    details[r.name]
                )

                c.execute(
                    "INSERT INTO history(username, timestamp, resume_name, similarity, ats_score) VALUES (?, ?, ?, ?, ?)",
                    (
                        st.session_state.user,
                        datetime.datetime.now().isoformat(),
                        r.name,
                        sim,
                        ats_score,
                    ),
                )
                time.sleep(0.5)

            conn.commit()
            # Remove animated background now that evaluation finished
            try:
                anim_placeholder.empty()
            except Exception:
                # If placeholder is not present for any reason, ignore silently
                pass

            progress.progress(1.0)
            # Store resume details in session state for chatbot
            st.session_state.resume_details = details
            st.success("✅ Evaluation Completed!")

            # Display detailed results
            for name, d in details.items():
                with st.expander(f"📄 Resume: {name}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Similarity", f"{d['similarity']:.2f}%")
                        st.metric("📖 ATS Score", f"{d['ats_score']:.2f}")
                        st.metric("🔠 Grammar Errors", f"{d['grammar_errors']}")
                    with col2:
                        st.metric("💼 Action Verbs", f"{d['action_verbs_count']}")
                        if d["action_verbs_list"]:
                            st.subheader("📝 Verbs Used in Resume")
                            st.write(", ".join(d["action_verbs_list"]))
                        st.metric("📜 Word Count", f"{d['word_count']}")

                    st.markdown("""
                        <style>
                        .stExpander {
                            background-color: #2d3748 !important;
                            border-radius: 8px !important;
                        }
                        .stExpander > div:first-child {
                            color: white !important;
                            font-weight: 600 !important;
                            text-shadow: none !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("✅ Matched Skills", expanded=False):
                        if d["matched_skills"]:
                            st.table(pd.DataFrame(d["matched_skills"], columns=["Skill"]))
                        else:
                            st.info("No matched skills found.")

                    with st.expander("❌ Missing Skills", expanded=False):
                        if d["missing_skills"]:
                            st.table(pd.DataFrame(d["missing_skills"], columns=["Skill"]))
                        else:
                            st.success("No missing skills!")

                    with st.expander("🔍 Missing Keywords", expanded=False):
                        if d["missing_keywords"]:
                            st.table(pd.DataFrame(d["missing_keywords"], columns=["Keyword"]))
                        else:
                            st.success("✅ No missing keywords!")

                    with st.expander("🔠 Grammar Issues", expanded=False):
                        if d["grammar_details"]:
                            st.table(pd.DataFrame(d["grammar_details"]))
                        else:
                            st.success("✅ No grammar issues found.")

                    with st.expander("📑 Sections Found", expanded=False):
                        section_df = pd.DataFrame(
                            [
                                {"Section": k, "Status": "✔ Found" if v else "❌ Missing"}
                                for k, v in d["sections_found"].items()
                            ]
                        )
                        st.table(section_df)

                    st.subheader("💡 Tailored Recommendation")
                    st.write(details[name]["recommendation"])

            # --- Leaderboard ---
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                    <h2>🏆 Resume Leaderboard</h2>
                    <p style='color: #718096;'>Ranked by similarity score</p>
                </div>
            """, unsafe_allow_html=True)
            
            ranked = sorted(
                details.items(), key=lambda x: x[1]["similarity"], reverse=True
            )
            
            for i, (name, d) in enumerate(ranked, start=1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                st.markdown(f"""
                    <div class='leaderboard-item'>
                        <span style='font-size: 1.5rem; margin-right: 1rem;'>{medal}</span>
                        <strong style='font-size: 1.1rem; color: #1a202c;'>{name}</strong>
                        <span style='float: right; color: #3182ce; font-weight: 600; font-size: 1.1rem;'>{d['similarity']:.2f}% match</span>
                    </div>
                """, unsafe_allow_html=True)

            # Overview Chart
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                    <h2>📊 Visual Analysis</h2>
                    <p style='color: #718096;'>Comprehensive comparison of all evaluated resumes</p>
                </div>
            """, unsafe_allow_html=True)
            save_plot(scores, ats)

            # PDF Report
            buffer = generate_pdf_report(details, plot_path="plot.png")

            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                    <h2>📄 Export Results</h2>
                    <p style='color: #718096;'>Download a comprehensive PDF report of your evaluation</p>
                </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    "📥 Download Complete PDF Report",
                    buffer,
                    "resumate_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    # --- Chatbot Section (only for authenticated users) ---
    st.sidebar.markdown("---")
    if groq_api_key:
        st.sidebar.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='margin: 0;'>💬 AI Assistant</h2>
                <p style='font-size: 0.9rem; color: #a0aec0; margin: 0.5rem 0 0 0;'>Get personalized resume advice</p>
            </div>
        """, unsafe_allow_html=True)

        # Build context from uploaded documents
        context_info = ""
        if st.session_state.jd_text:
            jd_summary = st.session_state.jd_text[:500] + "..." if len(st.session_state.jd_text) > 500 else st.session_state.jd_text
            context_info += f"\n\n**JOB DESCRIPTION SUMMARY:**\n{jd_summary}\n"
        
        if st.session_state.resume_details:
            context_info += "\n**RESUME EVALUATION RESULTS:**\n"
            for resume_name, details in st.session_state.resume_details.items():
                context_info += f"\n- Resume: {resume_name}\n"
                context_info += f"  - Similarity Score: {details.get('similarity', 0):.2f}%\n"
                context_info += f"  - ATS Score: {details.get('ats_score', 0):.2f}\n"
                context_info += f"  - Grammar Errors: {details.get('grammar_errors', 0)}\n"
                context_info += f"  - Action Verbs: {details.get('action_verbs_count', 0)}\n"
                context_info += f"  - Word Count: {details.get('word_count', 0)}\n"
                if details.get('missing_skills'):
                    context_info += f"  - Missing Skills: {', '.join(details['missing_skills'][:5])}\n"
                if details.get('missing_keywords'):
                    context_info += f"  - Missing Keywords: {', '.join(details['missing_keywords'][:5])}\n"

        # Simple chat history (avoid langchain_groq incompatibilities)
        # We'll keep a simple list of (role, text) in session_state and call the Groq client directly.
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        if st.session_state.chat_history:
            with st.sidebar.expander("💬 Chat History", expanded=False):
                for role, content in st.session_state.chat_history:
                    icon = "👤" if role == "user" else "🤖"
                    display_role = "You" if role == "user" else "Resumate"
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                            <strong>{icon} {display_role}:</strong><br/>
                            <span style='font-size: 0.9rem;'>{content}</span>
                        </div>
                    """, unsafe_allow_html=True)

        # Sidebar chat interface
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        user_query = st.sidebar.text_input("💭 Ask anything about your resumes...", key="chat_input", placeholder="e.g., Which resume is best?")

        if user_query:
            with st.sidebar:
                with st.spinner("🤔 Thinking..."):
                    # Build messages for Groq chat API
                    msgs = []
                    system_msg = "You are Resumate, an intelligent resume mentor bot. Use the following context to inform your responses:\n" + context_info
                    msgs.append({"role": "system", "content": system_msg})

                    for role, content in st.session_state.chat_history:
                        msgs.append({"role": role, "content": content})

                    msgs.append({"role": "user", "content": user_query})

                    try:
                        response = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            messages=msgs,
                            temperature=0.7,
                            max_tokens=300,
                        )
                        reply = response.choices[0].message.content.strip()
                    except Exception as e:
                        reply = f"(Groq API error) {e}"

                    # Append to history
                    st.session_state.chat_history.append(("user", user_query))
                    st.session_state.chat_history.append(("assistant", reply))

                st.markdown(f"""
                    <div style='background: rgba(49, 130, 206, 0.15); 
                                padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #3182ce;'>
                        <strong style='color: #ffffff;'>🤖 Resumate:</strong><br/>
                        <span style='font-size: 0.95rem; color: #ffffff;'>{reply}</span>
                    </div>
                """, unsafe_allow_html=True)
        
        # Show helpful prompts if no data yet
        if not st.session_state.jd_text or not st.session_state.resume_details:
            st.sidebar.markdown("""
                <div style='background: rgba(66, 153, 225, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #4299e1;'>
                    <strong>💡 Quick Tip:</strong><br/>
                    <span style='font-size: 0.9rem;'>Upload a JD and resumes first to get personalized AI advice!</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
                <div style='background: rgba(72, 187, 120, 0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <strong>✨ Try asking:</strong><br/>
                    <ul style='font-size: 0.85rem; margin: 0.5rem 0 0 0; padding-left: 1.5rem;'>
                        <li>Which resume is best?</li>
                        <li>What skills am I missing?</li>
                        <li>How to improve my ATS score?</li>
                        <li>Compare all resumes</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
            <div style='background: rgba(245, 101, 101, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #f56565;'>
                <strong>❌ Configuration Error</strong><br/>
                <span style='font-size: 0.9rem;'>GROQ_API_KEY not found in .env file</span>
            </div>
        """, unsafe_allow_html=True)
