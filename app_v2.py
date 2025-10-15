# resumate_app.py
# Standard library imports
import datetime
import hashlib
import io
import os
import re
import sqlite3
import time
from math import pi

# Third-party imports
# import language_tool_python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import spacy
import streamlit as st
import textstat
from dotenv import load_dotenv
from groq import Groq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
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
    import language_tool_python
    import os

    # Permanent folder for LanguageTool backend
    lt_cache_dir = r"C:\Users\ASUS\Documents\Resumate\Dev\LanguageTool"
    os.makedirs(lt_cache_dir, exist_ok=True)

    # Initialize LanguageTool using the permanent folder
    return language_tool_python.LanguageTool(
        'en-US',
        config={"data_dir": lt_cache_dir}
    )

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


# --- UI Setup ---
st.markdown(
    """<style>
.sidebar .sidebar-content{background:#f0f2f6;width:200px;}
.stButton>button{background:#4CAF50;color:#fff;}
</style>""",
    unsafe_allow_html=True,
)

# --- Session State ---
if "user" not in st.session_state:
    st.session_state.user = None

# --- Authentication UI ---
if not st.session_state.user:
    st.title("📄 ResuMate")
    auth_mode = st.sidebar.radio("Account", ["Login", "Register"])
    with st.form("auth"):
        u = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button(auth_mode)
    if ok:
        if auth_mode == "Login" and authenticate(u, pw):
            st.session_state.user = u
            st.rerun()
        elif auth_mode == "Register" and register(u, pw):
            st.success("Registered!")
        else:
            st.error("Failed")
else:
    st.sidebar.header(f"Welcome {st.session_state.user}")
    page = st.sidebar.selectbox("Menu", ["Home", "History", "Logout"])

    if page == "Logout":
        st.session_state.user = None
        st.rerun()
    elif page == "History":
        df = pd.read_sql(
            "SELECT timestamp, resume_name, similarity, ats_score FROM history WHERE username=? ORDER BY timestamp DESC",
            conn,
            params=(st.session_state.user,),
        )
        st.title("📜 Your History")
        st.dataframe(
            df if not df.empty else pd.DataFrame({"Info": ["No history available."]})
        )
    else:
        # Main Resume Evaluation
        st.title("📄 ResuMate - AI Resume Evaluator")
        jd = st.file_uploader("📑 Upload Job Description (PDF)", type=["pdf"])
        rs = st.file_uploader(
            "📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True
        )

        if st.button("🔍 Evaluate Resumes") and jd and rs:
            jd_txt = extract_text(jd)
            scores, ats, details = {}, {}, {}
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
            progress.progress(1.0)
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

                    if d["matched_skills"]:
                        st.subheader("✅ Matched Skills")
                        st.table(pd.DataFrame(d["matched_skills"], columns=["Skill"]))

                    if d["missing_skills"]:
                        st.subheader("❌ Missing Skills")
                        st.table(pd.DataFrame(d["missing_skills"], columns=["Skill"]))

                    if d["missing_keywords"]:
                        st.subheader("🔍 Missing Keywords")
                        st.table(
                            pd.DataFrame(d["missing_keywords"], columns=["Keyword"])
                        )
                    else:
                        st.success("✅ No missing keywords!")

                    if d["grammar_details"]:
                        st.subheader("🔠 Grammar Issues")
                        st.table(pd.DataFrame(d["grammar_details"]))
                    else:
                        st.success("✅ No grammar issues found.")

                    st.subheader("📑 Sections Found")
                    section_df = pd.DataFrame(
                        [
                            {"Section": k, "Status": "✔ Found" if v else "❌ Missing"}
                            for k, v in d["sections_found"].items()
                        ]
                    )

                    st.subheader("💡 Tailored Recommendation")
                    st.write(details[name]["recommendation"])
                    st.table(section_df)

            # --- Leaderboard ---
            st.subheader("🏆 Resume Leaderboard")
            ranked = sorted(
                details.items(), key=lambda x: x[1]["similarity"], reverse=True
            )
            for i, (name, d) in enumerate(ranked, start=1):
                st.markdown(f"**{i}. {name} – {d['similarity']:.2f}% match**")

            # Overview Chart
            st.subheader("📊 Resume Evaluation Overview")
            save_plot(scores, ats)

            # PDF Report
            buffer = io.BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(200, height - 50, "ResuMate - Resume Evaluation Report")
            y = height - 80
            pdf.setFont("Helvetica", 12)

            for name, d in details.items():
                pdf.drawString(50, y, f"📄 Resume: {name}")
                y -= 20
                pdf.drawString(70, y, f"📊 Similarity: {d['similarity']:.2f}%")
                y -= 15
                pdf.drawString(70, y, f"📖 ATS Score: {d['ats_score']:.2f}")
                y -= 15
                pdf.drawString(70, y, f"🔠 Grammar Errors: {d['grammar_errors']}")
                y -= 15
                pdf.drawString(70, y, f"💼 Action Verbs: {d['action_verbs_count']}")
                y -= 15
                pdf.drawString(70, y, f"📜 Word Count: {d['word_count']}")
                y -= 25

                pdf.drawString(50, y, "🔍 Missing Keywords:")
                y -= 15
                if d["missing_keywords"]:
                    for kw in d["missing_keywords"]:
                        pdf.drawString(70, y, f"- {kw}")
                        y -= 12
                else:
                    pdf.drawString(70, y, "✅ No missing keywords!")
                    y -= 12

                y -= 15
                pdf.drawString(50, y, "📑 Sections Found:")
                y -= 15
                for k, v in d["sections_found"].items():
                    pdf.drawString(70, y, f"- {k}: {'✔ Found' if v else '❌ Missing'}")
                    y -= 12

                y -= 30
                if y < 150:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 12)
                    y = height - 50

                if d["action_verbs_list"]:
                    pdf.drawString(
                        70, y, f"📝 Verbs: {', '.join(d['action_verbs_list'][:10])}..."
                    )
                    y -= 15

            try:
                pdf.showPage()
                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(200, height - 50, "📊 Overview Chart")
                graph = ImageReader("plot.png")
                pdf.drawImage(graph, 50, height - 400, width=500, height=300)
            except:
                pdf.drawString(50, height - 100, "⚠ Failed to embed chart.")

            pdf.save()
            buffer.seek(0)
            st.download_button(
                "📥 Download PDF Report",
                buffer,
                "resumate_report.pdf",
                mime="application/pdf",
            )

    # --- Chatbot Section (only for authenticated users) ---
    if groq_api_key:
        st.sidebar.title("💬 Resume Assistant Bot")

        # Initialize model
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama-3.1-8b-instant",  # fast & affordable
            groq_api_key=groq_api_key,
        )

        # Conversation memory
        memory = ConversationBufferMemory()

        # Prompt Template
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=(
                "You are Resumate, an intelligent resume mentor bot. "
                "You analyze resumes and job descriptions and provide improvement feedback.\n\n"
                "Conversation history:\n{history}\n\n"
                "Human: {input}\nAI:"
            ),
        )

        # LangChain Conversation
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False,
        )

        # Sidebar chat interface
        user_query = st.sidebar.text_input("Ask Resumate anything...")

        if user_query:
            with st.sidebar:
                with st.spinner("Thinking..."):
                    response = conversation.run(input=user_query)
                st.write("**Resumate:**", response)
    else:
        st.sidebar.error("❌ GROQ_API_KEY not found in .env file")
