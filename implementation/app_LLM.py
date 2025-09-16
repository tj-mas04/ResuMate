# resumate_app_ollama.py
import os
import streamlit as st
import sqlite3, hashlib, datetime, io, time
import PyPDF2, matplotlib.pyplot as plt, numpy as np, pandas as pd
import re, subprocess
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- Ollama Helper Functions ---
def ollama_generate(prompt, model="mistral:instruct"):
    """Generates text using a local Ollama model."""
    result = subprocess.run(
        ["ollama", "generate", model, "--prompt", prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# --- Database Setup ---
conn = sqlite3.connect('resumate.db', check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password_hash TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, username TEXT, timestamp TEXT, resume_name TEXT, similarity REAL, ats_score REAL)""")
conn.commit()

# --- Authentication Helpers ---
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

# --- PDF Text Extraction ---
def extract_text(f): 
    r = PyPDF2.PdfReader(f)
    return "".join(p.extract_text() or "" for p in r.pages).strip() or "No readable text."

# --- Resume Analysis Using Mistral ---
def llm_extract_skills(text):
    prompt = f"Extract the top skills, certifications, and key responsibilities from the resume text:\n{text}\nReturn a list of keywords separated by commas."
    resp = ollama_generate(prompt)
    return [kw.strip() for kw in resp.split(',')]

def llm_similarity(resume_text, jd_text):
    prompt = f"Score the resume's suitability for this job description on a scale of 0-100, based on skills, experience, and relevance. Resume:\n{resume_text}\nJD:\n{jd_text}"
    resp = ollama_generate(prompt)
    try:
        return float(re.findall(r'\d+\.?\d*', resp)[0])
    except:
        return 0.0

def llm_feedback(resume_text, jd_text):
    prompt = f"""
    You are an expert career coach.
    Evaluate this resume against the job description.
    Provide:
    1. Top 3 strengths
    2. Top 3 weaknesses
    3. Actionable suggestions to improve the resume
    Resume: {resume_text}
    JD: {jd_text}
    """
    resp = ollama_generate(prompt)
    return resp

def llm_summary(resume_text):
    prompt = f"Summarize this resume in 2-3 sentences highlighting skills, experience, and achievements:\n{resume_text}"
    resp = ollama_generate(prompt)
    return resp

def llm_ats_score(resume_text, jd_text):
    prompt = f"""
    You are an expert ATS evaluator.
    Evaluate this resume for its suitability against the following job description.
    Consider: 
    - Keywords and skill match
    - Relevance of experience
    - Section completeness
    - Readability and professional formatting
    Provide a single numeric score between 0 (poor) and 100 (perfect fit).

    Resume: {resume_text}
    Job Description: {jd_text}
    """
    resp = ollama_generate(prompt)
    try:
        score = float(re.findall(r'\d+\.?\d*', resp)[0])
        return max(0, min(100, score))
    except:
        return 0.0

# --- Sections Found ---
def sections_found(t): 
    return {s: bool(re.search(s, t, re.I)) for s in ['Education','Experience','Skills','Projects','Certifications']}

# --- Plotting ---
def save_plot(scores, ats):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(scores))
    ax.bar(x - 0.2, list(scores.values()), 0.4, label='Similarity (%)', color="skyblue")
    ax.bar(x + 0.2, list(ats.values()), 0.4, label='ATS Score', color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(list(scores.keys()), rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)
    plt.savefig("plot.png")
    return "plot.png"

# --- Streamlit UI ---
st.set_page_config('ResuMate', '📄', layout='wide')
st.markdown("""<style>
.sidebar .sidebar-content{background:#f0f2f6;width:200px;}
.stButton>button{background:#4CAF50;color:#fff;}
</style>""", unsafe_allow_html=True)

# --- Session State ---
if 'user' not in st.session_state: st.session_state.user = None

# --- Authentication ---
if not st.session_state.user:
    st.title('📄 ResuMate')
    auth_mode = st.sidebar.radio('Account', ['Login', 'Register'])
    with st.form('auth'):
        u = st.text_input('Username')
        pw = st.text_input('Password', type='password')
        ok = st.form_submit_button(auth_mode)
    if ok:
        if auth_mode == 'Login' and authenticate(u, pw): st.session_state.user = u; st.rerun()
        elif auth_mode == 'Register' and register(u, pw): st.success('Registered!')
        else: st.error('Failed')
else:
    st.sidebar.header(f'Welcome {st.session_state.user}')
    page = st.sidebar.selectbox('Menu', ['Home', 'History', 'Logout'])

    if page == 'Logout': st.session_state.user = None; st.rerun()
    elif page == 'History':
        df = pd.read_sql("SELECT timestamp, resume_name, similarity, ats_score FROM history WHERE username=? ORDER BY timestamp DESC", conn, params=(st.session_state.user,))
        st.title("📜 Your History")
        st.dataframe(df if not df.empty else pd.DataFrame({'Info': ['No history available.']}))
    else:
        # --- Main Evaluation ---
        st.title("📄 ResuMate - AI Resume Evaluator")
        jd = st.file_uploader("📑 Upload Job Description (PDF)", type=["pdf"])
        rs = st.file_uploader("📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

        if st.button("🔍 Evaluate Resumes") and jd and rs:
            jd_txt = extract_text(jd)
            scores, ats, details = {}, {}, {}
            progress = st.progress(0)

            for i, r in enumerate(rs):
                progress.progress(i / len(rs))
                t = extract_text(r)

                sim = llm_similarity(t, jd_txt)
                ats_score = llm_ats_score(t, jd_txt)
                skills = llm_extract_skills(t)
                feedback = llm_feedback(t, jd_txt)
                summary = llm_summary(t)
                found_sections = sections_found(t)

                scores[r.name] = sim
                ats[r.name] = ats_score
                details[r.name] = {
                    "similarity": sim,
                    "ats_score": ats_score,
                    "skills": skills,
                    "feedback": feedback,
                    "summary": summary,
                    "sections_found": found_sections
                }

                c.execute("INSERT INTO history(username, timestamp, resume_name, similarity, ats_score) VALUES (?, ?, ?, ?, ?)", (
                    st.session_state.user, datetime.datetime.now().isoformat(), r.name, sim, ats_score
                ))
                time.sleep(0.5)

            conn.commit()
            progress.progress(1.0)
            st.success("✅ Evaluation Completed!")

            # --- Display Results ---
            for name, d in details.items():
                with st.expander(f"📄 Resume: {name}", expanded=True):
                    st.subheader("📄 Summary")
                    st.text(d["summary"])

                    st.subheader("💡 AI Feedback")
                    st.text(d["feedback"])

                    st.subheader("🔑 Skills & Certifications")
                    st.table(pd.DataFrame(d["skills"], columns=["Skills"]))

                    st.subheader("📑 Sections Found")
                    section_df = pd.DataFrame(
                        [{"Section": k, "Status": "✔ Found" if v else "❌ Missing"} for k, v in d["sections_found"].items()]
                    )
                    st.table(section_df)

                    st.metric("📊 Similarity", f"{d['similarity']:.2f}%")
                    st.metric("📖 ATS Score", f"{d['ats_score']:.2f}")

            # --- Overview Chart ---
            st.subheader("📊 Resume Evaluation Overview")
            save_plot(scores, ats)

            # --- PDF Report ---
            buffer = io.BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(180, height - 50, "ResuMate - Resume Evaluation Report")
            y = height - 80
            pdf.setFont("Helvetica", 12)

            for name, d in details.items():
                pdf.drawString(50, y, f"📄 Resume: {name}"); y -= 20
                pdf.drawString(70, y, f"📊 Similarity: {d['similarity']:.2f}%"); y -= 15
                pdf.drawString(70, y, f"📖 ATS Score: {d['ats_score']:.2f}"); y -= 15
                pdf.drawString(70, y, f"🔑 Skills: {', '.join(d['skills'])}"); y -= 15
                pdf.drawString(70, y, f"💡 AI Feedback: {d['feedback']}"); y -= 30

