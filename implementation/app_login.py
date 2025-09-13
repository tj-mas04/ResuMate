# resumate_app.py
import streamlit as st
import sqlite3, hashlib, datetime, io, re, time
import PyPDF2, matplotlib.pyplot as plt, numpy as np, pandas as pd, textstat, language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- Database Setup ---
conn = sqlite3.connect('resumate.db', check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password_hash TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, username TEXT, timestamp TEXT, resume_name TEXT, similarity REAL, ats_score REAL)""")
conn.commit()

def hash_pw(password): return hashlib.sha256(password.encode()).hexdigest()
def register(u, pw): 
    try: c.execute("INSERT INTO users VALUES (?, ?)", (u, hash_pw(pw))); conn.commit(); return True
    except: return False
def authenticate(u, pw):
    c.execute("SELECT password_hash FROM users WHERE username=?", (u,))
    row = c.fetchone()
    return row and row[0] == hash_pw(pw)

# --- Evaluation Logic (Your Original Code) ---
def extract_text(f): 
    r = PyPDF2.PdfReader(f)
    return "".join(p.extract_text() or "" for p in r.pages).strip() or "No readable text."

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def compute_similarity(a, b):
    v = TfidfVectorizer(stop_words='english'); m = v.fit_transform([a, b])
    return cosine_similarity(m[0], m[1])[0][0] * 100

def compute_ats(t): return textstat.flesch_reading_ease(t)
def check_grammar(t): 
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(t)
    return len(matches), [{'Error': m.message, 'Sentence': m.context} for m in matches]

def missing_kw(jd, res): return list(set(jd) - set(res))
def sections_found(t): return {s: bool(re.search(s, t, re.I)) for s in ['Education','Experience','Skills','Projects','Certifications']}
def count_verbs(t): return sum(t.lower().count(v) for v in ['led','managed','developed','designed','initiated','executed','created'])
def word_count(t): return len(t.split())

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

# --- UI Setup ---
st.set_page_config('ResuMate', '📄', layout='wide')
st.markdown("""<style>
.sidebar .sidebar-content{background:#f0f2f6;width:200px;}
.stButton>button{background:#4CAF50;color:#fff;}
</style>""", unsafe_allow_html=True)

# --- Session State ---
if 'user' not in st.session_state: st.session_state.user = None

# --- Authentication UI ---
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
        # Main Resume Evaluation
        st.title("📄 ResuMate - AI Resume Evaluator")
        jd = st.file_uploader("📑 Upload Job Description (PDF)", type=["pdf"])
        rs = st.file_uploader("📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

        if st.button("🔍 Evaluate Resumes") and jd and rs:
            jd_txt = extract_text(jd)
            jd_kw = extract_keywords(jd_txt)
            scores, ats, details = {}, {}, {}
            progress = st.progress(0)

            for i, r in enumerate(rs):
                progress.progress(i / len(rs))
                t = extract_text(r)
                sim = compute_similarity(jd_txt, t)
                ats_score = compute_ats(t)
                grammar_count, grammar_list = check_grammar(t)
                res_kw = extract_keywords(t)
                missing = missing_kw(jd_kw, res_kw)
                found_sections = sections_found(t)
                verbs = count_verbs(t)
                wc = word_count(t)

                scores[r.name] = sim
                ats[r.name] = ats_score
                details[r.name] = {
                    "similarity": sim,
                    "ats_score": ats_score,
                    "grammar_errors": grammar_count,
                    "grammar_details": grammar_list,
                    "missing_keywords": missing,
                    "sections_found": found_sections,
                    "action_verbs": verbs,
                    "word_count": wc,
                }

                c.execute("INSERT INTO history(username, timestamp, resume_name, similarity, ats_score) VALUES (?, ?, ?, ?, ?)", (
                    st.session_state.user, datetime.datetime.now().isoformat(), r.name, sim, ats_score
                ))
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
                        st.metric("💼 Action Verbs", f"{d['action_verbs']}")
                        st.metric("📜 Word Count", f"{d['word_count']}")

                    if d["missing_keywords"]:
                        st.subheader("🔍 Missing Keywords")
                        st.table(pd.DataFrame(d["missing_keywords"], columns=["Keyword"]))
                    else:
                        st.success("✅ No missing keywords!")

                    if d["grammar_details"]:
                        st.subheader("🔠 Grammar Issues")
                        st.table(pd.DataFrame(d["grammar_details"]))
                    else:
                        st.success("✅ No grammar issues found.")

                    st.subheader("📑 Sections Found")
                    section_df = pd.DataFrame(
                        [{"Section": k, "Status": "✔ Found" if v else "❌ Missing"} for k, v in d["sections_found"].items()]
                    )
                    st.table(section_df)

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
                pdf.drawString(50, y, f"📄 Resume: {name}"); y -= 20
                pdf.drawString(70, y, f"📊 Similarity: {d['similarity']:.2f}%"); y -= 15
                pdf.drawString(70, y, f"📖 ATS Score: {d['ats_score']:.2f}"); y -= 15
                pdf.drawString(70, y, f"🔠 Grammar Errors: {d['grammar_errors']}"); y -= 15
                pdf.drawString(70, y, f"💼 Action Verbs: {d['action_verbs']}"); y -= 15
                pdf.drawString(70, y, f"📜 Word Count: {d['word_count']}"); y -= 25

                pdf.drawString(50, y, "🔍 Missing Keywords:"); y -= 15
                if d["missing_keywords"]:
                    for kw in d["missing_keywords"]:
                        pdf.drawString(70, y, f"- {kw}"); y -= 12
                else:
                    pdf.drawString(70, y, "✅ No missing keywords!"); y -= 12

                y -= 15
                pdf.drawString(50, y, "📑 Sections Found:"); y -= 15
                for k, v in d["sections_found"].items():
                    pdf.drawString(70, y, f"- {k}: {'✔ Found' if v else '❌ Missing'}"); y -= 12

                y -= 30
                if y < 150: pdf.showPage(); pdf.setFont("Helvetica", 12); y = height - 50

            try:
                pdf.showPage()
                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(200, height - 50, "📊 Overview Chart")
                graph = ImageReader("plot.png")
                pdf.drawImage(graph, 50, height - 400, width=500, height=300)
            except:
                pdf.drawString(50, height - 100, "⚠ Failed to embed chart.")

            pdf.save(); buffer.seek(0)
            st.download_button("📥 Download PDF Report", buffer, "resumate_report.pdf", mime="application/pdf")
