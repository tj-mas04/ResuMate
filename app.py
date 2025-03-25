import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textstat
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="ResuMate", page_icon="📄", layout="wide")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages if page.extract_text())
    return text if text.strip() else "No readable text found in the PDF."

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def compute_similarity(job_desc_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([job_desc_text, resume_text])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100

def compute_ats_score(text):
    return textstat.flesch_reading_ease(text)

def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    error_details = [{"Error": match.message, "Sentence": match.context} for match in matches]
    return len(matches), error_details

def find_missing_keywords(jd_keywords, resume_keywords):
    return list(set(jd_keywords) - set(resume_keywords))

def check_resume_sections(text):
    sections = ["Education", "Experience", "Skills", "Projects", "Certifications"]
    return {section: bool(re.search(section, text, re.IGNORECASE)) for section in sections}

def count_action_verbs(text):
    action_verbs = ["led", "managed", "developed", "designed", "initiated", "executed", "created"]
    return sum(text.lower().count(verb) for verb in action_verbs)

def word_count_analysis(text):
    return len(text.split())

def save_plot(resume_scores, resume_ats_scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    resumes = list(resume_scores.keys())
    similarity_scores = list(resume_scores.values())
    ats_scores = list(resume_ats_scores.values())
    x = np.arange(len(resumes))
    width = 0.4
    ax.bar(x - width/2, similarity_scores, width, label="Similarity Score (%)", color="skyblue")
    ax.bar(x + width/2, ats_scores, width, label="ATS Score", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(resumes, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Resume Similarity & ATS Readability Score")
    ax.legend()
    st.pyplot(fig)
    plt.savefig("plot.png")
    return "plot.png"

st.title("📄 ResuMate - AI Resume Evaluator")

st.sidebar.header("Upload Files")
job_desc_file = st.sidebar.file_uploader("📑 Upload Job Description (PDF)", type=["pdf"])
resume_files = st.sidebar.file_uploader("📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("🔍 Evaluate Resumes"):
    if job_desc_file and resume_files:
        with st.spinner("🔄 Evaluating resumes... Please wait!"):
            job_description_text = extract_text_from_pdf(job_desc_file)
            jd_keywords = extract_keywords(job_description_text, top_n=10)
            resume_texts = {res.name: extract_text_from_pdf(res) for res in resume_files}

            resume_scores = {}
            resume_ats_scores = {}
            report_details = {}

            for res, text in resume_texts.items():
                similarity_score = compute_similarity(job_description_text, text)
                ats_score = compute_ats_score(text)
                grammar_errors_count, grammar_errors_list = check_grammar(text)
                resume_keywords = extract_keywords(text, top_n=10)
                missing_keywords = find_missing_keywords(jd_keywords, resume_keywords)
                sections_found = check_resume_sections(text)
                action_verbs_count = count_action_verbs(text)
                word_count = word_count_analysis(text)

                resume_scores[res] = similarity_score
                resume_ats_scores[res] = ats_score
                report_details[res] = {
                    "similarity": similarity_score,
                    "ats_score": ats_score,
                    "grammar_errors": grammar_errors_count,
                    "grammar_errors_details": grammar_errors_list,
                    "missing_keywords": missing_keywords,
                    "sections_found": sections_found,
                    "action_verbs": action_verbs_count,
                    "word_count": word_count,
                }

            st.success("✅ Evaluation Completed!")
            
            # Show results in a structured UI
            for res, details in report_details.items():
                with st.expander(f"📄 **Resume: {res}**", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="📊 Similarity Score", value=f"{details['similarity']:.2f}%")
                        st.metric(label="📖 ATS Readability Score", value=f"{details['ats_score']:.2f}")
                        st.metric(label="🔠 Grammar Errors", value=f"{details['grammar_errors']}")
                    
                    with col2:
                        st.metric(label="💼 Action Verbs Used", value=f"{details['action_verbs']}")
                        st.metric(label="📜 Total Word Count", value=f"{details['word_count']}")
                    
                    # Missing Keywords
                    if details["missing_keywords"]:
                        missing_keywords_df = pd.DataFrame(details["missing_keywords"], columns=["🔍 Missing Keywords"])
                        st.table(missing_keywords_df)
                    else:
                        st.success("✅ No missing keywords!")
                    
                    st.metric(label="🔠 Grammar Errors", value=f"{details['grammar_errors']}")

                    # Display grammar error details
                    if details["grammar_errors_details"]:
                        st.subheader("🔍 Grammar Issues Found")
                        grammar_df = pd.DataFrame(details["grammar_errors_details"])
                        st.table(grammar_df)
                    else:
                        st.success("✅ No grammar issues found!")
                    
                    
                    # Key Sections Found
                    sections_df = pd.DataFrame(
                        [{"Section": sec, "✅ Found": "✔" if present else "❌ Missing"} for sec, present in details["sections_found"].items()]
                    )
                    st.table(sections_df)

            # Generate Graphs
            st.subheader("📊 Resume Evaluation Overview")
            chart_path = save_plot(resume_scores, resume_ats_scores)

            # Generate and Download Report
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter

            c.setFont("Helvetica-Bold", 16)
            c.drawString(200, height - 50, "ResuMate - Resume Evaluation Report")

            y_position = height - 80
            c.setFont("Helvetica", 12)

            for res, details in report_details.items():
                c.drawString(50, y_position, f"📄 Resume: {res}")
                y_position -= 20

                c.drawString(70, y_position, f"📊 Similarity Score: {details['similarity']:.2f}%")
                y_position -= 15
                c.drawString(70, y_position, f"📖 ATS Readability Score: {details['ats_score']:.2f}")
                y_position -= 15
                c.drawString(70, y_position, f"🔠 Grammar Errors: {details['grammar_errors']}")
                y_position -= 15
                c.drawString(70, y_position, f"💼 Action Verbs Used: {details['action_verbs']}")
                y_position -= 15
                c.drawString(70, y_position, f"📜 Total Word Count: {details['word_count']}")
                y_position -= 25

                # Missing Keywords
                c.drawString(50, y_position, "🔍 Missing Keywords:")
                y_position -= 15
                if details["missing_keywords"]:
                    for keyword in details["missing_keywords"]:
                        c.drawString(70, y_position, f"- {keyword}")
                        y_position -= 12
                else:
                    c.drawString(70, y_position, "✅ No missing keywords!")
                    y_position -= 12

                y_position -= 20

                # Resume Sections Found
                c.drawString(50, y_position, "📑 Resume Sections Found:")
                y_position -= 15
                for section, present in details["sections_found"].items():
                    status = "✔ Found" if present else "❌ Missing"
                    c.drawString(70, y_position, f"- {section}: {status}")
                    y_position -= 12

                y_position -= 30  # Space before next resume

                if y_position < 100:  # Prevents text from going off the page
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = height - 50
                
                try:
                    chart_image = "plot.png"  # Ensure the graph is saved before this point
                    c.showPage()  # New page for the graph
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(200, height - 50, "📊 Resume Evaluation Overview")

                    graph = ImageReader(chart_image)
                    c.drawImage(graph, 50, height - 400, width=500, height=300)  # Adjust size & position
                except Exception as e:
                    c.drawString(50, height - 100, "⚠ Error embedding graph.")

            c.save()
            buffer.seek(0)
           
            st.download_button(label="📥 Download Report", data=buffer, file_name="resume_evaluation_report.pdf", mime="application/pdf")
    
    else:
        st.error("⚠ Please upload both Job Description and at least one Resume!")
