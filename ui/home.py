"""
Home page UI for resume evaluation.
"""
import time
import pandas as pd
import streamlit as st
from database.models import History
from services.pdf_service import extract_text_from_pdf
from utils.visualization import create_evaluation_plot
from utils.pdf_generator import generate_pdf_report


def render_home_page(nlp_service, ats_service, grammar_service, ai_service):
    """
    Render the main home page for resume evaluation.
    
    Args:
        nlp_service: Instance of NLPService
        ats_service: Instance of ATSService  
        grammar_service: Instance of GrammarService
        ai_service: Instance of AIService
    """
    # Page Header
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
        jd_file = st.file_uploader(
            "Upload JD (PDF)",
            type=["pdf"],
            help="Upload the job description you want to match against",
            label_visibility="collapsed"
        )
        if jd_file:
            st.success(f"✅ Loaded: {jd_file.name}")
    
    with col2:
        st.markdown("### 📄 Resumes")
        resume_files = st.file_uploader(
            "Upload Resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or multiple resumes to evaluate",
            label_visibility="collapsed"
        )
        if resume_files:
            st.success(f"✅ Loaded {len(resume_files)} resume(s)")
    
    st.markdown("---")
    
    # Evaluation Button
    if st.button("🔍 Evaluate Resumes") and jd_file and resume_files:
        _evaluate_resumes(
            jd_file, resume_files,
            nlp_service, ats_service, grammar_service, ai_service
        )


def _evaluate_resumes(jd_file, resume_files, nlp_service, ats_service, grammar_service, ai_service):
    """Handle the resume evaluation process."""
    # Extract JD text
    jd_text = extract_text_from_pdf(jd_file)
    st.session_state.jd_text = jd_text
    
    scores, ats_scores, details = {}, {}, {}
    progress = st.progress(0)
    
    # Process each resume
    for i, resume_file in enumerate(resume_files):
        resume_text = extract_text_from_pdf(resume_file)
        
        # Compute similarity
        similarity = nlp_service.compute_similarity(jd_text, resume_text)
        
        # Compute ATS score
        ats_score, matched_skills, missing_skills = ats_service.compute_ats_score(
            resume_text, jd_text
        )
        
        # Store for visualization
        scores[resume_file.name] = similarity
        ats_scores[resume_file.name] = ats_score
        
        # Count action verbs
        verbs_count, verbs_list = nlp_service.count_action_verbs(resume_text)
        
        # Grammar check
        grammar_errors, grammar_details = grammar_service.check_grammar(resume_text)
        
        # Extract keywords
        jd_keywords = nlp_service.extract_keywords(jd_text, top_n=20)
        resume_keywords = nlp_service.extract_keywords(resume_text, top_n=20)
        missing_keywords = ats_service.get_missing_keywords(jd_keywords, resume_keywords)
        
        # Compile details
        details[resume_file.name] = {
            "similarity": similarity,
            "ats_score": ats_score,
            "grammar_errors": grammar_errors,
            "grammar_details": grammar_details,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "missing_keywords": missing_keywords,
            "sections_found": ats_service.check_sections(resume_text),
            "action_verbs_count": verbs_count,
            "action_verbs_list": verbs_list,
            "word_count": len(resume_text.split()),
        }
        
        # Generate AI recommendation
        details[resume_file.name]["recommendation"] = ai_service.generate_recommendation(
            details[resume_file.name]
        )
        
        # Save to history
        History.add_record(
            st.session_state.user,
            resume_file.name,
            similarity,
            ats_score
        )
        
        # Update progress
        progress.progress((i + 1) / len(resume_files))
        time.sleep(0.3)
    
    # Store in session state
    st.session_state.resume_details = details
    
    st.success("✅ Evaluation Completed!")
    
    # Display results
    _display_results(details, scores, ats_scores)


def _display_results(details, scores, ats_scores):
    """Display detailed evaluation results."""
    # Individual resume results
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
                    st.write(", ".join(d["action_verbs_list"][:15]))
                st.metric("📜 Word Count", f"{d['word_count']}")
            
            # Matched Skills
            if d["matched_skills"]:
                st.subheader("✅ Matched Skills")
                st.table(pd.DataFrame(d["matched_skills"], columns=["Skill"]))
            
            # Missing Skills
            if d["missing_skills"]:
                st.subheader("❌ Missing Skills")
                st.table(pd.DataFrame(d["missing_skills"], columns=["Skill"]))
            
            # Missing Keywords
            if d["missing_keywords"]:
                st.subheader("🔍 Missing Keywords")
                st.table(pd.DataFrame(d["missing_keywords"], columns=["Keyword"]))
            else:
                st.success("✅ No missing keywords!")
            
            # Grammar Issues
            if d["grammar_details"]:
                st.subheader("🔠 Grammar Issues")
                st.table(pd.DataFrame(d["grammar_details"]))
            else:
                st.success("✅ No grammar issues found.")
            
            # Sections Found
            st.subheader("📑 Sections Found")
            section_df = pd.DataFrame([
                {"Section": k, "Status": "✔ Found" if v else "❌ Missing"}
                for k, v in d["sections_found"].items()
            ])
            st.table(section_df)
            
            # AI Recommendation
            st.subheader("💡 Tailored Recommendation")
            st.write(d["recommendation"])
    
    # Leaderboard
    _display_leaderboard(details)
    
    # Visual Analysis
    _display_charts(scores, ats_scores)
    
    # PDF Report
    _generate_download_button(details)


def _display_leaderboard(details):
    """Display resume leaderboard."""
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; margin: 2rem 0 1rem 0;'>
            <h2>🏆 Resume Leaderboard</h2>
            <p style='color: #718096;'>Ranked by similarity score</p>
        </div>
    """, unsafe_allow_html=True)
    
    ranked = sorted(
        details.items(),
        key=lambda x: x[1]["similarity"],
        reverse=True
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


def _display_charts(scores, ats_scores):
    """Display visual analysis charts."""
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; margin: 2rem 0 1rem 0;'>
            <h2>📊 Visual Analysis</h2>
            <p style='color: #718096;'>Comprehensive comparison of all evaluated resumes</p>
        </div>
    """, unsafe_allow_html=True)
    create_evaluation_plot(scores, ats_scores)


def _generate_download_button(details):
    """Generate PDF report download button."""
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; margin: 2rem 0 1rem 0;'>
            <h2>📄 Export Results</h2>
            <p style='color: #718096;'>Download a comprehensive PDF report of your evaluation</p>
        </div>
    """, unsafe_allow_html=True)
    
    buffer = generate_pdf_report(details, plot_path="plot.png")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            "📥 Download Complete PDF Report",
            buffer,
            "resumate_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
