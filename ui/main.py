from __future__ import annotations
import time
import datetime
import pandas as pd
import streamlit as st

from config import GROQ_API_KEY
from core.evaluation import (
    extract_text,
    extract_keywords,
    compute_similarity,
    compute_ats,
    sections_found,
    word_count,
)
from services.nlp_service import count_action_verbs
from services.grammar_service import check_grammar
from services.db_service import register, authenticate, insert_history, get_history_for_user
from services.plot_service import save_plot
from services.pdf_service import generate_pdf_report
from services.llm_service import generate_recommendation, chat_reply


def _inject_css():
    from pathlib import Path
    css_path = Path(__file__).resolve().parents[2] / 'resumate' / 'assets' / 'style.css'
    try:
        css = css_path.read_text(encoding='utf-8')
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        # Fallback: load nothing but log silently
        pass


def run():
    st.set_page_config("ResuMate", "📄", layout="wide")
    _inject_css()

    if "user" not in st.session_state:
        st.session_state.user = None
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = None
    if "resume_details" not in st.session_state:
        st.session_state.resume_details = {}
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = None

    if not st.session_state.user:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0;'>
                <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>📄 ResuMate</h1>
                <p style='font-size: 1.2rem; color: #718096;'>Your AI-Powered Resume Evaluation Assistant</p>
                <p style='font-size: 1rem; color: #a0aec0;'>Analyze resumes • Get ATS Scores • Improve Your Career</p>
            </div>
        """, unsafe_allow_html=True)

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
        return

    # Authenticated UI
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
        return

    if page == "📊 History":
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1>📊 Evaluation History</h1>
                <p style='font-size: 1.1rem; color: #718096;'>Track your resume evaluation progress over time</p>
            </div>
        """, unsafe_allow_html=True)

        df = pd.read_sql(
            "SELECT timestamp, resume_name, similarity, ats_score FROM history WHERE username=? ORDER BY timestamp DESC",
            get_history_for_user.__self__.conn if hasattr(get_history_for_user, '__self__') else None,
        ) if False else pd.DataFrame()
        # Fallback: direct query via service function
        rows = list(get_history_for_user(st.session_state.user))
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "resume_name", "similarity", "ats_score"])

        if not df.empty:
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
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            df.columns = ['📅 Date', '📄 Resume', '🔗 Similarity %', '🎯 ATS Score']
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No evaluation history yet. Start by uploading resumes on the Home page!")
        return

    # Home Page
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>📄 AI Resume Evaluator</h1>
            <p style='font-size: 1.1rem; color: #718096;'>Upload your job description and resumes to get instant AI-powered insights</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📑 Job Description")
        jd = st.file_uploader("Upload JD (PDF)", type=["pdf"], help="Upload the job description you want to match against", label_visibility="collapsed")
        if jd:
            st.success(f"✅ Loaded: {jd.name}")
    with col2:
        st.markdown("### 📄 Resumes")
        rs = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True, help="Upload one or multiple resumes to evaluate", label_visibility="collapsed")
        if rs:
            st.success(f"✅ Loaded {len(rs)} resume(s)")

    st.markdown("---")

    if st.button("🔍 Evaluate Resumes") and jd and rs:
        jd_txt = extract_text(jd)
        st.session_state.jd_text = jd_txt
        scores, ats, details = {}, {}, {}

        anim_placeholder = st.empty()
        anim_placeholder.markdown("""
            <style>
            .resumate-anim-bg { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; background: linear-gradient(270deg, #0f172a, #1e293b, #0ea5e9, #6ee7b7); background-size: 600% 600%; opacity: 0.14; animation: resumateGradient 18s ease infinite; }
            @keyframes resumateGradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
            </style>
            <div class="resumate-anim-bg"></div>
        """, unsafe_allow_html=True)

        progress = st.progress(0)

        for i, r in enumerate(rs):
            t = extract_text(r)
            sim = compute_similarity(jd_txt, t)
            ats_score, matched_skills, missing_skills = compute_ats(t, jd_txt)
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
                "missing_keywords": list(set(extract_keywords(jd_txt)) - set(extract_keywords(t))),
                "sections_found": sections_found(t),
                "action_verbs_count": verbs_count,
                "action_verbs_list": verbs_list,
                "word_count": word_count(t),
            }

            details[r.name]["recommendation"] = generate_recommendation(details[r.name])

            insert_history(
                st.session_state.user,
                datetime.datetime.now().isoformat(),
                r.name,
                sim,
                ats_score,
            )
            time.sleep(0.5)
            progress.progress((i + 1) / max(1, len(rs)))

        try:
            anim_placeholder.empty()
        except Exception:
            pass

        st.session_state.resume_details = details
        st.success("✅ Evaluation Completed!")

        # Persist scores/ats so plots can be switched without re-evaluating
        st.session_state.scores = scores
        st.session_state.ats = ats
        if "plot_type" not in st.session_state:
            st.session_state.plot_type = "Auto (recommended)"

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
                        [{"Section": k, "Status": "✔ Found" if v else "❌ Missing"} for k, v in d["sections_found"].items()]
                    )
                    st.table(section_df)

                st.subheader("💡 Tailored Recommendation")
                st.write(details[name]["recommendation"])

        # Create and cache initial plot and PDF for download
        # Save initial plot (prefer in-memory bytes if available)
        plot_bytes = save_plot(st.session_state.scores, st.session_state.ats, st.session_state.plot_type, display=False, return_bytes=True)
        st.session_state.plot_png_bytes = plot_bytes
        st.session_state.plot_png = None if plot_bytes else save_plot(st.session_state.scores, st.session_state.ats, st.session_state.plot_type, display=False, return_bytes=False)
        # Show diagnostic info about the plot export
        export_err = st.session_state.get("_plot_export_error")
        pb = st.session_state.get("plot_png_bytes")
        if isinstance(pb, (bytes, bytearray)):
            st.success(f"Plot exported to memory ({len(pb)} bytes)")
        elif isinstance(st.session_state.get("plot_png"), str):
            st.success(f"Plot saved to disk: {st.session_state.get('plot_png')}")
        elif export_err:
            st.warning(f"Plot export warning: {export_err}")
        else:
            st.info("Plot export not available; PDF may not include a chart.")

        # Generate initial PDF (embed current saved plot bytes or file if available)
        try:
            pb = st.session_state.get("plot_png_bytes")
            if isinstance(pb, (bytes, bytearray)):
                st.session_state.pdf_buffer = generate_pdf_report(details, plot_bytes=pb)
            else:
                plot_path = st.session_state.get("plot_png")
                if isinstance(plot_path, str):
                    st.session_state.pdf_buffer = generate_pdf_report(details, plot_path=plot_path)
                else:
                    st.session_state.pdf_buffer = generate_pdf_report(details, plot_path=None)
        except Exception:
            st.session_state.pdf_buffer = None

    # If there are cached evaluation results, show leaderboard & visualisations without re-evaluating
    if st.session_state.get("resume_details"):
        details = st.session_state.resume_details

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                <h2>🏆 Resume Leaderboard</h2>
                <p style='color: #718096;'>Ranked by similarity score</p>
            </div>
        """, unsafe_allow_html=True)

        ranked = sorted(details.items(), key=lambda x: x[1]["similarity"], reverse=True)
        for i, (name, d) in enumerate(ranked, start=1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            st.markdown(f"""
                <div class='leaderboard-item'>
                    <span style='font-size: 1.5rem; margin-right: 1rem;'>{medal}</span>
                    <strong style='font-size: 1.1rem; color: #ffffff;'>{name}</strong>
                    <span style='float: right; color: #3182ce; font-weight: 600; font-size: 1.1rem;'>{d['similarity']:.2f}% match</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                <h2>📊 Visual Analysis</h2>
                <p style='color: #718096;'>Comprehensive comparison of all evaluated resumes</p>
            </div>
        """, unsafe_allow_html=True)

        if "plot_type" not in st.session_state:
            st.session_state.plot_type = "Auto (recommended)"

        plot_type = st.selectbox(
            "📈 Choose plot type",
            [
                "Auto (recommended)",
                "Horizontal Grouped Bar",
                "Radar Chart",
                "Lollipop Chart",
                "Heatmap",
            ],
            key="plot_type",
        )

        # Render the selected plot using cached scores/ats (no re-evaluation)
        save_plot(st.session_state.scores, st.session_state.ats, st.session_state.plot_type)

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                <h2>📄 Export Results</h2>
                <p style='color: #718096;'>Download a comprehensive PDF report of your evaluation</p>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            buf = st.session_state.get("pdf_buffer")
            if buf:
                st.download_button(
                    "📥 Download Complete PDF Report",
                    buf,
                    "resumate_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            # if st.button("🔁 Regenerate PDF (include current plot)"):
            #     # regenerate static plot bytes (save-only) and then PDF
            #     saved_bytes = save_plot(st.session_state.scores, st.session_state.ats, st.session_state.plot_type, display=False, return_bytes=True)
            #     if saved_bytes:
            #         st.session_state.plot_png_bytes = saved_bytes
            #         st.session_state.plot_png = None
            #         st.success(f"Plot exported to memory ({len(saved_bytes)} bytes)")
            #     else:
            #         # fallback: try file-based save
            #         saved_path = save_plot(st.session_state.scores, st.session_state.ats, st.session_state.plot_type, display=False, return_bytes=False)
            #         if saved_path:
            #             st.session_state.plot_png = saved_path
            #             st.session_state.plot_png_bytes = None
            #             st.success(f"Plot saved to disk: {saved_path}")
            #         else:
            #             export_err = st.session_state.get("_plot_export_error")
            #             if export_err:
            #                 st.error(f"⚠️ Could not export the plot as a static image: {export_err}. Ensure 'kaleido' is installed (pip install kaleido). The PDF will use the previously-saved plot if present.")
            #             else:
            #                 st.error("⚠️ Could not export the plot as a static image. Ensure 'kaleido' is installed (pip install kaleido). The PDF will use the previously-saved plot if present.")
            #     try:
            #         pb = st.session_state.get("plot_png_bytes")
            #         if isinstance(pb, (bytes, bytearray)):
            #             st.session_state.pdf_buffer = generate_pdf_report(st.session_state.resume_details, plot_bytes=pb)
            #         else:
            #             plot_path = st.session_state.get("plot_png")
            #             if isinstance(plot_path, str):
            #                 st.session_state.pdf_buffer = generate_pdf_report(st.session_state.resume_details, plot_path=plot_path)
            #             else:
            #                 st.session_state.pdf_buffer = generate_pdf_report(st.session_state.resume_details, plot_path=None)
            #         st.success("✅ PDF regenerated with the current plot (if export succeeded).")
            #     except Exception:
            #         st.error("❌ Failed to regenerate PDF. Make sure plot export dependencies are installed and writable permissions exist.")

    # Chatbot Section
    st.sidebar.markdown("---")
    if GROQ_API_KEY:
        st.sidebar.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='margin: 0;'>💬 AI Assistant</h2>
                <p style='font-size: 0.9rem; color: #a0aec0; margin: 0.5rem 0 0 0;'>Get personalized resume advice</p>
            </div>
        """, unsafe_allow_html=True)

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

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

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

        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        user_query = st.sidebar.text_input("💭 Ask anything about your resumes...", key="chat_input", placeholder="e.g., Which resume is best?")

        if user_query:
            with st.sidebar:
                with st.spinner("🤔 Thinking..."):
                    msgs = []
                    system_msg = "You are Resumate, an intelligent resume mentor bot. Use the following context to inform your responses:\n" + context_info
                    msgs.append({"role": "system", "content": system_msg})
                    for role, content in st.session_state.chat_history:
                        msgs.append({"role": role, "content": content})
                    msgs.append({"role": "user", "content": user_query})

                    reply = chat_reply(msgs)
                    st.session_state.chat_history.append(("user", user_query))
                    st.session_state.chat_history.append(("assistant", reply))

                st.markdown(f"""
                    <div style='background: rgba(49, 130, 206, 0.15); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #3182ce;'>
                        <strong style='color: #ffffff;'>🤖 Resumate:</strong><br/>
                        <span style='font-size: 0.95rem; color: #ffffff;'>{reply}</span>
                    </div>
                """, unsafe_allow_html=True)

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
