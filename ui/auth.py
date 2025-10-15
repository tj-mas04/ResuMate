"""
Authentication UI components.
"""
import time
import streamlit as st
from database.models import User


def render_auth_page():
    """Render the authentication (login/register) page."""
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
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button(f"{'🚀 Login' if auth_mode == 'Login' else '✨ Create Account'}")
    
    if submit:
        if auth_mode == "Login":
            if User.authenticate(username, password):
                st.session_state.user = username
                st.success(f"✅ Welcome back, {username}!")
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
        else:  # Register
            if User.register(username, password):
                st.success("🎉 Registration successful! Please login.")
                st.balloons()
            else:
                st.error("❌ Username already exists. Please choose another.")
