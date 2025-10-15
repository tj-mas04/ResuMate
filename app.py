"""
ResuMate - AI-Powered Resume Evaluation System
Main application entry point.
"""
import time
import streamlit as st

# Import configuration
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, GROQ_API_KEY

# Import database
from database import init_database

# Import services
from services import NLPService, ATSService, GrammarService, AIService

# Import UI components
from ui import (
    apply_custom_styles,
    render_auth_page,
    render_home_page,
    render_history_page,
    render_chatbot
)

# Page configuration
st.set_page_config(PAGE_TITLE, PAGE_ICON, layout=LAYOUT)

# Apply custom styling
apply_custom_styles()

# Initialize database
init_database()

# Initialize services (with caching)
@st.cache_resource
def init_services():
    """Initialize all services with caching."""
    nlp_service = NLPService()
    grammar_service = GrammarService()
    ats_service = ATSService(nlp_service, grammar_service)
    
    # AI Service (optional, depends on API key)
    try:
        ai_service = AIService() if GROQ_API_KEY else None
    except ValueError:
        ai_service = None
    
    return nlp_service, ats_service, grammar_service, ai_service


# Load services
nlp_service, ats_service, grammar_service, ai_service = init_services()


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = None
    if "resume_details" not in st.session_state:
        st.session_state.resume_details = {}
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = None


init_session_state()


# Main application logic
def main():
    """Main application function."""
    
    # Check if user is authenticated
    if not st.session_state.user:
        # Show authentication page
        render_auth_page()
    else:
        # User is logged in - show sidebar with user info
        st.sidebar.markdown(f"""
            <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👤</div>
                <h3 style='margin: 0; color: #ffffff;'>Welcome</h3>
                <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0; color: #a0aec0;'>{st.session_state.user}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        page = st.sidebar.selectbox("📋 Navigation", ["🏠 Home", "📊 History", "🚪 Logout"])
        
        # Route to appropriate page
        if page == "🚪 Logout":
            # Clear session state and logout
            st.session_state.user = None
            st.session_state.jd_text = None
            st.session_state.resume_details = {}
            st.session_state.chat_memory = None
            st.success("👋 Logged out successfully!")
            time.sleep(1)
            st.rerun()
        
        elif page == "📊 History":
            # Show history page
            render_history_page(st.session_state.user)
        
        else:  # Home page
            # Show home page
            render_home_page(
                nlp_service,
                ats_service,
                grammar_service,
                ai_service
            )
        
        # Render chatbot in sidebar (for Home and History pages)
        if page != "🚪 Logout":
            render_chatbot(
                ai_service,
                st.session_state.jd_text,
                st.session_state.resume_details
            )


if __name__ == "__main__":
    main()
