"""
Chatbot UI components.
"""
import streamlit as st
from services.ai_service import AIService


def render_chatbot(ai_service, jd_text=None, resume_details=None):
    """
    Render the AI chatbot in the sidebar.
    
    Args:
        ai_service: Instance of AIService
        jd_text (str, optional): Job description text
        resume_details (dict, optional): Resume evaluation details
    """
    st.sidebar.markdown("---")
    
    if not ai_service:
        st.sidebar.markdown("""
            <div style='background: rgba(245, 101, 101, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #f56565;'>
                <strong>❌ Configuration Error</strong><br/>
                <span style='font-size: 0.9rem;'>GROQ_API_KEY not found in .env file</span>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='margin: 0;'>💬 AI Assistant</h2>
            <p style='font-size: 0.9rem; color: #a0aec0; margin: 0.5rem 0 0 0;'>Get personalized resume advice</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Build context from uploaded documents
    context_info = ""
    if jd_text:
        jd_summary = jd_text[:500] + "..." if len(jd_text) > 500 else jd_text
        context_info += f"\n\n**JOB DESCRIPTION SUMMARY:**\n{jd_summary}\n"
    
    if resume_details:
        context_info += "\n**RESUME EVALUATION RESULTS:**\n"
        for resume_name, details in resume_details.items():
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
    
    # Initialize conversation if not exists
    if st.session_state.chat_memory is None:
        conversation, memory = ai_service.create_chatbot(context_info)
        st.session_state.chat_memory = memory
        st.session_state.conversation = conversation
    
    # Display chat history
    if hasattr(st.session_state.chat_memory, 'chat_memory') and st.session_state.chat_memory.chat_memory.messages:
        with st.sidebar.expander("📜 Chat History", expanded=False):
            for msg in st.session_state.chat_memory.chat_memory.messages:
                role = "You" if "Human" in str(type(msg)) else "Resumate"
                icon = "👤" if role == "You" else "🤖"
                st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                        <strong>{icon} {role}:</strong><br/>
                        <span style='font-size: 0.9rem;'>{msg.content}</span>
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat interface
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    user_query = st.sidebar.text_input(
        "💭 Ask anything about your resumes...",
        key="chat_input",
        placeholder="e.g., Which resume is best?"
    )
    
    if user_query and hasattr(st.session_state, 'conversation'):
        with st.sidebar:
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.conversation.run(input=user_query)
            st.markdown(f"""
                <div style='background: rgba(49, 130, 206, 0.15); 
                            padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #3182ce;'>
                    <strong style='color: #ffffff;'>🤖 Resumate:</strong><br/>
                    <span style='font-size: 0.95rem; color: #ffffff;'>{response}</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Show helpful prompts
    if not jd_text or not resume_details:
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
