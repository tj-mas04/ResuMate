"""
History page UI components.
"""
import pandas as pd
import streamlit as st
from database.models import History


def render_history_page(username):
    """
    Render the evaluation history page.
    
    Args:
        username (str): Current logged-in username
    """
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>📊 Evaluation History</h1>
            <p style='font-size: 1.1rem; color: #718096;'>Track your resume evaluation progress over time</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fetch history data
    history_data = History.get_user_history(username)
    
    if history_data:
        # Convert to DataFrame
        df = pd.DataFrame(
            history_data,
            columns=['timestamp', 'resume_name', 'similarity', 'ats_score']
        )
        
        # Summary Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Evaluations", len(df))
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
