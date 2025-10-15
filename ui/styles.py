"""
Custom CSS styles for the application.
"""
import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """<style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main Background */
        .stApp {
            background: #f8f9fa;
        }
        
        /* Main Content Area */
        .main .block-container {
            padding: 2rem 3rem;
            background: #ffffff;
            border-radius: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin: 2rem auto;
            max-width: 1400px;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Title Styling */
        h1 {
            color: #1a202c !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 1rem !important;
            text-align: center;
        }
        
        h2 {
            color: #2d3748 !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: #3182ce !important;
            font-weight: 600 !important;
        }
        
        /* Button Styling */
        .stButton>button {
            background: #3182ce;
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(49, 130, 206, 0.3);
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.5);
            background: #2c5282;
        }
        
        .stButton>button:active {
            transform: translateY(0px);
        }
        
        /* Download Button Styling */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
        }
        
        .stDownloadButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(72, 187, 120, 0.6);
        }
        
        /* Input Fields */
        .stTextInput>div>div>input,
        .stSelectbox>div>div>select {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 0.75rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            color: #1a202c !important;
        }
        
        .stTextInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus {
            border-color: #3182ce;
            box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: #f7fafc;
            border: 2px dashed #cbd5e0;
            border-radius: 15px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #3182ce;
            background: #edf2f7;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #3182ce !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #2d3748 !important;
            font-weight: 600 !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #f7fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            font-weight: 600 !important;
            color: #1a202c !important;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: #edf2f7;
            border-color: #3182ce;
        }
        
        .streamlit-expanderContent {
            border: 1px solid #e2e8f0;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 1.5rem;
            background: #ffffff;
        }
        
        /* Tables */
        .dataframe {
            border: none !important;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .dataframe thead tr th {
            background: #3182ce;
            color: white !important;
            font-weight: 600 !important;
            padding: 1rem;
            border: none !important;
        }
        
        .dataframe tbody tr:nth-child(even) {
            background-color: #f7fafc;
        }
        
        .dataframe tbody tr:hover {
            background-color: #edf2f7;
            transition: all 0.2s ease;
        }
        
        .dataframe tbody tr td {
            padding: 0.75rem;
            border: none !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: #3182ce;
            border-radius: 10px;
        }
        
        /* Success/Error/Info Messages */
        .stSuccess {
            background-color: #c6f6d5;
            color: #22543d;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #48bb78;
        }
        
        .stError {
            background-color: #fed7d7;
            color: #742a2a;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #f56565;
        }
        
        .stInfo {
            background-color: #bee3f8;
            color: #2c5282;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #4299e1;
        }
        
        /* Leaderboard Styling */
        .leaderboard-item {
            background: #f7fafc;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            border-left: 4px solid #3182ce;
            transition: all 0.3s ease;
        }
        
        .leaderboard-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.15);
            background: #edf2f7;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #3182ce;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #2c5282;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .main .block-container > div {
            animation: fadeIn 0.5s ease-out;
        }
    </style>""",
        unsafe_allow_html=True,
    )
