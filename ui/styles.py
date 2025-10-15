"""
Custom CSS styles for the application.
"""
import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """<style>
        /* Make all Streamlit subheaders and headers white for visibility */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stSubheader, .stHeader, .st-expanderHeader, .st-expanderHeader * {
            color: #fff !important;
        }
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        /* --- Custom Orange/Blue Contrast Theme --- */
        /* App background: dark blue */
        .stApp {
            background: linear-gradient(135deg, #181c2f 0%, #232946 100%);
        }

        /* Main content: dark card with orange border */
        .main .block-container {
            background: #232946;
            border-radius: 18px;
            box-shadow: 0 4px 32px rgba(24,28,47,0.18);
            border: 2px solid #ff9800;
            margin: 2rem auto;
            max-width: 1400px;
            padding: 2rem 3rem;
            color: #fff !important;
        }

        /* All main/result text: pure white for contrast on dark */
        .main .block-container, .main .block-container *,
        .streamlit-expanderContent, .streamlit-expanderContent *,
        .dataframe, .dataframe *,
        .stMetricValue, .stMetricLabel,
        .recommendation-box, .recommendation-box * {
            color: #fff !important;
        }

        /* Sidebar: blue gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #232946 0%, #181c2f 100%);
        }
        [data-testid="stSidebar"] * {
            color: #fff !important;
        }

        /* Buttons: orange with blue hover */
        .stButton>button {
            background: #ff9800;
            color: #fff !important;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(255, 152, 0, 0.18);
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton>button:hover {
            background: #232946;
            color: #ff9800 !important;
            box-shadow: 0 4px 12px rgba(35, 41, 70, 0.28);
            transform: translateY(-2px);
        }

        /* Download Button: blue with orange hover */
        .stDownloadButton>button {
            background: #232946;
            color: #fff;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(35, 41, 70, 0.25);
        }
        .stDownloadButton>button:hover {
            background: #ff9800;
            color: #232946;
            box-shadow: 0 6px 20px rgba(255, 152, 0, 0.35);
            transform: translateY(-2px);
        }

        /* Expander: blue with orange hover */
        .streamlit-expanderHeader {
            background: #232946;
            border-radius: 10px;
            border: 1px solid #ff9800;
            font-weight: 600 !important;
            color: #fff !important;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        .streamlit-expanderHeader:hover {
            background: #ff9800;
            color: #232946 !important;
            border-color: #232946;
        }
        .streamlit-expanderContent {
            border: 1px solid #ff9800;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 1.5rem;
            background: #fff;
            color: #181c2f !important;
        }

        /* Tables: white with blue/orange stripes and hover */
        .dataframe {
            border: none !important;
            border-radius: 10px;
            overflow: hidden;
        }
        .dataframe thead tr th {
            background: #232946;
            color: #fff !important;
            font-weight: 600 !important;
            padding: 1rem;
            border: none !important;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #f5f5f5;
        }
        .dataframe tbody tr:nth-child(odd) {
            background-color: #fff;
        }
        .dataframe tbody tr:hover {
            background-color: #ffecb3;
            transition: all 0.2s ease;
        }
        .dataframe tbody tr td {
            padding: 0.75rem;
            border: none !important;
            color: #181c2f !important;
        }

        /* Recommendation box: orange border, white bg, dark text */
        .recommendation-box {
            background: #fff;
            border: 1.5px solid #ff9800;
            border-left: 6px solid #232946;
            border-radius: 10px;
            padding: 1rem;
            color: #181c2f !important;
        }
        
        /* Main Background: dark for high contrast */
        .stApp {
            background: #181c2f;
        }
        
        /* Remove duplicate Main Content Area rule */
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Title Styling */
        h1 {
            color: #ffffff !important;
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
            background: #3498db;
            color: #fff !important;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(52, 152, 219, 0.18);
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton>button:hover {
            background: #217dbb;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.28);
            transform: translateY(-2px);
        }
        
        .stButton>button:active {
            transform: translateY(0px);
        }
        
        /* Download Button Styling */
        .stDownloadButton>button {
            background: #2ecc71;
            color: #fff;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.25);
        }
        .stDownloadButton>button:hover {
            background: #27ae60;
            box-shadow: 0 6px 20px rgba(46, 204, 113, 0.35);
            transform: translateY(-2px);
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
            background: #183243;
            border: 2px dashed #cbd5e0;
            border-radius: 15px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #3182ce;
            background: #edf2f7;
        }
        /* File uploader contents and filenames */
        [data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }
        /* Make internal filename pills readable if Streamlit shows them */
        [data-testid="stFileUploader"] .uploadedFile, 
        [data-testid="stFileUploader"] [class*="uploadedFile"],
        [data-testid="stFileUploader"] [data-testid*="fileName"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            color: #1a202c !important;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        /* Our custom filename pills */
        .file-pill {
            display: inline-flex;
            align-items: center;
            max-width: 100%;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem 0.25rem 0 0;
            font-size: 0.9rem;
            color: #1a202c;
        }
        .file-pill .name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 280px;
        }
        .file-pill .icon { margin-right: 0.5rem; color: #3182ce; }
        
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
            background: #f4f6fa;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            font-weight: 600 !important;
            color: #222 !important;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        .streamlit-expanderHeader:hover {
            background: #eaf1fb;
            border-color: #3498db;
        }
        .streamlit-expanderContent {
            border: 1px solid #e2e8f0;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 1.5rem;
            background: #f8f9fa;
            color: #222 !important;
        }
        
        /* Tables */
        .dataframe {
            border: none !important;
            border-radius: 10px;
            overflow: hidden;
        }
        .dataframe thead tr th {
            background: #f4f6fa;
            color: #222 !important;
            font-weight: 600 !important;
            padding: 1rem;
            border: none !important;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .dataframe tbody tr:nth-child(odd) {
            background-color: #fff;
        }
        .dataframe tbody tr:hover {
            background-color: #eaf1fb;
            transition: all 0.2s ease;
        }
        .dataframe tbody tr td {
            padding: 0.75rem;
            border: none !important;
            color: #222 !important;
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
        
        /* Recommendation box readability (if any custom containers are used) */
        .recommendation-box {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3182ce;
            border-radius: 10px;
            padding: 1rem;
            color: #1a202c !important;
        }
    </style>""",
        unsafe_allow_html=True,
    )
