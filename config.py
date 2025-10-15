"""
Configuration settings for ResuMate application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Database Configuration
DATABASE_NAME = "resumate.db"

# Model Configuration
SPACY_MODEL = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_RECOMMENDATION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# LanguageTool Configuration
LANGUAGE_TOOL_CACHE_DIR = r"C:\Users\ASUS\Documents\Resumate\Dev\LanguageTool"

# ATS Scoring Weights
ATS_WEIGHTS = {
    "similarity": 0.4,
    "skill_match": 0.2,
    "section_completeness": 0.2,
    "grammar": 0.2
}

# Resume Sections to Check
RESUME_SECTIONS = ["Education", "Experience", "Skills", "Projects", "Certifications"]

# NER Labels for Skill Extraction
NER_SKILL_LABELS = [
    "ORG", "PRODUCT", "WORK_OF_ART", "EDUCATION",
    "GPE", "NORP", "PERSON", "LANGUAGE"
]

# UI Configuration
PAGE_TITLE = "ResuMate"
PAGE_ICON = "📄"
LAYOUT = "wide"

# Keyword Extraction
DEFAULT_TOP_KEYWORDS = 10
ATS_TOP_KEYWORDS = 20
