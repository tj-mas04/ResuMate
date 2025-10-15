"""
Grammar checking service.
"""
import os
import streamlit as st
import language_tool_python
from config import LANGUAGE_TOOL_CACHE_DIR


class GrammarService:
    """Service for grammar checking."""
    
    def __init__(self):
        self.tool = self._load_grammar_tool()
    
    @staticmethod
    @st.cache_resource
    def _load_grammar_tool():
        """Load LanguageTool with caching."""
        os.makedirs(LANGUAGE_TOOL_CACHE_DIR, exist_ok=True)
        os.environ["LANGUAGE_TOOL_CACHE_DIR"] = LANGUAGE_TOOL_CACHE_DIR
        return language_tool_python.LanguageTool('en-US')
    
    def check_grammar(self, text):
        """
        Check grammar in text.
        
        Args:
            text (str): Text to check
            
        Returns:
            tuple: (error_count, list_of_errors)
        """
        matches = self.tool.check(text)
        errors = [
            {
                "Error": match.message,
                "Sentence": match.context
            }
            for match in matches
        ]
        return len(matches), errors
