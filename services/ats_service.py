"""
ATS scoring service.
"""
import re
from config import ATS_WEIGHTS, RESUME_SECTIONS


class ATSService:
    """Service for ATS scoring and resume analysis."""
    
    def __init__(self, nlp_service, grammar_service):
        """
        Initialize ATS service.
        
        Args:
            nlp_service: Instance of NLPService
            grammar_service: Instance of GrammarService
        """
        self.nlp = nlp_service
        self.grammar = grammar_service
    
    def compute_ats_score(self, resume_text, jd_text, jd_keywords=None):
        """
        Compute comprehensive ATS score.
        
        Args:
            resume_text (str): Resume text
            jd_text (str): Job description text
            jd_keywords (list, optional): Pre-extracted JD keywords
            
        Returns:
            tuple: (ats_score, matched_skills, missing_skills)
        """
        if jd_keywords is None:
            jd_keywords = self.nlp.extract_keywords(jd_text, top_n=20)
        
        # Semantic similarity
        sim_score = self.nlp.compute_similarity(jd_text, resume_text)
        
        # Keyword match
        resume_keywords = self.nlp.extract_keywords(resume_text, top_n=20)
        matched_kw = len(set(jd_keywords) & set(resume_keywords))
        
        # NER-based skills
        resume_skills = self.nlp.extract_skills_ner(resume_text)
        jd_skills = self.nlp.extract_skills_ner(jd_text)
        matched_skills = set(jd_skills) & set(resume_skills)
        missing_skills = set(jd_skills) - set(resume_skills)
        skill_match_score = (len(matched_skills) / max(1, len(jd_skills))) * 100
        
        # Section completeness
        sections = self.check_sections(resume_text)
        section_score = (sum(sections.values()) / len(sections)) * 100
        
        # Grammar score
        grammar_count, _ = self.grammar.check_grammar(resume_text)
        word_cnt = self._word_count(resume_text)
        max_errors = max(1, word_cnt // 50)
        grammar_score = max(0, 100 - min(grammar_count / max_errors * 100, 100))
        
        # Weighted ATS score
        ats_score = (
            ATS_WEIGHTS["similarity"] * sim_score +
            ATS_WEIGHTS["skill_match"] * skill_match_score +
            ATS_WEIGHTS["section_completeness"] * section_score +
            ATS_WEIGHTS["grammar"] * grammar_score
        )
        
        return round(ats_score, 2), list(matched_skills), list(missing_skills)
    
    @staticmethod
    def check_sections(text):
        """
        Check which resume sections are present.
        
        Args:
            text (str): Resume text
            
        Returns:
            dict: Section name -> bool (present or not)
        """
        return {
            section: bool(re.search(section, text, re.I))
            for section in RESUME_SECTIONS
        }
    
    @staticmethod
    def _word_count(text):
        """Count words in text."""
        return len(text.split())
    
    def get_missing_keywords(self, jd_keywords, resume_keywords):
        """
        Get keywords missing from resume.
        
        Args:
            jd_keywords (list): Job description keywords
            resume_keywords (list): Resume keywords
            
        Returns:
            list: Missing keywords
        """
        return list(set(jd_keywords) - set(resume_keywords))
