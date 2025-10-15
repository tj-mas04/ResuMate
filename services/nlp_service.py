"""
NLP service for text analysis, embeddings, and NER.
"""
import streamlit as st
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from config import SPACY_MODEL, SENTENCE_TRANSFORMER_MODEL, NER_SKILL_LABELS


class NLPService:
    """Service for NLP operations."""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.sbert_model = self._load_sentence_transformer()
    
    @staticmethod
    @st.cache_resource
    def _load_spacy_model():
        """Load spaCy model with caching."""
        return spacy.load(SPACY_MODEL)
    
    @staticmethod
    @st.cache_resource
    def _load_sentence_transformer():
        """Load sentence transformer model with caching."""
        return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    
    def extract_skills_ner(self, text):
        """
        Extract skills using Named Entity Recognition.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of extracted skills
        """
        doc = self.nlp(text)
        skills = set()
        for ent in doc.ents:
            if ent.label_ in NER_SKILL_LABELS:
                skills.add(ent.text.strip())
        return list(skills)
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract keywords using TF-IDF.
        
        Args:
            text (str): Input text
            top_n (int): Number of keywords to extract
            
        Returns:
            list: Top keywords
        """
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=top_n
            )
            vectorizer.fit_transform([text])
            return vectorizer.get_feature_names_out()
        except Exception:
            return []
    
    def compute_similarity(self, text1, text2):
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-100)
        """
        embeddings = self.sbert_model.encode(
            [text1, text2],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # Cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10
        )
        score = float(cosine_sim * 100.0)
        
        # Clamp between 0 and 100
        return max(0.0, min(100.0, score))
    
    def count_action_verbs(self, text):
        """
        Count and extract verbs using POS tagging.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (verb_count, list_of_verbs)
        """
        doc = self.nlp(text)
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        return len(verbs), verbs
