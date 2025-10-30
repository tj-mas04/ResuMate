from __future__ import annotations
import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


nlp = load_spacy_model()
sbert_model = load_sentence_model()


def extract_skills_ner(text: str) -> list[str]:
    """Use spaCy NER to extract potential skills/entities."""
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in [
            "ORG",
            "PRODUCT",
            "WORK_OF_ART",
            "EDUCATION",
            "GPE",
            "NORP",
            "PERSON",
            "LANGUAGE",
        ]:
            skills.add(ent.text.strip())
    return list(skills)


def count_action_verbs(text: str) -> tuple[int, list[str]]:
    doc = nlp(text)
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    return len(verbs), verbs
