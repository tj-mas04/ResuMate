from __future__ import annotations
import re
from math import pi
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from services.nlp_service import sbert_model, extract_skills_ner
from services.grammar_service import check_grammar


def extract_text(file) -> str:
    reader = PyPDF2.PdfReader(file)
    return ("".join(p.extract_text() or "" for p in reader.pages).strip() or "No readable text.")


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    vectorizer.fit_transform([text])
    return list(vectorizer.get_feature_names_out())


def compute_similarity(a: str, b: str) -> float:
    emb = sbert_model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    a_emb, b_emb = emb[0], emb[1]
    cos = float(np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-10))
    score = cos * 100.0
    return max(0.0, min(100.0, score))


def sections_found(text: str) -> dict[str, bool]:
    return {s: bool(re.search(s, text, re.I)) for s in ["Education", "Experience", "Skills", "Projects", "Certifications"]}


def word_count(text: str) -> int:
    return len(text.split())


def compute_ats(resume_text: str, jd_text: str, jd_keywords: list[str] | None = None):
    if jd_keywords is None:
        jd_keywords = extract_keywords(jd_text, top_n=20)

    sim_score = compute_similarity(jd_text, resume_text)

    res_keywords = extract_keywords(resume_text, top_n=20)
    matched_kw = len(set(jd_keywords) & set(res_keywords))

    resume_skills = extract_skills_ner(resume_text)
    jd_skills = extract_skills_ner(jd_text)
    matched_skills = set(jd_skills) & set(resume_skills)
    missing_skills = set(jd_skills) - set(resume_skills)
    skill_match_score = (len(matched_skills) / max(1, len(jd_skills))) * 100

    sections = sections_found(resume_text)
    section_score = (sum(sections.values()) / len(sections)) * 100

    grammar_count, _ = check_grammar(resume_text)
    max_errors = max(1, word_count(resume_text) // 50)
    grammar_score = max(0, 100 - min(grammar_count / max_errors * 100, 100))

    ats = 0.4 * sim_score + 0.2 * skill_match_score + 0.2 * section_score + 0.2 * grammar_score

    return round(ats, 2), list(matched_skills), list(missing_skills)
