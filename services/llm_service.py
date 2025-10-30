from __future__ import annotations
from typing import Any
from config import get_groq_client


def generate_recommendation(details: dict) -> str:
    client = get_groq_client()
    if not client:
        return "(Groq API not configured) Set GROQ_API_KEY in your .env to enable AI feedback."

    prompt = f"""
    You are an expert resume reviewer for ATS systems.

    Analyze the following resume evaluation details and give a short, actionable paragraph
    (3-5 sentences) suggesting how to improve the resume to better match the job description.

    Resume Details:
    - Similarity Score: {details.get('similarity', 0)}%
    - ATS Score: {details.get('ats_score', 0)}
    - Missing Keywords: {', '.join(details.get('missing_keywords', []))}
    - Missing Skills: {', '.join(details.get('missing_skills', []))}
    - Grammar Errors: {details.get('grammar_errors', 0)}
    - Action Verbs Used: {details.get('action_verbs_count', 0)}

    Give clear and specific feedback, e.g.:
    “Add more mentions of cloud frameworks and deployment tools.”
    Avoid bullet points, give one cohesive paragraph.
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Groq API Error) Could not generate feedback: {e}"


def chat_reply(messages: list[dict[str, str]]) -> str:
    client = get_groq_client()
    if not client:
        return "(Groq API not configured) Set GROQ_API_KEY in your .env to chat."
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Groq API error) {e}"
