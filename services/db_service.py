from __future__ import annotations
import sqlite3
import hashlib
from typing import Any
from config import DB_PATH

# Create connection globally with check_same_thread=False for Streamlit
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Init tables
c.execute("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password_hash TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, username TEXT, timestamp TEXT, resume_name TEXT, similarity REAL, ats_score REAL)""")
conn.commit()


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register(username: str, password: str) -> bool:
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, _hash_pw(password)))
        conn.commit()
        return True
    except Exception:
        return False


def authenticate(username: str, password: str) -> bool:
    c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    return bool(row and row[0] == _hash_pw(password))


def insert_history(username: str, timestamp: str, resume_name: str, similarity: float, ats_score: float) -> None:
    c.execute(
        "INSERT INTO history(username, timestamp, resume_name, similarity, ats_score) VALUES (?, ?, ?, ?, ?)",
        (username, timestamp, resume_name, similarity, ats_score),
    )
    conn.commit()


def get_history_for_user(username: str):
    return c.execute(
        "SELECT timestamp, resume_name, similarity, ats_score FROM history WHERE username=? ORDER BY timestamp DESC",
        (username,),
    )
