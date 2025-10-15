"""
Database connection management.
"""
import sqlite3
from config import DATABASE_NAME

_connection = None


def get_connection():
    """Get or create database connection."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
    return _connection


def init_database():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    """)
    
    # History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            resume_name TEXT NOT NULL,
            similarity REAL NOT NULL,
            ats_score REAL NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    
    conn.commit()
    return conn


def close_connection():
    """Close database connection."""
    global _connection
    if _connection:
        _connection.close()
        _connection = None
