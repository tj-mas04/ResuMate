"""
Database models for User and History.
"""
import hashlib
import datetime
import sqlite3
from .connection import get_connection


class User:
    """User model for authentication."""
    
    @staticmethod
    def hash_password(password):
        """Hash a password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def register(username, password):
        """Register a new user."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            password_hash = User.hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    @staticmethod
    def authenticate(username, password):
        """Authenticate a user."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password_hash FROM users WHERE username=?",
            (username,)
        )
        row = cursor.fetchone()
        if row:
            return row[0] == User.hash_password(password)
        return False


class History:
    """History model for storing evaluation records."""
    
    @staticmethod
    def add_record(username, resume_name, similarity, ats_score):
        """Add a new evaluation record."""
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            """INSERT INTO history 
               (username, timestamp, resume_name, similarity, ats_score)
               VALUES (?, ?, ?, ?, ?)""",
            (username, timestamp, resume_name, similarity, ats_score)
        )
        conn.commit()
    
    @staticmethod
    def get_user_history(username):
        """Get all evaluation history for a user."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, resume_name, similarity, ats_score 
               FROM history 
               WHERE username=? 
               ORDER BY timestamp DESC""",
            (username,)
        )
        return cursor.fetchall()
