"""
Database package initialization.
"""
from .connection import get_connection, init_database
from .models import User, History

__all__ = ['get_connection', 'init_database', 'User', 'History']
