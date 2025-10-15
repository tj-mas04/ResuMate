"""
Text utility functions.
"""


def word_count(text):
    """
    Count words in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of words
    """
    return len(text.split())
