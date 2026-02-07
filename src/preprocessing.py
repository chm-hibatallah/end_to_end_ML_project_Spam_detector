# src/preprocessing.py - Basic text preprocessing
import re

def clean_text(text):
    """
    Basic text cleaning
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Alias for compatibility
preprocessor = None
quick_clean = clean_text