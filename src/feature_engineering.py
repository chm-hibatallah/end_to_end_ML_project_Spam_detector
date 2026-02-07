# src/feature_engineering.py - Feature extraction for SMS spam detection
import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class FeatureExtractor:
    """
    Feature extraction for SMS spam detection
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Spam-related keywords (can be extended)
        self.spam_keywords = {
            'free', 'win', 'winner', 'won', 'cash', 'prize', 'click', 'call', 
            'now', 'urgent', 'congratulations', 'selected', 'award', 'bonus',
            'discount', 'offer', 'deal', 'limited', 'exclusive', 'guaranteed',
            'risk', 'free', 'trial', 'subscription', 'unsubscribe', 'claim',
            'reward', 'lottery', 'draw', 'chance', 'opportunity', 'sms',
            'text', 'reply', 'stop', 'code', 'mobile', 'phone', 'number',
            'http', 'https', 'www', 'link', 'website', 'web', 'online',
            'urgent', 'important', 'alert', 'warning', 'notice', 'update'
        }
        
        # Common ham (non-spam) keywords
        self.ham_keywords = {
            'ok', 'yes', 'no', 'thanks', 'thank', 'please', 'sorry',
            'hi', 'hello', 'hey', 'dear', 'meeting', 'tomorrow',
            'today', 'yesterday', 'morning', 'evening', 'night',
            'weekend', 'home', 'work', 'office', 'school', 'college',
            'family', 'friend', 'mom', 'dad', 'brother', 'sister',
            'love', 'miss', 'see', 'meet', 'call', 'talk', 'speak'
        }
        
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """
        Extract various text-based features from a message
        
        Args:
            text (str): Input text message
            
        Returns:
            Dict[str, float]: Dictionary of feature names and values
        """
        if not isinstance(text, str):
            text = str(text)
        
        features = {}
        
        # Basic length features
        features['char_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character type features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(1, len(text))
        features['special_char_ratio'] = sum(1 for c in text if c in string.punctuation) / max(1, len(text))
        
        # Word-based features
        words = text.lower().split()
        features['unique_word_ratio'] = len(set(words)) / max(1, len(words))
        
        # Spam indicator features
        spam_word_count = sum(1 for word in words if word in self.spam_keywords)
        ham_word_count = sum(1 for word in words if word in self.ham_keywords)
        
        features['spam_keyword_ratio'] = spam_word_count / max(1, len(words))
        features['ham_keyword_ratio'] = ham_word_count / max(1, len(words))
        features['spam_ham_ratio'] = spam_word_count / max(1, ham_word_count)
        
        # URL and phone number features
        features['has_url'] = float(bool(re.search(r'http[s]?://|www\.', text, re.IGNORECASE)))
        features['has_phone'] = float(bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)))
        features['has_email'] = float(bool(re.search(r'\S+@\S+', text)))
        
        # Exclamation and question marks
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['exclamation_ratio'] = features['exclamation_count'] / max(1, len(text))
        features['question_ratio'] = features['question_count'] / max(1, len(text))
        
        # Money-related patterns
        features['has_currency'] = float(bool(re.search(r'[$£€¥]?\d+[.,]?\d*[$£€¥]?', text)))
        
        # Time urgency indicators
        urgency_words = ['urgent', 'immediately', 'now', 'today', 'asap', 'quick', 'fast', 'hurry']
        features['urgency_score'] = sum(text.lower().count(word) for word in urgency_words) / max(1, len(words))
        
        # Readability score (simplified)
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = len(words) / max(1, features['sentence_count'])
        
        # Entropy of characters (measure of randomness)
        if text:
            char_counts = {}
            for char in text.lower():
                char_counts[char] = char_counts.get(char, 0) + 1
            entropy = -sum((count/len(text)) * np.log2(count/len(text)) 
                          for count in char_counts.values())
            features['char_entropy'] = entropy
        else:
            features['char_entropy'] = 0
            
        return features
    
    def extract_ngram_features(self, texts: List[str], vectorizer_type: str = 'tfidf', 
                              ngram_range: Tuple[int, int] = (1, 2), 
                              max_features: int = 1000) -> np.ndarray:
        """
        Extract n-gram features from texts
        
        Args:
            texts (List[str]): List of text messages
            vectorizer_type (str): 'tfidf' or 'count'
            ngram_range (tuple): Range of n-gram sizes
            max_features (int): Maximum number of features
            
        Returns:
            np.ndarray: Feature matrix
        """
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words='english',
                lowercase=True,
                max_df=0.95,
                min_df=2
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words='english',
                lowercase=True,
                max_df=0.95,
                min_df=2
            )
        
        return vectorizer.fit_transform(texts), vectorizer
    
    def extract_all_features(self, texts: List[str], 
                           include_text_features: bool = True,
                           include_ngrams: bool = True,
                           ngram_type: str = 'tfidf',
                           ngram_max_features: int = 500,
                           reduce_dimensions: bool = False,
                           n_components: int = 50) -> Tuple[np.ndarray, Dict]:
        """
        Extract all features from texts
        
        Args:
            texts (List[str]): List of text messages
            include_text_features (bool): Whether to include handcrafted features
            include_ngrams (bool): Whether to include n-gram features
            ngram_type (str): Type of vectorizer ('tfidf' or 'count')
            ngram_max_features (int): Maximum n-gram features
            reduce_dimensions (bool): Whether to reduce dimensionality
            n_components (int): Number of components for dimensionality reduction
            
        Returns:
            Tuple[np.ndarray, Dict]: Feature matrix and vectorizer dictionary
        """
        feature_parts = []
        vectorizers = {}
        
        # Extract handcrafted text features
        if include_text_features:
            text_features = []
            for text in texts:
                features = self.extract_text_features(text)
                text_features.append(list(features.values()))
            
            text_features_array = np.array(text_features)
            feature_parts.append(text_features_array)
            vectorizers['text_features'] = list(self.extract_text_features('').keys())
        
        # Extract n-gram features
        if include_ngrams:
            ngram_features, ngram_vectorizer = self.extract_ngram_features(
                texts, vectorizer_type=ngram_type, max_features=ngram_max_features
            )
            feature_parts.append(ngram_features.toarray())
            vectorizers['ngram_vectorizer'] = ngram_vectorizer
        
        # Combine all features
        if feature_parts:
            features = np.hstack(feature_parts)
        else:
            features = np.zeros((len(texts), 0))
        
        # Dimensionality reduction if requested
        if reduce_dimensions and features.shape[1] > n_components:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            features = svd.fit_transform(features)
            vectorizers['svd'] = svd
        
        return features, vectorizers
    
    def get_feature_names(self, vectorizers: Dict) -> List[str]:
        """
        Get names of all features
        
        Args:
            vectorizers (Dict): Dictionary of vectorizers
            
        Returns:
            List[str]: List of feature names
        """
        feature_names = []
        
        # Add handcrafted feature names
        if 'text_features' in vectorizers:
            feature_names.extend(vectorizers['text_features'])
        
        # Add n-gram feature names
        if 'ngram_vectorizer' in vectorizers:
            ngram_vectorizer = vectorizers['ngram_vectorizer']
            feature_names.extend(ngram_vectorizer.get_feature_names_out())
        
        return feature_names


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering with more sophisticated techniques
    """
    def __init__(self):
        self.extractor = FeatureExtractor()
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features
        
        Args:
            df (pd.DataFrame): DataFrame with existing features
            
        Returns:
            pd.DataFrame: DataFrame with interaction features added
        """
        # Example interaction features
        if 'spam_keyword_ratio' in df.columns and 'uppercase_ratio' in df.columns:
            df['spam_uppercase_interaction'] = df['spam_keyword_ratio'] * df['uppercase_ratio']
        
        if 'has_url' in df.columns and 'has_phone' in df.columns:
            df['url_phone_interaction'] = df['has_url'] * df['has_phone']
        
        if 'exclamation_ratio' in df.columns and 'urgency_score' in df.columns:
            df['urgency_exclamation_interaction'] = df['exclamation_ratio'] * df['urgency_score']
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df (pd.DataFrame): DataFrame with features
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: DataFrame with polynomial features
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            for d in range(2, degree + 1):
                df[f'{col}_pow{d}'] = df[col] ** d
        
        return df
    
    def extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract semantic features using word embeddings (simplified version)
        
        Args:
            texts (List[str]): List of text messages
            
        Returns:
            np.ndarray: Semantic feature matrix
        """
        # This is a simplified version. In production, you might use:
        # 1. Word2Vec/GloVe embeddings
        # 2. BERT embeddings
        # 3. Sentence transformers
        
        # For now, we'll create simplified semantic features
        features = []
        
        for text in texts:
            text_features = {}
            
            # Sentiment indicators (simplified)
            positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'thanks', 'thank']
            negative_words = ['bad', 'poor', 'terrible', 'angry', 'hate', 'sorry', 'apologize']
            
            words = text.lower().split()
            text_features['positive_score'] = sum(word in positive_words for word in words)
            text_features['negative_score'] = sum(word in negative_words for word in words)
            text_features['sentiment_balance'] = text_features['positive_score'] - text_features['negative_score']
            
            # Formality indicators
            formal_words = ['sir', 'madam', 'please', 'kindly', 'regards', 'sincerely']
            informal_words = ['hey', 'yo', 'lol', 'haha', 'omg', 'wtf', 'brb']
            
            text_features['formality_score'] = sum(word in formal_words for word in words)
            text_features['informality_score'] = sum(word in informal_words for word in words)
            
            features.append(list(text_features.values()))
        
        return np.array(features)


# Utility functions
def prepare_feature_matrix(df: pd.DataFrame, text_column: str = 'message', 
                          label_column: str = 'label', 
                          feature_type: str = 'all') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepare feature matrix from DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of text column
        label_column (str): Name of label column
        feature_type (str): Type of features to extract ('text', 'ngram', 'all')
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: Features, labels, and vectorizers
    """
    extractor = FeatureExtractor()
    
    texts = df[text_column].fillna('').astype(str).tolist()
    labels = df[label_column].values if label_column in df.columns else None
    
    if feature_type == 'text':
        features, vectorizers = extractor.extract_all_features(
            texts, include_text_features=True, include_ngrams=False
        )
    elif feature_type == 'ngram':
        features, vectorizers = extractor.extract_all_features(
            texts, include_text_features=False, include_ngrams=True
        )
    else:  # 'all'
        features, vectorizers = extractor.extract_all_features(
            texts, include_text_features=True, include_ngrams=True
        )
    
    return features, labels, vectorizers


def save_vectorizers(vectorizers: Dict, path: str):
    """
    Save vectorizers to disk
    
    Args:
        vectorizers (Dict): Dictionary of vectorizers
        path (str): Path to save to
    """
    import joblib
    joblib.dump(vectorizers, path)
    print(f"Vectorizers saved to {path}")


def load_vectorizers(path: str) -> Dict:
    """
    Load vectorizers from disk
    
    Args:
        path (str): Path to load from
        
    Returns:
        Dict: Loaded vectorizers
    """
    import joblib
    vectorizers = joblib.load(path)
    print(f"Vectorizers loaded from {path}")
    return vectorizers


# Example usage
if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "WINNER!! You've won a free iPhone! Call now to claim.",
        "Hi John, are we meeting tomorrow for lunch?",
        "URGENT: Your bank account has been compromised.",
        "Don't forget to bring the documents to the meeting."
    ]
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features for a single text
    text = sample_texts[0]
    features = extractor.extract_text_features(text)
    print("Text features for sample message:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Extract all features for multiple texts
    print("\nExtracting all features for sample texts...")
    all_features, vectorizers = extractor.extract_all_features(
        sample_texts, 
        include_text_features=True,
        include_ngrams=True,
        ngram_max_features=100
    )
    
    print(f"Feature matrix shape: {all_features.shape}")
    print(f"Number of text features: {len(vectorizers.get('text_features', []))}")
    print(f"Number of n-gram features: {len(vectorizers.get('ngram_vectorizer', []).get_feature_names_out())}")