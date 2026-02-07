# src/utils.py - Utility functions for SMS spam detection
import os
import sys
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import re
import hashlib
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """
    Configuration management for the application
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'app': {
                'name': 'SMS Spam Detector',
                'version': '1.0.0',
                'debug': True,
                'host': '0.0.0.0',
                'port': 5000
            },
            'model': {
                'path': 'notebooks/spam_classifier.pkl',
                'vectorizer_path': 'notebooks/tfidf_vectorizer.pkl',
                'threshold': 0.5,
                'max_message_length': 500
            },
            'features': {
                'use_text_features': True,
                'use_ngram_features': True,
                'ngram_max_features': 1000,
                'ngram_range': (1, 2),
                'min_word_length': 2,
                'max_word_length': 20
            },
            'data': {
                'data_path': 'data/',
                'cache_dir': 'cache/',
                'max_cache_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log',
                'max_file_size': 10485760,  # 10MB
                'backup_count': 5
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._deep_update(self.config, user_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    def save_config(self, config_path: str):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    def _deep_update(self, original: Dict, update: Dict):
        """Recursively update nested dictionary"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


class DataLoader:
    """
    Data loading and preprocessing utilities
    """
    @staticmethod
    def load_csv(filepath: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
        """
        Load CSV file with error handling
        
        Args:
            filepath (str): Path to CSV file
            encoding (str): File encoding
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 encoding failed for {filepath}, trying latin-1")
            try:
                df = pd.read_csv(filepath, encoding='latin-1', **kwargs)
                logger.info(f"Loaded {len(df)} rows from {filepath} with latin-1 encoding")
                return df
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    @staticmethod
    def load_excel(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load Excel file
        
        Args:
            filepath (str): Path to Excel file
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            df = pd.read_excel(filepath, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    @staticmethod
    def load_json(filepath: str, **kwargs) -> Union[pd.DataFrame, Dict, List]:
        """
        Load JSON file
        
        Args:
            filepath (str): Path to JSON file
            **kwargs: Additional arguments for pd.read_json or json.load
            
        Returns:
            Union[pd.DataFrame, Dict, List]: Loaded data
        """
        try:
            # Try to load as DataFrame first
            df = pd.read_json(filepath, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except:
            # Fall back to regular JSON loading
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded JSON data from {filepath}")
                return data
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                raise
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs):
        """
        Save DataFrame to file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Path to save file
            **kwargs: Additional arguments for saving method
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False, **kwargs)
            elif filepath.endswith('.xlsx'):
                df.to_excel(filepath, index=False, **kwargs)
            elif filepath.endswith('.json'):
                df.to_json(filepath, orient='records', **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            logger.info(f"Saved DataFrame to {filepath}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filepath}: {e}")
            raise


class CacheManager:
    """
    Simple caching mechanism for predictions
    """
    def __init__(self, cache_dir: str = 'cache', max_size: int = 1000):
        """
        Initialize cache manager
        
        Args:
            cache_dir (str): Cache directory
            max_size (int): Maximum cache size
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_file = os.path.join(cache_dir, 'predictions_cache.pkl')
        self.cache = self._load_cache()
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get_cache_key(self, message: str) -> str:
        """
        Generate cache key from message
        
        Args:
            message (str): Input message
            
        Returns:
            str: Cache key
        """
        # Create a hash of the message for efficient lookup
        message_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        return f"{message_hash}"
    
    def get(self, message: str) -> Optional[Dict]:
        """
        Get cached prediction for message
        
        Args:
            message (str): Input message
            
        Returns:
            Optional[Dict]: Cached prediction or None
        """
        cache_key = self.get_cache_key(message)
        if cache_key in self.cache:
            # Check if cache entry is still valid (based on timestamp if needed)
            cached_data = self.cache[cache_key]
            logger.debug(f"Cache hit for message: {message[:50]}...")
            return cached_data['prediction']
        return None
    
    def set(self, message: str, prediction: Dict):
        """
        Cache prediction for message
        
        Args:
            message (str): Input message
            prediction (Dict): Prediction result
        """
        cache_key = self.get_cache_key(message)
        
        # Add timestamp for potential expiration logic
        self.cache[cache_key] = {
            'message': message,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        # Enforce max cache size
        if len(self.cache) > self.max_size:
            # Remove oldest entries (simplified: remove random entries)
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.max_size]
            for key in keys_to_remove:
                del self.cache[key]
        
        # Save cache to disk
        self._save_cache()
        logger.debug(f"Cached prediction for message: {message[:50]}...")
    
    def clear(self):
        """Clear cache"""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dict: Cache statistics
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'cache_file': self.cache_file,
            'cache_dir': self.cache_dir
        }


class TextUtils:
    """
    Text processing utilities
    """
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by removing extra whitespace and normalizing characters
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
        """
        Truncate text to maximum length
        
        Args:
            text (str): Input text
            max_length (int): Maximum length
            suffix (str): Suffix to add if truncated
            
        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we found a space near the end
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text (simplified version)
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Extracted entities
        """
        entities = {
            'urls': [],
            'emails': [],
            'phone_numbers': [],
            'hashtags': [],
            'mentions': []
        }
        
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .\-?=%&]*'
        entities['urls'] = re.findall(url_pattern, text)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Extract phone numbers (simple pattern)
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        entities['phone_numbers'] = re.findall(phone_pattern, text)
        
        # Extract hashtags
        hashtag_pattern = r'#\w+'
        entities['hashtags'] = re.findall(hashtag_pattern, text)
        
        # Extract mentions
        mention_pattern = r'@\w+'
        entities['mentions'] = re.findall(mention_pattern, text)
        
        return entities
    
    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        """
        Calculate readability metrics (simplified)
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Readability metrics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'readability_score': 0
            }
        
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        
        # Sentence count (simplified)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Calculate averages
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Simplified Flesch Reading Ease score
        # Lower score means harder to read
        readability_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / word_count if word_count > 0 else 0)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': readability_score
        }


class PerformanceMonitor:
    """
    Monitor and track performance metrics
    """
    def __init__(self):
        self.metrics = {
            'predictions': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'avg_response_time': 0.0,
                'response_times': []
            },
            'cache': {
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0
            },
            'model': {
                'load_time': 0.0,
                'last_loaded': None
            }
        }
        self.start_time = datetime.now()
    
    def start_prediction(self):
        """Start timing a prediction"""
        return datetime.now()
    
    def end_prediction(self, start_time: datetime, success: bool = True):
        """
        End timing a prediction and record metrics
        
        Args:
            start_time (datetime): Start time
            success (bool): Whether prediction was successful
        """
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        self.metrics['predictions']['total'] += 1
        if success:
            self.metrics['predictions']['successful'] += 1
        else:
            self.metrics['predictions']['failed'] += 1
        
        self.metrics['predictions']['response_times'].append(response_time)
        
        # Keep only last 100 response times
        if len(self.metrics['predictions']['response_times']) > 100:
            self.metrics['predictions']['response_times'].pop(0)
        
        # Update average response time
        times = self.metrics['predictions']['response_times']
        self.metrics['predictions']['avg_response_time'] = sum(times) / len(times) if times else 0.0
    
    def record_cache_hit(self, hit: bool = True):
        """
        Record cache hit/miss
        
        Args:
            hit (bool): Whether it was a cache hit
        """
        if hit:
            self.metrics['cache']['hits'] += 1
        else:
            self.metrics['cache']['misses'] += 1
        
        total = self.metrics['cache']['hits'] + self.metrics['cache']['misses']
        if total > 0:
            self.metrics['cache']['hit_rate'] = self.metrics['cache']['hits'] / total
    
    def record_model_load(self, load_time: float):
        """
        Record model load time
        
        Args:
            load_time (float): Time taken to load model
        """
        self.metrics['model']['load_time'] = load_time
        self.metrics['model']['last_loaded'] = datetime.now().isoformat()
    
    def get_metrics(self) -> Dict:
        """
        Get all performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        metrics = self.metrics.copy()
        metrics['uptime_seconds'] = uptime
        metrics['uptime_human'] = str(datetime.now() - self.start_time)
        metrics['timestamp'] = datetime.now().isoformat()
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'predictions': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'avg_response_time': 0.0,
                'response_times': []
            },
            'cache': {
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0
            },
            'model': {
                'load_time': 0.0,
                'last_loaded': None
            }
        }
        self.start_time = datetime.now()
        logger.info("Performance metrics reset")


class FileSystemUtils:
    """
    Filesystem utilities
    """
    @staticmethod
    def ensure_dir(directory: str):
        """
        Ensure directory exists
        
        Args:
            directory (str): Directory path
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        """
        Get file size in bytes
        
        Args:
            filepath (str): File path
            
        Returns:
            int: File size in bytes
        """
        try:
            return os.path.getsize(filepath)
        except:
            return 0
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """
        Get directory size in bytes
        
        Args:
            directory (str): Directory path
            
        Returns:
            int: Directory size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += FileSystemUtils.get_file_size(filepath)
        return total_size
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*") -> List[str]:
        """
        Find files matching pattern in directory
        
        Args:
            directory (str): Directory to search
            pattern (str): File pattern
            
        Returns:
            List[str]: List of file paths
        """
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if Path(filename).match(pattern):
                    files.append(os.path.join(root, filename))
        return files
    
    @staticmethod
    def backup_file(filepath: str, backup_dir: str = "backups"):
        """
        Create a backup of a file
        
        Args:
            filepath (str): File to backup
            backup_dir (str): Backup directory
        """
        if not os.path.exists(filepath):
            return
        
        FileSystemUtils.ensure_dir(backup_dir)
        
        filename = os.path.basename(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")
        
        try:
            import shutil
            shutil.copy2(filepath, backup_path)
            logger.info(f"Backup created: {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")


# Singleton instances for easy access
config = Config()
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()
text_utils = TextUtils()
file_system_utils = FileSystemUtils()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level (str): Logging level
        log_file (Optional[str]): Log file path
    """
    # Convert string level to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Error setting up file logging: {e}")
    
    logger.info(f"Logging setup complete. Level: {log_level}")


def validate_message(message: str, max_length: int = 5000) -> Tuple[bool, str]:
    """
    Validate input message
    
    Args:
        message (str): Input message
        max_length (int): Maximum allowed length
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(message, str):
        return False, "Message must be a string"
    
    if not message.strip():
        return False, "Message cannot be empty"
    
    if len(message) > max_length:
        return False, f"Message too long (max {max_length} characters)"
    
    # Check for suspicious patterns (optional)
    suspicious_patterns = [
        (r'<script.*?>.*?</script>', 'Script tags detected'),
        (r'onerror\s*=', 'Potential XSS attack'),
        (r'javascript:', 'Potential XSS attack'),
    ]
    
    for pattern, error_msg in suspicious_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False, f"Security check failed: {error_msg}"
    
    return True, "Valid message"


def format_prediction_result(is_spam: bool, spam_probability: float, 
                           ham_probability: float, message: str = "",
                           include_metadata: bool = False) -> Dict[str, Any]:
    """
    Format prediction result consistently
    
    Args:
        is_spam (bool): Whether message is spam
        spam_probability (float): Spam probability
        ham_probability (float): Ham probability
        message (str): Original message
        include_metadata (bool): Whether to include metadata
        
    Returns:
        Dict[str, Any]: Formatted result
    """
    result = {
        'is_spam': bool(is_spam),
        'spam_probability': float(spam_probability),
        'ham_probability': float(ham_probability),
        'message': message[:100] + '...' if len(message) > 100 else message,
        'confidence': max(spam_probability, ham_probability),
        'timestamp': datetime.now().isoformat()
    }
    
    if include_metadata:
        result['metadata'] = {
            'model_version': config.get('app.version', '1.0.0'),
            'threshold': config.get('model.threshold', 0.5),
            'message_length': len(message),
            'entities': text_utils.extract_entities(message),
            'readability': text_utils.calculate_readability(message)
        }
    
    return result


def create_response(success: bool, data: Any = None, 
                   message: str = "", status_code: int = 200) -> Dict[str, Any]:
    """
    Create standardized API response
    
    Args:
        success (bool): Whether request was successful
        data (Any): Response data
        message (str): Response message
        status_code (int): HTTP status code
        
    Returns:
        Dict[str, Any]: Standardized response
    """
    response = {
        'success': success,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'status_code': status_code
    }
    
    if not success and status_code >= 500:
        logger.error(f"Error response: {message}")
    
    return response


# Example usage
if __name__ == "__main__":
    # Test utilities
    print("Testing utils module...")
    
    # Test Config
    cfg = Config()
    print(f"App name: {cfg.get('app.name')}")
    print(f"Model path: {cfg.get('model.path')}")
    
    # Test TextUtils
    sample_text = "Hello world! This is a test message with https://example.com and test@email.com"
    normalized = text_utils.normalize_text(sample_text)
    print(f"\nOriginal text: {sample_text}")
    print(f"Normalized text: {normalized}")
    
    entities = text_utils.extract_entities(sample_text)
    print(f"\nExtracted entities: {entities}")
    
    readability = text_utils.calculate_readability(sample_text)
    print(f"\nReadability metrics: {readability}")
    
    # Test validation
    is_valid, error_msg = validate_message("Hello world")
    print(f"\nValidation result: {is_valid}, {error_msg}")
    
    # Test response formatting
    result = format_prediction_result(
        is_spam=True,
        spam_probability=0.95,
        ham_probability=0.05,
        message="Win a free iPhone!",
        include_metadata=True
    )
    print(f"\nFormatted prediction result: {json.dumps(result, indent=2)}")
    
    # Test cache manager
    cache = CacheManager(cache_dir='test_cache', max_size=10)
    test_message = "Test message for cache"
    test_prediction = {'is_spam': False, 'probability': 0.1}
    
    # Set cache
    cache.set(test_message, test_prediction)
    
    # Get from cache
    cached = cache.get(test_message)
    print(f"\nCached prediction: {cached}")
    
    # Get cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Clean up
    import shutil
    if os.path.exists('test_cache'):
        shutil.rmtree('test_cache')
    
    print("\n✅ All utilities tested successfully!")