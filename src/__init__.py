# src/__init__.py - Package initialization
from .preprocessing import preprocessor, quick_clean
from .feature_engineering import FeatureExtractor, AdvancedFeatureEngineer, prepare_feature_matrix
from .models import SpamClassifier, train_test_evaluate
from .utils import (
    Config,
    DataLoader,
    CacheManager,
    TextUtils,
    PerformanceMonitor,
    FileSystemUtils,
    config,
    cache_manager,
    performance_monitor,
    text_utils,
    file_system_utils,
    setup_logging,
    validate_message,
    format_prediction_result,
    create_response
)

__all__ = [
    'preprocessor',
    'quick_clean',
    'FeatureExtractor',
    'AdvancedFeatureEngineer',
    'prepare_feature_matrix',
    'SpamClassifier',
    'train_test_evaluate',
    'Config',
    'DataLoader',
    'CacheManager',
    'TextUtils',
    'PerformanceMonitor',
    'FileSystemUtils',
    'config',
    'cache_manager',
    'performance_monitor',
    'text_utils',
    'file_system_utils',
    'setup_logging',
    'validate_message',
    'format_prediction_result',
    'create_response'
]

__version__ = '1.0.0'