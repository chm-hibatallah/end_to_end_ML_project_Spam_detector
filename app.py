"""
Flask web application for SMS Spam Detection
Enhanced with all utilities and features
"""

import os
import sys
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from pathlib import Path

# Add src directory to path for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'src'))

# Import utility functions
from src.utils import (
    config,
    cache_manager,
    performance_monitor,
    text_utils,
    setup_logging,
    validate_message,
    format_prediction_result,
    create_response
)
from src.preprocessing import quick_clean
from src.feature_engineering import FeatureExtractor

app = Flask(__name__)

# Setup logging
setup_logging(
    log_level=config.get('logging.level', 'INFO'),
    log_file=config.get('logging.file', 'logs/app.log')
)

# Model paths - adapt to your structure
MODEL_PATH = BASE_DIR / 'notebooks' / 'spam_classifier.pkl'
VECTORIZER_PATH = BASE_DIR / 'notebooks' / 'tfidf_vectorizer.pkl'

# Alternative path if models are in different location
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / 'models' / 'spam_classifier.pkl'
    VECTORIZER_PATH = BASE_DIR / 'models' / 'tfidf_vectorizer.pkl'

# Global variables for model and vectorizer
model = None
vectorizer = None
feature_extractor = None

def load_model_and_vectorizer():
    """Load the trained model, vectorizer and initialize feature extractor."""
    global model, vectorizer, feature_extractor
    
    try:
        start_time = time.time()
        
        if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            print(f"\n{'='*60}")
            print("SMS Spam Detection - Flask Application")
            print(f"{'='*60}")
            print(f"✓ Base directory: {BASE_DIR}")
            print(f"✓ Loading model from: {MODEL_PATH}")
            
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            feature_extractor = FeatureExtractor()
            
            load_time = time.time() - start_time
            performance_monitor.record_model_load(load_time)
            
            print(f"✓ Model loaded successfully in {load_time:.2f} seconds!")
            print(f"✓ Model type: {type(model).__name__}")
            print(f"{'='*60}\n")
            return True
        else:
            print(f"\n✗ Model files not found!")
            print(f"  Expected at: {MODEL_PATH}")
            print(f"  Expected at: {VECTORIZER_PATH}")
            print(f"\n📋 Project structure check:")
            print(f"  {BASE_DIR}/")
            
            # List directories
            for item in BASE_DIR.iterdir():
                if item.is_dir():
                    print(f"    📁 {item.name}/")
                    # List some files in each directory
                    try:
                        files = list(item.glob('*.pkl'))[:3]
                        for f in files:
                            print(f"        📄 {f.name}")
                    except:
                        pass
            
            print(f"\n⚠️  Please ensure model files exist or run training script")
            print(f"{'='*60}\n")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make spam prediction for input message."""
    global model, vectorizer
    
    start_time = performance_monitor.start_prediction()
    
    try:
        # Get message from request
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify(create_response(
                success=False,
                message='No message provided',
                status_code=400
            )), 400
        
        # Validate message
        is_valid, error_msg = validate_message(
            message, 
            max_length=config.get('model.max_message_length', 5000)
        )
        
        if not is_valid:
            return jsonify(create_response(
                success=False,
                message=error_msg,
                status_code=400
            )), 400
        
        # Check cache first
        cached_result = cache_manager.get(message)
        if cached_result:
            performance_monitor.record_cache_hit(hit=True)
            performance_monitor.end_prediction(start_time, success=True)
            
            # Log prediction from cache
            cache_type = 'SPAM' if cached_result['is_spam'] else 'HAM'
            confidence = cached_result['confidence']
            print(f"Cache hit: '{message[:50]}...' → {cache_type} ({confidence:.1%})")
            
            return jsonify(create_response(
                success=True,
                data=cached_result,
                message='Prediction from cache'
            ))
        
        # Not in cache, check if model is loaded
        if model is None or vectorizer is None:
            performance_monitor.end_prediction(start_time, success=False)
            return jsonify(create_response(
                success=False,
                message='Model not loaded. Please run training script first.',
                status_code=500
            )), 500
        
        # Extract handcrafted features (optional)
        handcrafted_features = {}
        if feature_extractor:
            handcrafted_features = feature_extractor.extract_text_features(message)
        
        # Clean and vectorize message
        cleaned_message = quick_clean(message)
        message_vectorized = vectorizer.transform([cleaned_message])
        
        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        probabilities = model.predict_proba(message_vectorized)[0]
        
        # Prepare response with enhanced information
        result = format_prediction_result(
            is_spam=bool(prediction == 1),
            spam_probability=float(probabilities[1]),
            ham_probability=float(probabilities[0]),
            message=message,
            include_metadata=True
        )
        
        # Add handcrafted features to metadata
        if handcrafted_features and 'metadata' in result:
            result['metadata']['handcrafted_features'] = {
                k: float(v) for k, v in handcrafted_features.items()
            }
        
        # Cache the result
        cache_manager.set(message, result)
        performance_monitor.record_cache_hit(hit=False)
        performance_monitor.end_prediction(start_time, success=True)
        
        # Log prediction
        prediction_type = 'SPAM' if result['is_spam'] else 'HAM'
        confidence = result['confidence']
        print(f"Prediction: '{message[:50]}...' → {prediction_type} ({confidence:.1%})")
        
        return jsonify(create_response(
            success=True,
            data=result,
            message='Prediction successful'
        ))
    
    except Exception as e:
        performance_monitor.end_prediction(start_time, success=False)
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify(create_response(
            success=False,
            message=f'Prediction error: {str(e)}',
            status_code=500
        )), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'feature_extractor_loaded': feature_extractor is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add performance metrics
    status['performance'] = performance_monitor.get_metrics()
    
    # Add cache stats
    status['cache'] = cache_manager.get_stats()
    
    # Add config info
    status['config'] = {
        'model_path': str(MODEL_PATH),
        'vectorizer_path': str(VECTORIZER_PATH),
        'app_version': config.get('app.version', '1.0.0')
    }
    
    return jsonify(create_response(
        success=True,
        data=status,
        message='Health check completed'
    ))

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text features without prediction."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify(create_response(
                success=False,
                message='No message provided',
                status_code=400
            )), 400
        
        # Extract various analyses
        analysis = {
            'original': message,
            'normalized': text_utils.normalize_text(message),
            'cleaned': quick_clean(message),
            'entities': text_utils.extract_entities(message),
            'readability': text_utils.calculate_readability(message),
            'length': len(message),
            'word_count': len(message.split()),
            'character_count': {
                'total': len(message),
                'uppercase': sum(1 for c in message if c.isupper()),
                'digits': sum(1 for c in message if c.isdigit()),
                'special': sum(1 for c in message if not c.isalnum() and not c.isspace())
            }
        }
        
        # Add handcrafted features if extractor is available
        if feature_extractor:
            analysis['handcrafted_features'] = feature_extractor.extract_text_features(message)
        
        return jsonify(create_response(
            success=True,
            data=analysis,
            message='Text analysis completed'
        ))
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        return jsonify(create_response(
            success=False,
            message=f'Analysis error: {str(e)}',
            status_code=500
        )), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    """Get example messages for testing."""
    examples = {
        'spam': [
            "WINNER!! You've won a free iPhone! Call now to claim.",
            "URGENT: Your bank account has been compromised. Click here: http://fake-bank.com",
            "FREE entry to win £1000 cash prize! Text WIN to 88888.",
            "Congratulations! You've been selected for a prize. Call 555-1234",
            "You've won a luxury cruise! Reply YES to claim your award."
        ],
        'ham': [
            "Hi John, are we meeting tomorrow for lunch?",
            "Don't forget to bring the documents to the meeting.",
            "Hey, what time are we catching up tonight?",
            "Meeting rescheduled to 3 PM tomorrow.",
            "See you at the cinema at 7 PM."
        ]
    }
    
    return jsonify(create_response(
        success=True,
        data=examples,
        message='Example messages'
    ))

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get performance metrics."""
    metrics_data = performance_monitor.get_metrics()
    return jsonify(create_response(
        success=True,
        data=metrics_data,
        message='Performance metrics'
    ))

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    stats = cache_manager.get_stats()
    return jsonify(create_response(
        success=True,
        data=stats,
        message='Cache statistics'
    ))

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache."""
    cache_manager.clear()
    return jsonify(create_response(
        success=True,
        message='Cache cleared successfully'
    ))

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    config_data = {
        'app': config.get('app'),
        'model': config.get('model'),
        'features': config.get('features'),
        'paths': {
            'model': str(MODEL_PATH),
            'vectorizer': str(VECTORIZER_PATH),
            'base_dir': str(BASE_DIR)
        }
    }
    
    return jsonify(create_response(
        success=True,
        data=config_data,
        message='Configuration'
    ))

if __name__ == '__main__':
    # Try to load model at startup
    model_loaded = load_model_and_vectorizer()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded!")
        print("Possible solutions:")
        print("1. Check if model files exist in the correct location")
        print("2. Run the training script: python train_model.py")
        print("3. Update the MODEL_PATH in app.py")
        print("="*60 + "\n")
        
        # Create a placeholder for development
        print("⚠️  Running in development mode without model...")
        print("   Some features will be limited\n")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    # Run the Flask app
    print("Starting Flask server...")
    print(f"📊 Health check: http://127.0.0.1:{config.get('app.port', 5000)}/health")
    print(f"🌐 Application: http://127.0.0.1:{config.get('app.port', 5000)}")
    print(f"📈 Metrics: http://127.0.0.1:{config.get('app.port', 5000)}/metrics")
    print("="*60)
    print("Press CTRL+C to quit\n")
    
    app.run(
        debug=config.get('app.debug', True),
        host=config.get('app.host', '127.0.0.1'),
        port=config.get('app.port', 5000)
    )