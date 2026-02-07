"""
Train spam detection model from spam.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_save_model():
    """Train model from spam.csv and save it."""
    
    print("="*60)
    print("Training SMS Spam Detection Model")
    print("="*60)
    
    # Load data from CSV
    print("\n1. Loading data from spam.csv...")
    try:
        df = pd.read_csv('data/spam.csv', encoding='latin-1')
        
        # The CSV typically has columns: v1 (label), v2 (message)
        # Keep only the first two columns
        df = df.iloc[:, 0:2]
        df.columns = ['label', 'message']
        
        print(f"   ✓ Total messages: {len(df)}")
        print(f"   ✓ Spam: {sum(df['label'] == 'spam')}")
        print(f"   ✓ Ham: {sum(df['label'] == 'ham')}")
        
    except FileNotFoundError:
        print("   ✗ Error: spam.csv not found in data/ folder")
        return None, None
    
    # Convert labels to binary
    print("\n2. Preparing data...")
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['message'].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✓ Train samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    
    # Create vectorizer
    print("\n3. Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Vectorize data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   ✓ Feature dimensions: {X_train_vec.shape[1]}")
    
    # Train model
    print("\n4. Training Naive Bayes model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)
    print("   ✓ Model trained successfully!")
    
    # Evaluate
    print("\n5. Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model and vectorizer
    print(f"\n6. Saving model to models/...")
    joblib.dump(model, 'models/spam_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    print("   ✓ Model saved: models/spam_classifier.pkl")
    print("   ✓ Vectorizer saved: models/tfidf_vectorizer.pkl")
    
    # Test predictions
    print("\n7. Testing predictions...")
    test_messages = [
        "WINNER!! You've won $1000! Call now!",
        "Hey, are you free for lunch tomorrow?",
        "Congratulations! Click here to claim your prize",
        "Meeting at 3pm in conference room"
    ]
    
    for msg in test_messages:
        msg_vec = vectorizer.transform([msg])
        pred = model.predict(msg_vec)[0]
        prob = model.predict_proba(msg_vec)[0]
        
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob[pred] * 100
        
        print(f"\n   Message: {msg[:45]}...")
        print(f"   → {label} (confidence: {confidence:.1f}%)")
    
    print("\n" + "="*60)
    print("✓ Training complete! Your model is ready to use.")
    print("="*60 + "\n")
    
    return model, vectorizer


if __name__ == '__main__':
    train_and_save_model()