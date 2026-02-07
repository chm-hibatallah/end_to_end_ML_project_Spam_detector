
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Create a simple pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression())
])

# Dummy training data
X_train = ["win free money", "hello how are you", "claim your prize", "meeting tomorrow"]
y_train = [1, 0, 1, 0]  # 1=spam, 0=ham

# Train on dummy data
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'models/spam_classifier.pkl')
print("✅ Dummy model created!")