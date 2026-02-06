# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model
model = joblib.load('C:\\Users\\dell\\Documents\\sms-spam-detection\\notebooks\\spam_classifier.pkl')

# Text cleaning function
def clean_text(text):
    """
    Clean and preprocess text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    
    # Clean and predict
    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])
    probability = model.predict_proba([cleaned_message])[0]
    
    result = {
        'is_spam': bool(prediction[0]),
        'spam_probability': float(probability[1]),
        'ham_probability': float(probability[0])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)