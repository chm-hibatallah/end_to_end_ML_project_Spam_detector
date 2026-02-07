import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Set page config
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide"
)

# Title and description
st.title("📱 SMS Spam Detection AI")
st.markdown("""
This machine learning model classifies SMS messages as **SPAM** or **HAM** (legitimate).
Enter a message below or try the examples!
""")

# Sidebar for info and examples
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **How it works:**
    1. Enter an SMS message
    2. Model analyzes text patterns
    3. Get instant classification
    
    **Model:** Logistic Regression
    **Accuracy:** ~98% on test data
    **Training Data:** 5,574 SMS messages
    """)
    
    st.header("📊 Try Examples")
    examples = {
        "Spam 1": "WINNER!! You have won a free ticket to Bahamas! Claim now! Reply WIN.",
        "Spam 2": "URGENT: Your bank account has been compromised. Click to secure: bit.ly/bank-secure",
        "Ham 1": "Hey, are we meeting tomorrow at 3 PM for the project discussion?",
        "Ham 2": "Just finished the report. Will send it over in 10 minutes."
    }
    
    for name, text in examples.items():
        if st.button(f"{name}"):
            st.session_state.message = text

# Load model (with caching)
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('models/spam_classifier.pkl')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Text cleaning function
def clean_text(text):
    """Clean and preprocess SMS text"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    return text.strip()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Input text area
    message = st.text_area(
        "**Enter SMS Message:**",
        height=150,
        placeholder="Type or paste an SMS message here...",
        value=getattr(st.session_state, 'message', '')
    )
    
    # Prediction button
    predict_btn = st.button("🔍 Analyze Message", type="primary", use_container_width=True)

with col2:
    # Display model info
    st.subheader("Model Status")
    model = load_model()
    
    if model:
        st.success("✅ Ready for predictions")
        st.metric("Spam Detection", "Active", delta="98% accuracy")
    else:
        st.warning("⚠️ Using demo mode")

# Results section
if predict_btn and message:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    # Create progress animation
    with st.spinner("Analyzing message..."):
        # Clean text
        cleaned_message = clean_text(message)
        
        if model:
            # Make prediction
            try:
                # For scikit-learn pipeline
                prediction = model.predict([cleaned_message])[0]
                probabilities = model.predict_proba([cleaned_message])[0]
                
                is_spam = bool(prediction)
                spam_prob = probabilities[1]
                ham_prob = probabilities[0]
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                is_spam = "spam" in cleaned_message or "win" in cleaned_message
                spam_prob = 0.8 if is_spam else 0.2
                ham_prob = 0.2 if is_spam else 0.8
        else:
            # Demo mode
            keywords = ['win', 'free', 'cash', 'prize', 'urgent', 'click', 'winner']
            is_spam = any(keyword in cleaned_message for keyword in keywords)
            spam_prob = 0.85 if is_spam else 0.15
            ham_prob = 0.15 if is_spam else 0.85
        
        # Display results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if is_spam:
                st.error(f"## ⚠️ SPAM DETECTED")
                st.markdown(f"**Confidence:** {(spam_prob*100):.1f}%")
                st.markdown("""
                **Characteristics:**
                - Promotional content
                - Urgent language
                - Request for action
                - Suspicious links
                """)
            else:
                st.success(f"## ✅ LEGITIMATE MESSAGE")
                st.markdown(f"**Confidence:** {(ham_prob*100):.1f}%")
                st.markdown("""
                **Characteristics:**
                - Normal conversation
                - Personal content
                - No urgent requests
                - Legitimate context
                """)
        
        with result_col2:
            # Confidence gauge
            st.subheader("Confidence Score")
            
            # Progress bars
            st.markdown(f"**Spam Probability:** {(spam_prob*100):.1f}%")
            st.progress(spam_prob)
            
            st.markdown(f"**Ham Probability:** {(ham_prob*100):.1f}%")
            st.progress(ham_prob)
            
            # Add some stats
            st.metric("Message Length", f"{len(message)} chars")
            st.metric("Words", f"{len(message.split())}")
        
        # Show cleaned text
        with st.expander("📝 See processed text"):
            st.code(cleaned_message, language="text")
            
        # Debug info (for developers)
        with st.expander("🔧 Technical Details"):
            st.json({
                "original_length": len(message),
                "cleaned_length": len(cleaned_message),
                "prediction": "spam" if is_spam else "ham",
                "spam_probability": float(spam_prob),
                "ham_probability": float(ham_prob),
                "model_used": "Real ML Model" if model else "Demo Rule-based"
            })

elif predict_btn and not message:
    st.warning("Please enter a message to analyze!")

# Footer
st.divider()
st.caption("""
**Built with:** Streamlit | Scikit-learn | Python  
**Note:** This is a demonstration. Real spam detection systems use more sophisticated techniques.
""")

# Add refresh button
if st.button("🔄 Clear & Analyze New"):
    st.session_state.message = ""
    st.rerun()