import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the models and vectorizer
log_reg = pickle.load(open('log_reg_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

# App title with a colorful heading
st.markdown("""
    <h1 style='text-align: center; font-size: 48px; font-weight: 800; color: #1a237e;'>💬 Sentiment Analysis of Text</h1>
    <p style='text-align: center; font-size: 16px; color: #90a4ae;'>Predict whether a sentence expresses Positive or Negative sentiment.</p>
""", unsafe_allow_html=True)


# ------------------ Sentiment Prediction Section ------------------

# Add this before your text_area
st.markdown("""
    <style>
    textarea {
        background-color: #2e3a59 !important;
        color: white !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 📝 Enter Text for Sentiment Prediction")
user_input = st.text_area("Type your sentence here...", height=150, max_chars=500)


if st.button("🔍 Analyze Sentiment"):
    if user_input:
        # Clean and preprocess input text
        clean = re.sub('[^a-zA-Z]', ' ', user_input).lower().split()
        cleaned_text = ' '.join(clean)
        vect_input = vectorizer.transform([cleaned_text])
        
        # Get prediction
        prediction = log_reg.predict(vect_input)[0]

        # Display result with styled output
        if prediction == 1:
            st.markdown("""
                <div style='background-color: #A1D6A4; padding: 15px; border-radius: 12px; text-align: center;'>
                    <h3 style='color: #2e3a59;'>😊 Positive Sentiment</h3>
                    <h1>💖</h1>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
                <div style='background-color: #F9D0D0; padding: 15px; border-radius: 12px; text-align: center;'>
                    <h3 style='color: #2e3a59;'>😞 Negative Sentiment</h3>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

# ------------------ Styling ------------------
st.markdown("""
    <style>
        .stTextArea textarea {
            font-size: 16px;
            color: #333;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)


