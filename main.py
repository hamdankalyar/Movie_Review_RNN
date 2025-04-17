# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ============== Streamlit App ==============

# Page configuration
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #2b5876;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .stTextArea textarea {
        background-color: #f9f9f9;
        font-size: 16px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(to right, #2b5876, #4e4376);
        color: white;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="title">🎬 IMDB Movie Review Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a movie review below to detect whether it\'s <b>Positive</b> or <b>Negative</b>.</div>', unsafe_allow_html=True)

# User Input
user_input = st.text_area("✍️ Your Review Here:", height=200)

# Prediction
if st.button('🔍 Analyze Sentiment'):

    if user_input.strip() == "":
        st.warning("Please enter a review before classification.")
    else:
        with st.spinner("Analyzing..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            score = prediction[0][0]
            sentiment = "😊 Positive" if score > 0.5 else "😞 Negative"
            emoji_color = "#4CAF50" if score > 0.5 else "#F44336"

        # Result Box
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color:{emoji_color};">Sentiment: {sentiment}</h2>
            <p style="font-size:18px;">Prediction Confidence: <b>{score:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Please enter a movie review above and click 'Analyze Sentiment'.")
