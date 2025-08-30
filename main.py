import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import time

# Load model and preprocessing components
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('simple_rnn_imdb.keras')

def preprocess_text(text):
    # Clean text and handle punctuation
    text = ''.join([c.lower() for c in text if c.isalnum() or c.isspace() or c in ".,!?'"])
    words = text.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Custom CSS
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #2E86C1;
        padding: 20px;
    }
    .subtitle {
        color: #566573;
        font-style: italic;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='title'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze the sentiment of your movie reviews using AI!</p>", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("📝 How to Use")
    st.write("1. Enter your movie review in the text box")
    st.write("2. Click 'Analyze Sentiment' to see results")
    st.write("3. Get instant sentiment analysis!")
    
    st.header("ℹ️ About")
    st.write("This app uses RNN Deep Learning Model to analyze the emotional tone of movie reviews.")

# Main content
st.markdown("### Share Your Thoughts! 🎭")
user_input = st.text_area(
    "What did you think about the movie?",
    height=150,
    placeholder="Type your movie review here... (e.g., 'The plot was incredible and the acting was superb!')"
)

if st.button('Analyze Sentiment ✨', key='analyze'):
    if user_input.strip():
        with st.spinner('Analyzing your review...'):
            time.sleep(1)  # Add slight delay for better UX
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input, verbose=0)
            score = float(prediction[0][0])
            
            # Define sentiment and confidence
            sentiment = 'Positive' if score > 0.5 else 'Negative'
            confidence = (score if score > 0.5 else 1 - score) * 100
            
            # Display results with custom styling
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### Sentiment: {sentiment}")
                emoji = "😊 " if sentiment == "Positive" else "😔 "
                st.markdown(f"### {emoji}")
                
            with col2:
                st.markdown(f"### Confidence: {confidence:.1f}%")
                st.progress(confidence/100)
            
            # Additional analysis
            st.markdown("---")
            st.markdown("### Detailed Analysis")
            
            if score > 0.8:
                st.success("Highly Positive Review! 🌟")
            elif score > 0.6:
                st.info("Moderately Positive Review 👍")
            elif score > 0.4:
                st.warning("Mixed or Neutral Review 🤔")
            elif score > 0.2:
                st.error("Moderately Negative Review 👎")
            else:
                st.error("Highly Negative Review! 💔")
                
    else:
        st.warning("Please enter a review first! 📝")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Made by Mantesh Mhetre | Movie Review Analyzer v1.0
    </div>
    """, 
    unsafe_allow_html=True
)