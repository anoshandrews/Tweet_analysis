import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re

model = joblib.load('mnb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title('Sentiment Analysis')

review = st.text_area('Enter your review')

def predict_sentiment(review):
    
    review_cleaned = re.sub(r'[^a-zA-Z\s]', '', review).lower().strip()
    review_vectorized = vectorizer.transform([review_cleaned])
    
    sentiment = model.predict(review_vectorized)
    
    if sentiment == 'negative':
        return 'Negative'
    elif sentiment == 'neutral':
        return 'Neutral'
    elif sentiment == 'positive':
        return 'Positive'

if st.button('Predict sentiment'):
    if review:
        sentiment = predict_sentiment(review)
        st.write(f'The prediction of the review is: **{sentiment}**')
    else:
        st.write('Please enter a valid review to analyze')