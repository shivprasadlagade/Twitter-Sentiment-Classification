import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
def clean_text(text)
    # Remove URLs, mentions, hashtags, special char, and convert to lowercase
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z0-9 ]+", "", str(text))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_words)

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('logreg_model.pkl')
    vectorizer = joblib.load('tfidf.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

st.title("Twitter Sentiment Classifier")
st.write("Enter a tweet and get sentiment prediction (Negative, Neutral, Positive).")

tweet_input = st.text_area("Tweet text:")

if st.button("Predict"):
    processed = clean_text(tweet_input)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    st.success(f"Predicted Sentiment: {pred}")
