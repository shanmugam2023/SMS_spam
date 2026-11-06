from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

@st.cache_resource
def load_model():
    model_file = MODEL_DIR / "spam_model.pkl"
    vector_file = MODEL_DIR / "tfidf_vectorizer.pkl"
    if model_file.exists() and vector_file.exists():
        with model_file.open("rb") as f:
            model = pickle.load(f)
        with vector_file.open("rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer

    data = pd.read_csv(ROOT / "data" / "sms_dataset.csv").dropna()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    features = vectorizer.fit_transform(data["message"])
    model = MultinomialNB().fit(features, data["label"])
    MODEL_DIR.mkdir(exist_ok=True)
    with model_file.open("wb") as f:
        pickle.dump(model, f)
    with vector_file.open("wb") as f:
        pickle.dump(vectorizer, f)
    return model, vectorizer

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("SMS Spam Detection Using Machine Learning")
st.write("Enter a text message. The ML model predicts whether it looks like spam.")
model, vectorizer = load_model()
message = st.text_area("SMS message", placeholder="Type or paste a message here...")
if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict(vectorizer.transform([message]))[0]
        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM (HAM)")
        st.caption("Educational demonstration only; predictions may be incorrect.")
