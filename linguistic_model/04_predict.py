# src/04_predict.py

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(w)
        for w in words if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/suicide_model.pkl")

text = input("Enter text: ")
clean = clean_text(text)
vec = vectorizer.transform([clean])
prob = model.predict_proba(vec)[0][1]

print(f"⚠ Suicide risk probability: {prob:.2f}")
