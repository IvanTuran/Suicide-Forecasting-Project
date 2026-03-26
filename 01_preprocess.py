# src/01_preprocess.py

import pandas as pd
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

# Load data
df = pd.read_csv("data/suicidal_ideation_reddit.csv")

df["clean_text"] = df["usertext"].apply(clean_text)

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)

print("Preprocessing complete. Saved to data/cleaned_data.csv")
