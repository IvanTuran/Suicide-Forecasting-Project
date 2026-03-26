# src/02_train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Remove rows where clean_text is empty or NaN
df = df.dropna(subset=["clean_text"])  # Drop NaN
df = df[df["clean_text"].str.strip() != ""]  # Drop empty strings
df = df[df["clean_text"].str.len() > 10]  # Keep only longer than 10 chars

print(f"Using {len(df)} valid posts for training")

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# Save artifacts
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(model, "models/suicide_model.pkl")

print("Model trained and saved to models/")
