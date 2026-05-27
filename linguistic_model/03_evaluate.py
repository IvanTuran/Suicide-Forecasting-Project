# src/03_evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Remove rows where clean_text is empty or NaN (prevents vectorizer error)
df = df.dropna(subset=["clean_text"])  # Drop rows with missing clean_text
df = df[df["clean_text"].str.strip() != ""]  # Drop empty strings
df = df[df["clean_text"].str.len() > 10]  # Optional: keep only longer texts

print(f"Using {len(df)} valid posts for evaluation")

# Load model and vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/suicide_model.pkl")

# Transform text and predict
X = vectorizer.transform(df["clean_text"])
y = df["label"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Print results
print("\nClassification Report:")
print(classification_report(y, y_pred))

print(f"\nROC-AUC Score: {roc_auc_score(y, y_prob):.4f}")
