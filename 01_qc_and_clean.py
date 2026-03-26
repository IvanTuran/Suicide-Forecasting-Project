import pandas as pd
import numpy as np

IN_PATH = "data/hybrid_state_model_input.csv"
OUT_PATH = "data/hybrid_state_model_input_clean.csv"

df = pd.read_csv(IN_PATH)

print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# Required columns (edit if your column names differ)
required = [
    "state_abbr", "state",
    "mean_risk", "high_risk_pct", "post_count",
    "gini", "unemployment_rate_pct", "bachelors_plus_pct", "median_household_income_usd"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

#Basic cleaning
df["state"] = df["state"].astype(str).str.strip()
df["state_abbr"] = df["state_abbr"].astype(str).str.strip()

numeric_cols = [
    "mean_risk", "high_risk_pct", "post_count",
    "gini", "unemployment_rate_pct", "bachelors_plus_pct", "median_household_income_usd"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Checks
print("\nMissing values per column:\n", df.isna().sum())

# Check for duplicates
dupes = df["state_abbr"].duplicated().sum()
print("\nDuplicate state_abbr count:", dupes)
if dupes > 0:
    print(df[df["state_abbr"].duplicated(keep=False)].sort_values("state_abbr"))

# Check range sanity
def rng(col):
    return (df[col].min(), df[col].max())

print("\nRanges:")
for c in numeric_cols:
    print(c, rng(c))

# remove extreme-low sample sizes (sensitivity test later)
# For main analysis: keep all, but flag low-N states.
df["low_sample_flag"] = (df["post_count"] < 10).astype(int)

# Save cleaned
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned file -> {OUT_PATH}")
