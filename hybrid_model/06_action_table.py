# scripts/06_action_table.py
import os
import numpy as np
import pandas as pd

IN_PATH = "data/hybrid_state_model_input_clean.csv"
OUT = "outputs/tables/state_action_table.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(IN_PATH).dropna().copy()

# Quartile buckets (duplicates='drop' avoids crashes if ties occur)
df["distress_q"] = pd.qcut(df["mean_risk"], 4, labels=["low","mid-low","mid-high","high"], duplicates="drop")
df["unemp_q"] = pd.qcut(df["unemployment_rate_pct"], 4, labels=["low","mid-low","mid-high","high"], duplicates="drop")
df["income_q"] = pd.qcut(df["median_household_income_usd"], 4, labels=["low","mid-low","mid-high","high"], duplicates="drop")

def recommend(row):
    # Public-health / policy-level, non-clinical suggestions
    if row["distress_q"] == "high" and row["unemp_q"] == "high":
        return "Job-loss support + outreach (988 awareness, community gatekeeper training)"
    if row["distress_q"] == "high" and row["income_q"] == "low":
        return "Expand low-cost access points (tele-mental health, school/community screening programs)"
    if row["distress_q"] == "high" and row["income_q"] in ["mid-high","high"]:
        return "Targeted stigma-reduction + optimized crisis routing and follow-up resources"
    if row["distress_q"] in ["mid-high"] and row["unemp_q"] == "high":
        return "Preventive outreach via workforce programs + local resource navigation"
    return "Baseline prevention resources; monitor trends"

df["recommendation"] = df.apply(recommend, axis=1)

# Signal strength and feasibility
# signal_strength is a transparent proxy for confidence in the social-media signal volume
df["signal_strength"] = np.log1p(df["post_count"])
df["feasibility_score"] = np.clip(
    (df["signal_strength"] / df["signal_strength"].max()) * 10,
    1, 10
).round(2)

cols = [
    "state_abbr","state",
    "mean_risk","high_risk_pct","post_count",
    "gini","unemployment_rate_pct","bachelors_plus_pct","median_household_income_usd",
    "distress_q","unemp_q","income_q",
    "signal_strength","feasibility_score",
    "recommendation"
]

df_out = df[cols].sort_values(["mean_risk","post_count"], ascending=[False, False])
df_out.to_csv(OUT, index=False)

print("Saved:", OUT)
print(df_out.head(10).to_string(index=False))
