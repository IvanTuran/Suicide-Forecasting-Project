import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

IN_PATH = "data/hybrid_state_model_input_clean.csv"
OUT_TABLE = "outputs/tables/model_cv_results.csv"

df = pd.read_csv(IN_PATH)

# For now, target is the distress signal itself. Later we swap y to CDC suicide rate for validation.
targets = ["mean_risk"]  # you guys can also run "high_risk_pct"

ses = ["gini", "unemployment_rate_pct", "bachelors_plus_pct", "median_household_income_usd"]
reddit = ["mean_risk", "high_risk_pct", "post_count"]  # when y is CDC later, these are inputs
reddit_only_for_y_meanrisk = ["high_risk_pct", "post_count"]  # avoid leakage if y=mean_risk

def eval_cv(X, y):
    kf = RepeatedKFold(n_splits=5, n_repeats=50, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(random_state=42))
    ])

    r2s, rmses = [], []
    for tr, te in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))

    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(rmses)), float(np.std(rmses))

rows = []

for yname in targets:
    sub = df.dropna(subset=[yname] + ses + ["high_risk_pct", "post_count"]).copy()
    y = sub[yname]

    # If predicting mean_risk, don't include mean_risk as an input
    X_ses = sub[ses]
    X_reddit = sub[reddit_only_for_y_meanrisk]
    X_hybrid = sub[ses + reddit_only_for_y_meanrisk]

    r2, r2sd, rmse, rmsesd = eval_cv(X_ses, y)
    rows.append({"target": yname, "model": "SES-only", "r2_mean": r2, "r2_sd": r2sd, "rmse_mean": rmse, "rmse_sd": rmsesd, "n": len(sub)})

    r2, r2sd, rmse, rmsesd = eval_cv(X_reddit, y)
    rows.append({"target": yname, "model": "Reddit-only", "r2_mean": r2, "r2_sd": r2sd, "rmse_mean": rmse, "rmse_sd": rmsesd, "n": len(sub)})

    r2, r2sd, rmse, rmsesd = eval_cv(X_hybrid, y)
    rows.append({"target": yname, "model": "Hybrid", "r2_mean": r2, "r2_sd": r2sd, "rmse_mean": rmse, "rmse_sd": rmsesd, "n": len(sub)})

out = pd.DataFrame(rows).sort_values(["target", "model"])
out.to_csv(OUT_TABLE, index=False)
print("Saved:", OUT_TABLE)
print(out)
