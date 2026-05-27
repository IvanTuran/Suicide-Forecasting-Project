import pandas as pd
import statsmodels.api as sm

IN_PATH = "data/hybrid_state_model_input_clean.csv"
OUT = "outputs/tables/regression_summary.txt"

df = pd.read_csv(IN_PATH)

ses_vars = ["gini", "unemployment_rate_pct", "bachelors_plus_pct", "median_household_income_usd"]
reddit_vars = ["mean_risk", "high_risk_pct", "post_count"]

def fit_models(y, Xcols, name):
    sub = df[[y] + Xcols].dropna()
    X = sm.add_constant(sub[Xcols])
    yv = sub[y]

    ols = sm.OLS(yv, X).fit()
    rlm = sm.RLM(yv, X, M=sm.robust.norms.HuberT()).fit()  # robust regression

    return ols, rlm, len(sub)

with open(OUT, "w") as f:
    for y in ["mean_risk", "high_risk_pct"]:
        f.write(f"\n====================\nTARGET: {y}\n====================\n")

        ols_ses, rlm_ses, n1 = fit_models(y, ses_vars, "SES")
        f.write("\n--- OLS (SES-only) ---\n")
        f.write(ols_ses.summary().as_text())
        f.write("\n--- Robust (SES-only) ---\n")
        f.write(rlm_ses.summary().as_text())

        #Including reddit_vars for a "hybrid explanation" on distress target
        ols_hyb, rlm_hyb, n2 = fit_models(y, ses_vars + ["post_count"], "Hybrid-lite")
        f.write("\n--- OLS (SES + post_count) ---\n")
        f.write(ols_hyb.summary().as_text())
        f.write("\n--- Robust (SES + post_count) ---\n")
        f.write(rlm_hyb.summary().as_text())

print("Saved:", OUT)
