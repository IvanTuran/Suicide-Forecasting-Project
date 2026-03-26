import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

IN_PATH = "data/hybrid_state_model_input_clean.csv"
OUT_TABLE = "outputs/tables/correlations.csv"
OUT_PARTIAL = "outputs/tables/partial_correlations.csv"

df = pd.read_csv(IN_PATH)

target_vars = ["mean_risk", "high_risk_pct"]
ses_vars = ["gini", "unemployment_rate_pct", "bachelors_plus_pct", "median_household_income_usd"]

rows = []
pvals = []

for y in target_vars:
    for x in ses_vars:
        sub = df[[x, y]].dropna()
        r_p, p_p = pearsonr(sub[x], sub[y])
        r_s, p_s = spearmanr(sub[x], sub[y])

        rows.append({
            "y": y, "x": x,
            "pearson_r": r_p, "pearson_r2": r_p**2, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_r2": r_s**2, "spearman_p": p_s,
            "n": len(sub)
        })
        pvals.append(p_p)

# Multiple testing correction on Pearson p-values
reject, p_corr, _, _ = multipletests(pvals, method="fdr_bh")
for i in range(len(rows)):
    rows[i]["pearson_p_fdr"] = p_corr[i]
    rows[i]["pearson_sig_fdr"] = bool(reject[i])

corr_df = pd.DataFrame(rows).sort_values(["y", "pearson_p_fdr"])
corr_df.to_csv(OUT_TABLE, index=False)
print("Saved:", OUT_TABLE)

# Partial correlations:
# For each SES variable x vs y, control for the other SES variables
partials = []
for y in target_vars:
    for x in ses_vars:
        controls = [c for c in ses_vars if c != x]
        sub = df[[y, x] + controls].dropna()

        # Regress y on controls -> residuals
        Xc = sm.add_constant(sub[controls])
        model_y = sm.OLS(sub[y], Xc).fit()
        ry = model_y.resid

        # Regress x on controls -> residuals
        model_x = sm.OLS(sub[x], Xc).fit()
        rx = model_x.resid

        r_partial, p_partial = pearsonr(rx, ry)

        partials.append({
            "y": y, "x": x,
            "controls": ",".join(controls),
            "partial_r": r_partial,
            "partial_r2": r_partial**2,
            "partial_p": p_partial,
            "n": len(sub)
        })

partial_df = pd.DataFrame(partials).sort_values(["y", "partial_p"])
partial_df.to_csv(OUT_PARTIAL, index=False)
print("Saved:", OUT_PARTIAL)
