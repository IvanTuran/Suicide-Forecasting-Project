# scripts/05_shap_hybrid.py
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

IN_PATH = "data/hybrid_state_model_input_clean.csv"
OUT_SUMMARY = "outputs/figures/shap_summary.png"
OUT_BAR = "outputs/figures/shap_bar.png"

os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)

df = pd.read_csv(IN_PATH)

ses = ["gini", "unemployment_rate_pct", "bachelors_plus_pct", "median_household_income_usd"]
Xcols = ses + ["high_risk_pct", "post_count"]  # no leakage if y=mean_risk
ycol = "mean_risk"

sub = df.dropna(subset=Xcols + [ycol]).copy()
X = sub[Xcols]
y = sub[ycol]

model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)

# TreeExplainer is best for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Beeswarm
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(OUT_SUMMARY, dpi=300)
plt.close()

# Bar summary
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUT_BAR, dpi=300)
plt.close()

print("Saved:", OUT_SUMMARY)
print("Saved:", OUT_BAR)
