import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load correlations (long format)
corr_long = pd.read_csv("outputs/tables/correlations.csv")

# Pivot into matrix form
corr_matrix = corr_long.pivot(index="y", columns="x", values="pearson_r")

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Pearson Correlation: SES Indicators vs Linguistic Suicide Risk", fontsize=14)
plt.tight_layout()

# Save figure
plt.savefig("outputs/figures/correlation_matrix.png", dpi=300)
plt.close()

print("Saved: outputs/figures/correlation_matrix.png")
