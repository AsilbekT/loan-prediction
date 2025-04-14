# correlation_analysis.py
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.preprocessing import load_clean_data

# Paths
DATA_PATH = os.path.join("data", "accepted_2007_to_2018Q4.csv")
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and clean data
print("🔄 Cleaning and loading data...")
df, features = load_clean_data(DATA_PATH)
df = df.to_pandas()  # correlation needs pandas, not cudf

# Calculate correlation
print("📊 Calculating correlation matrix...")
corr_matrix = df.corr()

# Focus on top correlated features
if "loan_status" in corr_matrix.columns:
    top_corr = corr_matrix["loan_status"].abs().sort_values(ascending=False)
    top_features = top_corr[1:26].index.tolist()
    filtered_corr = corr_matrix.loc[top_features, top_features]

    # Plot heatmap
    print("📈 Saving heatmap...")
    plt.figure(figsize=(14, 12))
    sns.heatmap(filtered_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Top 25 Feature Correlations with Loan Status")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_top25.png"))

    print("✅ correlation_top25.png saved to analysis/")
else:
    print("❌ 'loan_status' not found in columns after preprocessing.")
