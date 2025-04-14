import os
import cudf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
DATA_PATH = "data/accepted_2007_to_2018Q4.csv"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load headers properly using pandas
print("📥 Reading headers and data...")
headers = pd.read_csv(DATA_PATH, nrows=0).columns
df = cudf.read_csv(DATA_PATH, skiprows=1, names=headers)
df.columns = df.columns.str.strip()

# Step 2: Filter numeric columns
numeric_df = df.select_dtypes(include=["number"])

# Step 3: Generate statistical summary
print("🧮 Calculating descriptive statistics...")
desc = numeric_df.describe().to_pandas().transpose()
desc["variance"] = numeric_df.var().to_pandas()
desc["skewness"] = numeric_df.skew().to_pandas()
desc["kurtosis"] = numeric_df.kurtosis().to_pandas()
desc["missing_values"] = df.isnull().sum().to_pandas()

# Save to CSV
desc.to_csv(os.path.join(OUTPUT_DIR, "statistical_summary.csv"))
print("✅ Saved: analysis/statistical_summary.csv")

# Step 4: Visualizations
print("📊 Generating visualizations...")

# Top 20 by variance
plt.figure(figsize=(14, 6))
top_var = desc["variance"].sort_values(ascending=False).head(20)
sns.barplot(x=top_var.values, y=top_var.index)
plt.title("Top 20 Features by Variance")
plt.xlabel("Variance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_variance.png"))
plt.close()

# Top 20 by absolute skewness
plt.figure(figsize=(14, 6))
top_skew = desc["skewness"].abs().sort_values(ascending=False).head(20)
sns.barplot(x=top_skew.values, y=top_skew.index)
plt.title("Top 20 Features by Absolute Skewness")
plt.xlabel("Absolute Skewness")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_skewness.png"))
plt.close()

# Top 20 by missing values
plt.figure(figsize=(14, 6))
top_missing = desc["missing_values"].sort_values(ascending=False).head(20)
sns.barplot(x=top_missing.values, y=top_missing.index)
plt.title("Top 20 Features by Missing Values")
plt.xlabel("Missing Value Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_missing_values.png"))
plt.close()

print("🎉 All charts saved in analysis/")
