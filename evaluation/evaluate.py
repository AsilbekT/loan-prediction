import os
import xgboost as xgb
import cudf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from cuml.model_selection import train_test_split

# -------------------- Step 1: Load Model Features --------------------
print("Step 1: Loading feature names...")
MODEL_DIR = "models/loan_default_tuned"
FEATURE_FILE = os.path.join(MODEL_DIR, "feature_names.txt")
MODEL_FILE = os.path.join(MODEL_DIR, "loan_default_tuned_model.json")
CSV_PATH = "data/accepted_2007_to_2018Q4.csv"

with open(FEATURE_FILE, "r") as f:
    features = f.read().split(",")

# -------------------- Step 2: Load and Prepare Dataset --------------------
print("Step 2: Reading data...")
# Load headers with pandas
headers = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
df = cudf.read_csv(CSV_PATH, names=headers, skiprows=1)

print("Step 3: Filtering relevant loan statuses...")
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
df["loan_status"] = (df["loan_status"] == "Charged Off").astype("int32")

print("Step 4: Encoding and cleaning...")
# Encode categoricals
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# Fill nulls safely
for col in df.columns:
    if df[col].isnull().any():
        if str(df[col].dtype).startswith("float") or str(df[col].dtype).startswith("int"):
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna(0)

# -------------------- Step 5: Evaluation --------------------
print("Step 5: Evaluating model...")
df = df[[*features, "loan_status"]]
X = df[features]
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_pd = X_test.to_pandas()
y_test_pd = y_test.to_pandas()

dtest = xgb.DMatrix(X_test_pd)

model = xgb.Booster()
model.load_model(MODEL_FILE)

preds = model.predict(dtest)
pred_labels = (preds > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_pd, pred_labels))

print("\nClassification Report:")
print(classification_report(y_test_pd, pred_labels))

auc = roc_auc_score(y_test_pd, preds)
print(f"\nAUC Score: {auc:.4f}")


with open("evaluation/report.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test_pd, pred_labels)) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_pd, pred_labels) + "\n")
    f.write(f"AUC Score: {auc:.4f}\n")
