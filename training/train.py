import cudf
import xgboost as xgb
import numpy as np
import pandas as pd
from cuml.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import time
from datetime import datetime
import sys

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.preprocessing import load_clean_data

# Define consistent model output path relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "loan_default_v1")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess data
print("Loading and preprocessing data...")
DATA_PATH = os.path.join(BASE_DIR, "data", "accepted_2007_to_2018Q4.csv")
df, features = load_clean_data(DATA_PATH)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_pd, X_test_pd = X_train.to_pandas(), X_test.to_pandas()
y_train_pd, y_test_pd = y_train.to_pandas(), y_test.to_pandas()

# XGBoost training
params = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "binary:logistic",
    "eval_metric": "auc"
}

print("Training model...")
dtrain = xgb.DMatrix(X_train_pd, label=y_train_pd)
dtest = xgb.DMatrix(X_test_pd, label=y_test_pd)
start = time.time()
model = xgb.train(params, dtrain, num_boost_round=100)
print("Training complete in", round(time.time() - start, 2), "seconds")

# Evaluation
preds = model.predict(dtest)
pred_labels = (preds > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test_pd, pred_labels))
print("Classification Report:")
print(classification_report(y_test_pd, pred_labels))
auc = roc_auc_score(y_test_pd, preds)
print("AUC:", round(auc, 4))

# Save artifacts
model.save_model(os.path.join(MODEL_DIR, "loan_default_model.json"))
with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
    f.write(",".join(X_train_pd.columns.tolist()))
with open(os.path.join(MODEL_DIR, "meta.txt"), "w") as f:
    f.write(f"Trained: {datetime.now()}\nAUC: {auc:.4f}\n")