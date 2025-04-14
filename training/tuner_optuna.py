# File: training/tuner_optuna.py
import cudf
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from cuml.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.preprocessing import load_clean_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "loan_default_tuned")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data for tuning...")
DATA_PATH = os.path.join(BASE_DIR, "data", "accepted_2007_to_2018Q4.csv")
df, features = load_clean_data(DATA_PATH)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_pd = X_train.to_pandas()
X_valid_pd = X_valid.to_pandas()
y_train_pd = y_train.to_pandas()
y_valid_pd = y_valid.to_pandas()

dtrain = xgb.DMatrix(X_train_pd, label=y_train_pd)
dvalid = xgb.DMatrix(X_valid_pd, label=y_valid_pd)

def objective(trial):
    param = {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    model = xgb.train(param, dtrain, num_boost_round=100)
    preds = model.predict(dvalid)
    auc = roc_auc_score(y_valid_pd, preds)
    return auc

study = optuna.create_study(direction="maximize")
print("Starting hyperparameter optimization...")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best AUC:", round(study.best_value, 4))
print("Best Params:", study.best_params)

# Save best model
print("Training final model with best parameters...")
model = xgb.train({
    **study.best_params,
    "tree_method": "hist",
    "device": "cuda",
    "objective": "binary:logistic",
    "eval_metric": "auc"
}, dtrain, num_boost_round=100)

model.save_model(os.path.join(MODEL_DIR, "loan_default_tuned_model.json"))
with open(os.path.join(MODEL_DIR, "best_params.txt"), "w") as f:
    for k, v in study.best_params.items():
        f.write(f"{k}: {v}\n")

print("Model and best parameters saved to:", MODEL_DIR)
