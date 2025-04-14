# 🚀 Loan Default Prediction API

This project is a **Loan Default Risk Prediction API** that leverages GPU-accelerated machine learning using RAPIDS cuDF and XGBoost. It provides a RESTful endpoint to predict whether a loan is likely to default based on LendingClub data (2007–2018).

---

## 🧠 Project Members

- Harpreet Singh
- Asilbek Turgunboev

---

## 🌍 Production URL

**Domain**: [loan.ataxi.uz](http://loan.ataxi.uz)

---

📊 Sample Inputs & Explanations

This section illustrates different kinds of inputs sent to the /predict API endpoint and what happens in each case.

✅ Example 1 – Valid Input, Expected Output: ```"Fully Paid"```
```json
{
  "loan_amnt": 8000,
  "term": 36,
  "int_rate": 10.25,
  "installment": 259.96,
  "grade": 2,
  "sub_grade": 1,
  "emp_length": 7,
  "home_ownership": 1,
  "annual_inc": 45000,
  "verification_status": 1,
  "dti": 12.3,
  "delinq_2yrs": 0,
  "earliest_cr_line": 2005,
  "fico_range_low": 690,
  "fico_range_high": 695,
  "inq_last_6mths": 1,
  "open_acc": 10,
  "pub_rec": 0,
  "revol_bal": 3000,
  "revol_util": 28.7,
  "total_acc": 21,
  "initial_list_status": 1,
  "out_prncp": 0,
  "out_prncp_inv": 0,
  "total_pymnt": 8200,
  "total_pymnt_inv": 8200,
  "total_rec_prncp": 8000,
  "total_rec_int": 200,
  "total_rec_late_fee": 0,
  "recoveries": 0,
  "collection_recovery_fee": 0,
  "last_pymnt_amnt": 259.96,
  "last_fico_range_high": 699,
  "last_fico_range_low": 695,
  "collections_12_mths_ex_med": 0,
  "policy_code": 1,
  "application_type": 0,
  "acc_now_delinq": 0,
  "tot_coll_amt": 0,
  "tot_cur_bal": 15000,
  "open_acc_6m": 1,
  "open_act_il": 1,
  "open_il_12m": 1,
  "open_il_24m": 2,
  "mths_since_rcnt_il": 10,
  "total_bal_il": 7000,
  "il_util": 62,
  "open_rv_12m": 1,
  "open_rv_24m": 2,
  "max_bal_bc": 2500,
  "all_util": 38,
  "total_rev_hi_lim": 12000,
  "inq_fi": 0,
  "total_cu_tl": 2,
  "inq_last_12m": 1,
  "acc_open_past_24mths": 4,
  "avg_cur_bal": 4800,
  "bc_open_to_buy": 6000,
  "bc_util": 31,
  "chargeoff_within_12_mths": 0,
  "delinq_amnt": 0,
  "mo_sin_old_il_acct": 150,
  "mo_sin_old_rev_tl_op": 170,
  "mo_sin_rcnt_rev_tl_op": 10,
  "mo_sin_rcnt_tl": 8,
  "mort_acc": 1,
  "mths_since_recent_bc": 6,
  "mths_since_recent_inq": 3,
  "num_accts_ever_120_pd": 0,
  "num_actv_bc_tl": 3,
  "num_actv_rev_tl": 2,
  "num_bc_sats": 4,
  "num_bc_tl": 5,
  "num_il_tl": 3,
  "num_op_rev_tl": 3,
  "num_rev_accts": 10,
  "num_rev_tl_bal_gt_0": 5,
  "num_sats": 9,
  "num_tl_120dpd_2m": 0,
  "num_tl_30dpd": 0,
  "num_tl_90g_dpd_24m": 0,
  "num_tl_op_past_12m": 2,
  "pct_tl_nvr_dlq": 98,
  "percent_bc_gt_75": 0,
  "pub_rec_bankruptcies": 0,
  "tax_liens": 0,
  "tot_hi_cred_lim": 30000,
  "total_bal_ex_mort": 13000,
  "total_bc_limit": 9000,
  "total_il_high_credit_limit": 18000,
  "hardship_flag": 0,
  "disbursement_method": 1,
  "debt_settlement_flag": 0
}
```
📤 Sample Response:
```json
{
  "default_probability": 0.0191,
  "prediction": "Fully Paid"
}
```
✅ Example 2 – High-Risk Input, Expected Output: ```"Default"```
```json
{
  "loan_amnt": 35000,
  "term": 60,
  "int_rate": 25.5,
  "installment": 1100.12,
  "grade": 6,
  "sub_grade": 5,
  "emp_length": 0,
  "home_ownership": 3,
  "annual_inc": 18000,
  "verification_status": 0,
  "dti": 45.2,
  "delinq_2yrs": 3,
  "earliest_cr_line": 2000,
  "fico_range_low": 600,
  "fico_range_high": 610,
  "inq_last_6mths": 5,
  "open_acc": 4,
  "pub_rec": 1,
  "revol_bal": 20000,
  "revol_util": 89.0,
  "total_acc": 9,
  "initial_list_status": 1,
  "out_prncp": 10000,
  "out_prncp_inv": 10000,
  "total_pymnt": 2500,
  "total_pymnt_inv": 2500,
  "total_rec_prncp": 2000,
  "total_rec_int": 500,
  "total_rec_late_fee": 50,
  "recoveries": 0,
  "collection_recovery_fee": 0,
  "last_pymnt_amnt": 100,
  "last_fico_range_high": 620,
  "last_fico_range_low": 610,
  "collections_12_mths_ex_med": 1,
  "policy_code": 1,
  "application_type": 1,
  "acc_now_delinq": 1,
  "tot_coll_amt": 3000,
  "tot_cur_bal": 5000,
  "open_acc_6m": 0,
  "open_act_il": 0,
  "open_il_12m": 0,
  "open_il_24m": 1,
  "mths_since_rcnt_il": 30,
  "total_bal_il": 5000,
  "il_util": 95,
  "open_rv_12m": 0,
  "open_rv_24m": 1,
  "max_bal_bc": 3500,
  "all_util": 90,
  "total_rev_hi_lim": 15000,
  "inq_fi": 2,
  "total_cu_tl": 1,
  "inq_last_12m": 3,
  "acc_open_past_24mths": 1,
  "avg_cur_bal": 1600,
  "bc_open_to_buy": 300,
  "bc_util": 98,
  "chargeoff_within_12_mths": 1,
  "delinq_amnt": 1000,
  "mo_sin_old_il_acct": 240,
  "mo_sin_old_rev_tl_op": 200,
  "mo_sin_rcnt_rev_tl_op": 8,
  "mo_sin_rcnt_tl": 5,
  "mort_acc": 0,
  "mths_since_recent_bc": 18,
  "mths_since_recent_inq": 6,
  "num_accts_ever_120_pd": 2,
  "num_actv_bc_tl": 1,
  "num_actv_rev_tl": 2,
  "num_bc_sats": 1,
  "num_bc_tl": 2,
  "num_il_tl": 2,
  "num_op_rev_tl": 1,
  "num_rev_accts": 5,
  "num_rev_tl_bal_gt_0": 2,
  "num_sats": 4,
  "num_tl_120dpd_2m": 0,
  "num_tl_30dpd": 1,
  "num_tl_90g_dpd_24m": 1,
  "num_tl_op_past_12m": 1,
  "pct_tl_nvr_dlq": 40,
  "percent_bc_gt_75": 90,
  "pub_rec_bankruptcies": 1,
  "tax_liens": 1,
  "tot_hi_cred_lim": 40000,
  "total_bal_ex_mort": 32000,
  "total_bc_limit": 5000,
  "total_il_high_credit_limit": 15000,
  "hardship_flag": 1,
  "disbursement_method": 0,
  "debt_settlement_flag": 1
}
```
Response
```json
{
    "default_probability": 0.9992,
    "prediction": "Default"
}
```

❌ Example 3 – Invalid Input (raw text fields)
```json
{
  "loan_amnt": 10000,
  "term": "36 months",
  "grade": "C",
  "emp_length": "10+ years",
  "home_ownership": "RENT",
  "annual_inc": 60000
}

```
## 📂 Project Structure

```
loan-prediction/
│
├── app/                    # Flask REST API
│   └── app.py              # Main application file
│
├── training/               # Model training & tuning
│   ├── train.py
│   ├── preprocessing.py
│   └── tuner_optuna.py
│
├── evaluation/             # Model evaluation
│   └── evaluate.py
│
├── models/                 # Trained models and feature files
│   └── loan_default_tuned/
│       ├── loan_default_tuned_model.json
│       └── best_params.txt
│
├── data/                   # Raw dataset (CSV format)
│   └── accepted_2007_to_2018Q4.csv
│
├── requirements.txt
├── Dockerfile
└── README.md               
```
---

## 🧪 How to Use the API

**Endpoint:** `/predict`

**Request Example:**

```bash
curl -X POST http://localhost/predict \\
    -H "Content-Type: application/json" \\
    -d '{
      "loan_amnt": 10000,
      "term": "36 months",
      "int_rate": 13.56,
      "installment": 339.31,
      "grade": "C",
      ...
    }'
```
Response:
```
{
  "default_probability": 0.7143,
  "prediction": "Default"
}
```
⚙️ Model Training

Train a baseline model:
```bash
python training/train.py
```

Train with hyperparameter tuning (Optuna):
```bash
python training/tuner_optuna.py
```

🔍 Model Evaluation
```bash
python evaluation/evaluate.py
```

🐳 Docker Deployment
```bash
docker build -t loan-api .
docker run -d -p 5000:5000 loan-api
```

🔥 Gunicorn + systemd Deployment

Systemd file:

```/etc/systemd/system/loan-api.service```

```bash
[Unit]
Description=Gunicorn instance to serve Loan Default API
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=/var/www/loan-prediction/app
Environment="PATH=/path/to/minicode/bin"
ExecStart=/path/to/minicode/bin/gunicorn \\
          --workers 1 \\
          --bind unix:/run/loan-api.sock \\
          app:app

[Install]
WantedBy=multi-user.target

```

🌐 Nginx Proxy Config

```/etc/nginx/sites-available/loan-api```
```bash
server {
    listen 80;
    server_name loan.ataxi.uz;

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/loan-api.sock;
    }
}

```

✅ Health Check
```curl http://loan.ataxi.uz/health```
📦 Requirements
```bash
Flask
gunicorn
xgboost
pandas
cudf
cuml
optuna
scikit-learn
```

📌 Note

**Make sure to run everything inside the rapids-ml conda environment.**
**Models are trained on GPU and optimized with Optuna for AUC performance.**