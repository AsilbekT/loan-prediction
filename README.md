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

## 📂 Project Structure

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
├── gunicorn.service        # systemd service file
├── Dockerfile (optional)
└── README.md               # This file

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
Environment="PATH=/root/miniconda3/envs/rapids-ml/bin"
ExecStart=/root/miniconda3/envs/rapids-ml/bin/gunicorn \\
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