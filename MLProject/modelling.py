"""
modelling.py (versi Workflow CI)

Dipakai oleh MLflow Project agar training bisa dijalankan otomatis lewat
GitHub Actions setiap kali trigger dipantik (misal: setiap push).

Tracking URI sengaja tidak di-hardcode ke localhost supaya script ini bisa
jalan di lingkungan CI (GitHub Actions runner) yang tidak punya server
MLflow UI. Hasil run akan tersimpan sebagai folder ./mlruns yang nantinya
diupload sebagai artefak oleh workflow CI.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nama experiment diset lewat flag `--experiment-name` pada command
# `mlflow run` di ci.yml (bukan di sini) — memanggil mlflow.set_experiment()
# di dalam script akan bentrok dengan active run yang sudah dibuat oleh CLI.

train_df = pd.read_csv("diabetes_preprocessing/train.csv")
test_df = pd.read_csv("diabetes_preprocessing/test.csv")

X_train = train_df.drop(columns=["Outcome"])
y_train = train_df["Outcome"]
X_test = test_df.drop(columns=["Outcome"])
y_test = test_df["Outcome"]

# autolog: MLflow otomatis mencatat parameter, metrik, dan model
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="random_forest_basic"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Akurasi pada test set : {acc:.4f}")
    print(f"F1-score pada test set: {f1:.4f}")

print("Training selesai. Artefak MLflow tersimpan di ./mlruns")
