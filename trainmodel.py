# train_model.py
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

start_time = time.time()
print("=" * 60, flush=True)
print("⚡ STARTING HIGH-SPEED AQI MODEL TRAINING PIPELINE", flush=True)
print("=" * 60, flush=True)

# 1. Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

csv_path = os.path.join(BASE_DIR, "aqi.csv")

# Generate dataset fast if missing
if not os.path.exists(csv_path):
    print("⚠️  'aqi.csv' missing. Rapidly generating sample dataset...", flush=True)
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    cities = ["Mumbai Central", "Pune Urban", "Central Delhi", "Bengaluru East", "Chennai North"]
    data = []
    for c in cities:
        data.append(pd.DataFrame({
            "date": dates.strftime("%d-%m-%Y"),
            "state": "Sample State",
            "area": c,
            "number_of_monitoring_stations": np.random.randint(2, 12, len(dates)),
            "prominent_pollutants": np.random.choice(["PM2.5", "PM10", "NO2", "SO2"], len(dates)),
            "aqi_value": np.random.randint(30, 380, len(dates)),
            "air_quality_status": None
        }))
    pd.concat(data, ignore_index=True).to_csv(csv_path, index=False)
    print("✅ Created dataset with 1,825 records.", flush=True)

# 2. Fast Data Vectorization & Cleaning
print("📊 [1/4] Vectorizing dataset & cleaning nulls...", flush=True)
df = pd.read_csv(csv_path)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["aqi_value"] = pd.to_numeric(df["aqi_value"], errors="coerce")
df["number_of_monitoring_stations"] = pd.to_numeric(
    df["number_of_monitoring_stations"], errors="coerce"
).fillna(1)
df = df.dropna(subset=["aqi_value", "date"]).reset_index(drop=True)

# Fast NumPy Vectorized Status Assignment
vals = df["aqi_value"].values
conds = [vals <= 50, vals <= 100, vals <= 200, vals <= 300, vals <= 400]
choices = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor"]
df["air_quality_status"] = np.select(conds, choices, default="Severe")

df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# Label Encoding
encoders = {}
for col in ["state", "area", "prominent_pollutants", "air_quality_status"]:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

feature_cols = [
    "area_enc", "number_of_monitoring_stations", 
    "prominent_pollutants_enc", "month", "day", "dayofweek"
]
X = df[feature_cols]

# 3. Parallel Classifier Training
print("🤖 [2/4] Training Random Forest Classifier (All CPU Cores)...", flush=True)
y_cls = df["air_quality_status_enc"]
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(
    n_estimators=30, max_depth=12, n_jobs=-1, random_state=42
).fit(X_tr_c, y_tr_c)

cls_acc = accuracy_score(y_te_c, clf_model.predict(X_te_c))
cls_f1 = f1_score(y_te_c, clf_model.predict(X_te_c), average="weighted", zero_division=0)

# 4. Parallel Regressor Training
print("📈 [3/4] Training Random Forest Regressor (All CPU Cores)...", flush=True)
y_reg = df["aqi_value"]
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(
    n_estimators=30, max_depth=12, n_jobs=-1, random_state=42
).fit(X_tr_r, y_tr_r)

reg_pred = reg_model.predict(X_te_r)
reg_rmse = np.sqrt(mean_squared_error(y_te_r, reg_pred))
reg_r2 = r2_score(y_te_r, reg_pred)

# 5. Fit Scaler & Save All Model Files
print("💾 [4/4] Saving model artifacts to 'models/'...", flush=True)
area_profiles = df.groupby("area").agg(
    avg_aqi=("aqi_value", "mean"),
    max_aqi=("aqi_value", "max"),
    avg_stations=("number_of_monitoring_stations", "mean")
).reset_index()

scaler = StandardScaler().fit(area_profiles[["avg_aqi", "max_aqi", "avg_stations"]])

metrics_summary = {
    "cls_acc": cls_acc,
    "cls_f1": cls_f1,
    "reg_rmse": reg_rmse,
    "reg_r2": reg_r2
}

joblib.dump(clf_model, os.path.join(MODEL_DIR, "classifier.joblib"))
joblib.dump(reg_model, os.path.join(MODEL_DIR, "regressor.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
joblib.dump(metrics_summary, os.path.join(MODEL_DIR, "metrics.joblib"))

elapsed = round(time.time() - start_time, 2)
print("=" * 60, flush=True)
print(f"🎉 TRAINING COMPLETE IN {elapsed} SECONDS!", flush=True)
print(f"📊 Accuracy: {cls_acc * 100:.2f}% | Regressor RMSE: {reg_rmse:.2f} AQI Units", flush=True)
print("=" * 60, flush=True)