# -------------------------------------------------
# train_model.py - ADVANCED AQI TRAINING PIPELINE
# -------------------------------------------------

import os
import pickle
import logging
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor   # ✅ NEW

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.info("🚀 TRAINING SCRIPT STARTED")

# -------------------------
# Env + MongoDB
# -------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_database"]
data_col = db["training_features"]
model_col = db["model_registry"]

# -------------------------
# Load data
# -------------------------
df = pd.DataFrame(list(data_col.find({}, {"_id": 0})))
if df.empty:
    raise Exception("❌ No training data found")

logging.info(f"Training data shape: {df.shape}")

# -------------------------
# Features & Target
# -------------------------
FEATURES = [
    "temperature", "humidity", "pressure",
    "windspeed", "winddirection", "precipitation",
    "hour", "day", "month", "day_of_week"
]
TARGET = "pm2_5"

X = df[FEATURES]
y = df[TARGET]

# -------------------------
# Train/Test Split (Time-Series Safe)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------
# Models to Train
# -------------------------
models = {
    "RidgeRegression": Ridge(alpha=1.0),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),

    # ✅ NEW — XGBoost (STRONG MODEL)
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

results = []

logging.info("\n📊 Training Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    results.append({
        "model_name": name,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_r2": cv_r2
    })

    logging.info(
        f"{name:<18} | RMSE: {rmse:8.3f} | MAE: {mae:8.3f} | R²: {r2:6.3f} | CV R²: {cv_r2:6.3f}"
    )

# -------------------------
# Sort Models by RMSE
# -------------------------
results_sorted = sorted(results, key=lambda x: x["rmse"])

best_model = results_sorted[0]
second_best = results_sorted[1]

logging.info("\n🏆 MODEL RANKING:")
for i, res in enumerate(results_sorted):
    logging.info(f"{i+1}. {res['model_name']} (RMSE: {res['rmse']:.3f})")

logging.info(f"\n🥇 BEST MODEL: {best_model['model_name']}")
logging.info(f"🥈 SECOND BEST: {second_best['model_name']}")

# -------------------------
# Store TOP 2 models in MongoDB
# -------------------------
model_col.delete_many({})

for rank, model_data in enumerate([best_model, second_best], start=1):
    model_col.insert_one({
        "model_name": model_data["model_name"],
        "rank": rank,
        "created_at": datetime.utcnow(),
        "target": TARGET,
        "features": FEATURES,
        "rmse": model_data["rmse"],
        "mae": model_data["mae"],
        "r2": model_data["r2"],
        "cv_r2": model_data["cv_r2"],
        "model_binary": pickle.dumps(model_data["model"])
    })

logging.info("\n✅ TOP 2 MODELS STORED SUCCESSFULLY")
logging.info("🎯 TRAINING PIPELINE FINISHED")
