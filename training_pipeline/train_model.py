# -------------------------------------------------
# train_model.py - AQI PREDICTION PIPELINE
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

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.info("TRAINING SCRIPT STARTED")

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
    raise Exception("No training data found")

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
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------
# Models to train
# -------------------------
models = {
    "RidgeRegression": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

results = []

logging.info("Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Optional: cross-validation score to check overfitting
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

    results.append({
        "model_name": name,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_r2": cv_r2
    })

    logging.info(f"{name} → RMSE: {rmse:.2f} | R²: {r2:.2f} | CV R²: {cv_r2:.2f}")

# -------------------------
# Select best model based on RMSE
# -------------------------
best = min(results, key=lambda x: x["rmse"])
logging.info(f"BEST MODEL: {best['model_name']}")

# -------------------------
# Save best model to MongoDB
# -------------------------
model_col.delete_many({})  # keep only the latest model

model_col.insert_one({
    "model_name": best["model_name"],
    "created_at": datetime.utcnow(),
    "target": TARGET,
    "features": FEATURES,
    "rmse": best["rmse"],
    "mae": best["mae"],
    "r2": best["r2"],
    "cv_r2": best["cv_r2"],
    "model_binary": pickle.dumps(best["model"])
})

logging.info("MODEL STORED SUCCESSFULLY")
