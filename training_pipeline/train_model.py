# -------------------------------------------------
# train_model.py - Clean version
# -------------------------------------------------

import os
import pickle
import logging
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

# -------------------------
# TensorFlow optional
# -------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# -------------------------------------------------
# Logging setup - simpler console output
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logging.info("TRAINING SCRIPT STARTED")

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# -------------------------------------------------
# Connect to MongoDB
# -------------------------------------------------
client = MongoClient(MONGO_URI)
db = client["aqi_database"]
data_col = db["training_features"]
model_col = db["model_registry"]
logging.info("Connected to MongoDB")

# -------------------------------------------------
# Load training data
# -------------------------------------------------
df = pd.DataFrame(list(data_col.find({}, {"_id": 0})))
if df.empty:
    raise Exception("No training data found")
logging.info(f"Training data shape: {df.shape}")

# -------------------------------------------------
# Features & target
# -------------------------------------------------
FEATURES = [
    "pm2_5", "pm10", "no2", "so2", "co", "o3", "nh3",
    "temperature", "humidity", "pressure",
    "windspeed", "winddirection", "precipitation",
    "hour", "day", "month", "day_of_week", "aqi_change"
]
TARGET = "aqi"

X = df[FEATURES]
y = df[TARGET]

# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
logging.info("Train-test split done")

# -------------------------------------------------
# Define models
# -------------------------------------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "RidgeRegression": Ridge(alpha=1.0)
}

# Optional: TensorFlow FeedForward NN
if TF_AVAILABLE:
    def create_nn_model(input_dim):
        model = Sequential([
            Dense(64, activation="relu", input_dim=input_dim),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    models["FeedForwardNN"] = create_nn_model(X_train.shape[1])
    logging.info("TensorFlow available: FeedForwardNN added")

# -------------------------------------------------
# Train & evaluate
# -------------------------------------------------
results = []

logging.info("\nTraining models...")
for name, model in models.items():
    logging.info(f"Training {name}...")
    try:
        if TF_AVAILABLE and name == "FeedForwardNN":
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            preds = model.predict(X_test).flatten()
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({"model_name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model})
        logging.info(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}\n")

    except Exception as e:
        logging.error(f"Error training {name}: {e}")

# -------------------------------------------------
# Best model selection
# -------------------------------------------------
best = min(results, key=lambda x: x["rmse"])
logging.info("BEST MODEL SELECTED")
logging.info(f"Model: {best['model_name']}")
logging.info(f"RMSE : {best['rmse']:.3f}")
logging.info(f"R²   : {best['r2']:.3f}")

# -------------------------------------------------
# Serialize model
# -------------------------------------------------
model_bytes = pickle.dumps(best["model"])

# -------------------------------------------------
# Store / version model in MongoDB
# -------------------------------------------------
existing_model = model_col.find_one({"model_name": best["model_name"]})
if existing_model:
    model_col.update_one(
        {"_id": existing_model["_id"]},
        {"$set": {
            "model_binary": model_bytes,
            "rmse": best["rmse"],
            "mae": best["mae"],
            "r2": best["r2"],
            "created_at": datetime.utcnow()
        }}
    )
    logging.info(f"Updated existing model in MongoDB: {best['model_name']}")
else:
    model_doc = {
        "model_name": best["model_name"],
        "created_at": datetime.utcnow(),
        "rmse": best["rmse"],
        "mae": best["mae"],
        "r2": best["r2"],
        "features": FEATURES,
        "model_binary": model_bytes
    }
    model_col.insert_one(model_doc)
    logging.info(f"Stored new model in MongoDB: {best['model_name']}")

logging.info("TRAINING PIPELINE COMPLETE")
