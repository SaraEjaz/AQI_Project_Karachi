# -------------------------------------------------
# predict_aqi.py - REAL AQI FORECAST
# -------------------------------------------------

import os
import pickle
import requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

print("🚀 AQI PREDICTION SCRIPT STARTED")

# -------------------------
# AQI FORMULA (US EPA)
# -------------------------
def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            )
    return None

def aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

# -------------------------
# Env + Mongo
# -------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_database"]

features_col = db["training_features"]    # Historical training data
model_col = db["model_registry"]          # Stored ML model
pred_col = db["predictions"]              # AQI forecast
weather_col = db["weather"]               # Current weather info for dashboard

# -------------------------
# Load best model
# -------------------------
model_doc = model_col.find_one()
model = pickle.loads(model_doc["model_binary"])
FEATURES = model_doc["features"]

print(f"🏆 Loaded model: {model_doc['model_name']}")

# -------------------------
# Fetch latest weather forecast (4 days)
# -------------------------
LAT, LON = 24.8607, 67.0011
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
    f"windspeed_10m,winddirection_10m,precipitation"
    f"&forecast_days=4&timezone=Asia/Karachi"
)

weather = requests.get(url).json()
df = pd.DataFrame(weather["hourly"])

df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "windspeed_10m": "windspeed",
    "winddirection_10m": "winddirection",
    "precipitation": "precipitation"
}, inplace=True)

df["timestamp"] = pd.to_datetime(df["time"])
df.drop(columns=["time"], inplace=True)

# Add features for model
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

# -------------------------
# Predict PM2.5 → AQI
# -------------------------
df["predicted_pm25"] = model.predict(df[FEATURES])
df["predicted_aqi"] = df["predicted_pm25"].apply(pm25_to_aqi)
df["aqi_category"] = df["predicted_aqi"].apply(aqi_category)

# -------------------------
# Store AQI forecast in MongoDB
# -------------------------
pred_col.delete_many({})  # Clear old forecast
pred_col.insert_many(
    df[["timestamp", "predicted_pm25", "predicted_aqi", "aqi_category"]].to_dict("records")
)

# -------------------------
# Store current weather for dashboard
# -------------------------
weather_col.delete_many({})  # keep only latest
current_weather = df.iloc[0]

weather_col.insert_one({
    "timestamp": pd.Timestamp(current_weather["timestamp"]).to_pydatetime(),  # ensure datetime
    "temp_c": float(current_weather["temperature"]),
    "humidity": float(current_weather["humidity"]),
    "pressure": float(current_weather["pressure"]),
    "windspeed": float(current_weather["windspeed"]),
    "winddirection": float(current_weather["winddirection"]),
    "precipitation": float(current_weather["precipitation"])
})


print("📈 AQI FORECAST COMPLETE")
print("\n🔮 NEXT 4 DAYS AQI (Hourly):")
print(df[["timestamp", "predicted_aqi", "aqi_category"]].head(24*4))  # show first 24h of each day

print("🎉 AQI PREDICTION PIPELINE FINISHED SUCCESSFULLY")
