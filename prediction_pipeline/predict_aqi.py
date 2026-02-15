# -------------------------------------------------
# predict_aqi.py - REAL AQI FORECAST (Updated)
# -------------------------------------------------

import os
import pickle
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import requests

print("🚀 AQI PREDICTION SCRIPT STARTED")

# -------------------------
# AQI Formula (US EPA)
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
            return round((aqi_high - aqi_low) / (c_high - c_low) * (pm25 - c_low) + aqi_low)
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

features_col = db["training_features"]
model_col = db["model_registry"]
pred_col = db["predictions"]
weather_col = db["weather"]

# -------------------------
# Load best model
# -------------------------
model_doc = model_col.find_one({"rank": 1})
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
df_weather = pd.DataFrame(weather["hourly"])
df_weather.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "windspeed_10m": "windspeed",
    "winddirection_10m": "winddirection",
    "precipitation": "precipitation"
}, inplace=True)
df_weather["timestamp"] = pd.to_datetime(df_weather["time"])
df_weather.drop(columns=["time"], inplace=True)
df_weather["hour"] = df_weather["timestamp"].dt.hour
df_weather["day"] = df_weather["timestamp"].dt.day
df_weather["month"] = df_weather["timestamp"].dt.month
df_weather["day_of_week"] = df_weather["timestamp"].dt.dayofweek

# -------------------------
# Fetch latest pollutant data from training_features
# -------------------------
latest_pollutants = pd.DataFrame(list(features_col.find(
    {}, 
    {f: 1 for f in FEATURES if f not in ["hour","day","month","day_of_week"]} | {"timestamp":1},
    sort=[("timestamp",-1)]
)))
latest_pollutants = latest_pollutants.sort_values("timestamp").iloc[-1:]  # last row

# Repeat pollutant values for all forecast hours
for col in latest_pollutants.columns:
    if col != "timestamp":
        df_weather[col] = latest_pollutants.iloc[0][col]

# -------------------------
# Predict PM2.5 → AQI
# -------------------------
df_weather["predicted_pm25"] = model.predict(df_weather[FEATURES])
df_weather["predicted_aqi"] = df_weather["predicted_pm25"].apply(pm25_to_aqi)
df_weather["aqi_category"] = df_weather["predicted_aqi"].apply(aqi_category)

# -------------------------
# Store AQI forecast in MongoDB
# -------------------------
pred_col.delete_many({})
pred_col.insert_many(df_weather[["timestamp","predicted_pm25","predicted_aqi","aqi_category"]].to_dict("records"))

# -------------------------
# Store current weather for dashboard
# -------------------------
weather_col.delete_many({})
current_weather = df_weather.iloc[0]
weather_col.insert_one({
    "timestamp": pd.Timestamp(current_weather["timestamp"]).to_pydatetime(),
    "temp_c": float(current_weather["temperature"]),
    "humidity": float(current_weather["humidity"]),
    "pressure": float(current_weather["pressure"]),
    "windspeed": float(current_weather["windspeed"]),
    "winddirection": float(current_weather["winddirection"]),
    "precipitation": float(current_weather["precipitation"])
})

print("📈 AQI FORECAST COMPLETE")
print(df_weather[["timestamp","predicted_aqi","aqi_category"]].head(24*4))
print("🎉 AQI PREDICTION PIPELINE FINISHED SUCCESSFULLY")
