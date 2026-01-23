print("🚀 AQI PREDICTION SCRIPT STARTED")

import os
import requests
import pandas as pd
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

LAT = 24.8607
LON = 67.0011

# ---------------------------
# Connect to MongoDB
# ---------------------------
client = MongoClient(MONGO_URI)
db = client["aqi_database"]

features_col = db["features"]
model_registry_col = db["model_registry"]
predictions_col = db["predictions"]

print("✅ Connected to MongoDB")

# ---------------------------
# Load best trained model
# ---------------------------
model_doc = model_registry_col.find_one(sort=[("rmse", 1)])

if not model_doc:
    raise Exception("❌ No trained model found in MongoDB")

model = pickle.loads(model_doc["model_binary"])
feature_columns = model_doc["features"]

print(f"🏆 Loaded model: {model_doc['model_name']}")

# ---------------------------
# AQI category function
# ---------------------------
def aqi_category(aqi):
    if aqi <= 1.5:
        return "Good"
    elif aqi <= 2.5:
        return "Moderate"
    elif aqi <= 3.5:
        return "Poor"
    else:
        return "Very Poor"

# ---------------------------
# Get latest pollution values
# ---------------------------
latest_pollution = features_col.find_one(sort=[("timestamp", -1)])

pollutants = {
    "pm2_5": latest_pollution["pm2_5"],
    "pm10": latest_pollution["pm10"],
    "no2": latest_pollution["no2"],
    "so2": latest_pollution["so2"],
    "co": latest_pollution["co"],
    "o3": latest_pollution["o3"],
    "nh3": latest_pollution.get("nh3", 0)
}

print("🌫️ Using latest pollution values")

# ---------------------------
# Fetch 3-day weather forecast
# ---------------------------
weather_url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
    f"windspeed_10m,winddirection_10m,precipitation"
    f"&forecast_days=3"
    f"&timezone=Asia/Karachi"
)

weather_data = requests.get(weather_url).json()
weather_df = pd.DataFrame(weather_data["hourly"])

weather_df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "windspeed_10m": "windspeed",
    "winddirection_10m": "winddirection",
    "precipitation": "precipitation"
}, inplace=True)

weather_df["timestamp"] = pd.to_datetime(weather_df["time"])
weather_df.drop(columns=["time"], inplace=True)

print(f"🌤️ Weather rows fetched: {len(weather_df)}")

# ---------------------------
# Combine pollution + weather
# ---------------------------
for col, val in pollutants.items():
    weather_df[col] = val

weather_df["hour"] = weather_df["timestamp"].dt.hour
weather_df["day"] = weather_df["timestamp"].dt.day
weather_df["month"] = weather_df["timestamp"].dt.month
weather_df["day_of_week"] = weather_df["timestamp"].dt.dayofweek

weather_df["aqi_change"] = 0  # future assumption

# ---------------------------
# Predict AQI
# ---------------------------
X_pred = weather_df[feature_columns]
weather_df["predicted_aqi"] = model.predict(X_pred)
weather_df["aqi_category"] = weather_df["predicted_aqi"].apply(aqi_category)

print("📈 AQI prediction completed")

# ---------------------------
# Store predictions
# ---------------------------
records = weather_df[
    ["timestamp", "predicted_aqi", "aqi_category"]
].to_dict("records")

predictions_col.insert_many(records)

print(f"✅ Stored {len(records)} AQI predictions in MongoDB")

# ---------------------------
# Display results
# ---------------------------
print("\n🔮 NEXT 3 DAYS AQI FORECAST FOR KARACHI (Hourly):")
print(weather_df[
    ["timestamp", "predicted_aqi", "aqi_category"]
].head(24))

print("\n🎉 AQI PREDICTION PIPELINE FINISHED SUCCESSFULLY")
