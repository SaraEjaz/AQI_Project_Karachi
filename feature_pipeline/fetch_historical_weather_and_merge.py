# -------------------------------------------------
# fetch_historical_weather_and_merge.py
# -------------------------------------------------

print("🚀 SCRIPT STARTED")

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# -------------------------------------------------
# 1️⃣ Load environment variables
# -------------------------------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

LAT = 24.8607
LON = 67.0011
TIMEZONE = "Asia/Karachi"

# -------------------------------------------------
# 2️⃣ Connect to MongoDB
# -------------------------------------------------
client = MongoClient(MONGO_URI)
db = client["aqi_database"]

pollution_col = db["historical_pollutants"]
merged_col = db["training_features"]

print("✅ Connected to MongoDB")

# -------------------------------------------------
# 3️⃣ Load historical pollution data
# -------------------------------------------------
pollution_df = pd.DataFrame(list(pollution_col.find({}, {"_id": 0})))

if pollution_df.empty:
    raise Exception("❌ No pollution data found. Run pollution script first.")

pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"])
pollution_df = pollution_df.sort_values("timestamp")

print(f"✅ Loaded pollution data: {len(pollution_df)} rows")

# -------------------------------------------------
# 4️⃣ Determine date range for weather
# -------------------------------------------------
start_date = pollution_df["timestamp"].min().date()
end_date = pollution_df["timestamp"].max().date()

print(f"📅 Weather range: {start_date} → {end_date}")

# -------------------------------------------------
# 5️⃣ Fetch historical weather from Open-Meteo
# -------------------------------------------------
weather_url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={start_date}"
    f"&end_date={end_date}"
    "&hourly=temperature_2m,relative_humidity_2m,"
    "pressure_msl,windspeed_10m,winddirection_10m,precipitation"
    f"&timezone={TIMEZONE}"
)

print("🌤️ Fetching historical weather...")
response = requests.get(weather_url)

if response.status_code != 200:
    raise Exception("❌ Weather API failed")

weather_data = response.json()

weather_df = pd.DataFrame(weather_data["hourly"])
weather_df["timestamp"] = pd.to_datetime(weather_df["time"])
weather_df.drop(columns=["time"], inplace=True)

weather_df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "windspeed_10m": "windspeed",
    "winddirection_10m": "winddirection",
    "precipitation": "precipitation"
}, inplace=True)

print(f"✅ Weather rows fetched: {len(weather_df)}")

# -------------------------------------------------
# 6️⃣ Merge pollution + weather
# -------------------------------------------------
merged_df = pd.merge(
    pollution_df,
    weather_df,
    on="timestamp",
    how="inner"
)

print(f"🔗 Merged rows: {len(merged_df)}")

# -------------------------------------------------
# 7️⃣ Feature engineering
# -------------------------------------------------
merged_df["hour"] = merged_df["timestamp"].dt.hour
merged_df["day"] = merged_df["timestamp"].dt.day
merged_df["month"] = merged_df["timestamp"].dt.month
merged_df["day_of_week"] = merged_df["timestamp"].dt.dayofweek

merged_df = merged_df.sort_values("timestamp")
merged_df["aqi_change"] = merged_df["aqi"].diff().fillna(0)

print("🧠 Derived features added")

print("📊 Sample rows:")
print(merged_df.head())

# -------------------------------------------------
# 8️⃣ Store merged training data
# -------------------------------------------------
existing_ts = set(
    doc["timestamp"]
    for doc in merged_col.find({}, {"timestamp": 1})
)

final_df = merged_df[~merged_df["timestamp"].isin(existing_ts)]

if not final_df.empty:
    merged_col.insert_many(final_df.to_dict("records"))
    print(f"✅ Stored {len(final_df)} training records in MongoDB")
else:
    print("⚠️ No new records to store")

print("🎉 FEATURE PIPELINE COMPLETE")
