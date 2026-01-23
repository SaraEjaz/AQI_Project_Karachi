# ---------------------------
# fetch_historical_pollution.py
# ---------------------------

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import time

# ---------------------------
# Load API key & MongoDB URI
# ---------------------------
load_dotenv()
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")
MONGO_URI = os.getenv("MONGO_URI")

LAT = 24.8607
LON = 67.0011

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["aqi_database"]
features_col = db["historical_pollutants"]

# ---------------------------
# Compute start & end timestamps (last 3 months)
# ---------------------------
start_date = datetime.utcnow() - timedelta(days=90)  # 3 months back
end_date = datetime.utcnow()

start_unix = int(time.mktime(start_date.timetuple()))
end_unix = int(time.mktime(end_date.timetuple()))

print(f"Fetching historical pollutants from {start_date} to {end_date}")

# ---------------------------
# Fetch historical pollutants
# ---------------------------
url = (
    f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
    f"lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={AIR_POLLUTION_API}"
)

response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch historical data: {response.status_code} {response.text}")

data = response.json()
print(f"✅ Total records fetched: {len(data['list'])}")

# ---------------------------
# Convert to DataFrame
# ---------------------------
rows = []
for item in data["list"]:
    ts = datetime.utcfromtimestamp(item["dt"])
    row = {
        "timestamp": ts,
        "pm2_5": item["components"].get("pm2_5"),
        "pm10": item["components"].get("pm10"),
        "no2": item["components"].get("no2"),
        "so2": item["components"].get("so2"),
        "co": item["components"].get("co"),
        "o3": item["components"].get("o3"),
        "nh3": item["components"].get("nh3"),
        "aqi": item["main"].get("aqi")
    }
    rows.append(row)

df = pd.DataFrame(rows)
print("Sample rows:")
print(df.head())

# ---------------------------
# Store in MongoDB
# ---------------------------
# Remove duplicates if already exists
existing_timestamps = set([doc["timestamp"] for doc in features_col.find({}, {"timestamp":1})])
df_to_store = df[~df["timestamp"].isin(existing_timestamps)]

if not df_to_store.empty:
    features_col.insert_many(df_to_store.to_dict("records"))
    print(f"✅ Stored {len(df_to_store)} historical pollutant records in MongoDB")
else:
    print("⚠️ No new records to store")
