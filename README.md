AQI Prediction System (MongoDB Version)

A system to forecast Air Quality Index (AQI) using historical pollution and weather data for Karachi, Pakistan.
Built with Python, MongoDB, and machine learning models including XGBoost, Random Forest, and Gradient Boosting.

Setup:
1. Install MongoDB Atlas and start the service.
2. Create a .env file with the following:
   MONGO_URI=mongodb://localhost:27017
   AIR_POLLUTION_API=your_openweathermap_api_key
3. Install Python dependencies:
   pip install -r requirements.txt

How to Run:

1. Feature Pipeline:
   - Fetch historical pollution and weather data, merge them, and generate features.
   - Run:
     python feature_pipeline/fetch_historical_pollution.py
     python feature_pipeline/fetch_historical_weather_and_merge.py

2. Training Pipeline:
   - Train machine learning models and store the top models in MongoDB.
   - Run:
     python training_pipeline/train_model.py

3. Prediction Pipeline:
   - Fetch latest weather, predict PM2.5 and AQI, and store forecasts in MongoDB.
   - Run:
     python prediction_pipeline/predict_aqi.py

Project Structure:

aqi_project/
├─ feature_pipeline/
│  ├─ fetch_features.py
│  └─ fetch_historical_weather_and_merge.py
├─ models/
│  └─ aqi_model.pkl
├─ notebooks/
│  ├─ 01_feature_pipeline.ipynb
│  ├─ 02_training_pipeline.ipynb
│  └─ 03_dashboard.ipynb
├─ prediction_pipeline/
│  └─ predict_aqi.py
├─ training_pipeline/
│  └─ train_model.py
├─ README.md
└─ requirements.txt

Notes:
- Ensure MongoDB is running before executing pipelines.
- The system is configured for Karachi, Pakistan. Update LAT/LON in scripts for other locations.
- A valid OpenWeatherMap API key is required for historical pollution data.
