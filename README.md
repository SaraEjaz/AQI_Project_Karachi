# AQI Prediction System (MongoDB version)

## Setup
1. Install MongoDB locally
2. Start MongoDB service
3. Create `.env` with API keys and Mongo URI
4. Install Python dependencies:


## How to Run
1. Feature Pipeline:

2. Training Pipeline:

3. Open notebooks for evaluation:
- 01_feature_pipeline.ipynb
- 02_training_pipeline.ipynb
- 03_dashboard.ipynb

```
aqi_project
├─ feature_pipeline
│  ├─ fetch_features.py
│  └─ fetch_historical_weather_and_merge.py
├─ models
│  └─ aqi_model.pkl
├─ notebooks
│  ├─ 04_eda.ipynb
│  └─ 05_preprocessing.ipynb
├─ prediction_pipeline
│  └─ predict_aqi.py
├─ README.md
├─ requirements.txt
├─ training_pipeline
│  └─ train_model.py
└─ training_pipeline.log

```