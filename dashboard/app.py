import os 
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import plotly.graph_objects as go
from datetime import date

# -----------------------------
# PAGE CONFIG (Dark Theme)
# -----------------------------
st.set_page_config(
    page_title="🌟 Karachi AQI Dashboard",
    layout="wide"
)



st.markdown("""
    <style>
    .stApp {
        background-color:#0A0A0A;
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Today Card */
    .metric-card {
        background: linear-gradient(135deg, #FF3B3B, #FF0000); /* bright red */
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.3s;
        border: 1.5px solid #FF0000;
    }
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px #FF0000; /* red glow on hover */
    }

    /* Next 3 Day Prediction Cards */
    .card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #00FFFF, #007BFF); /* cyan blue gradient */
        color: white;
        text-align: center;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.3s;
        border: 1.5px solid #00FFFF;
    }
    .card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 3px #00FFFF; /* cyan glow on hover */
    }

    h1 {
        font-size: 3em;
    }

    .tips {
    background: rgba(255, 165, 0, 0.5); /* orange with 30% opacity */
    padding: 30px;
    border-radius: 6px;
    color: white; /* text contrasts with orange */
    border: 1.5px solid #FFA500; /* neon-style orange border */
    box-shadow: 0 0 2px #FFA500; /* glowing neon effect */
    font-weight: bold;
    transition: transform 0.2s, box-shadow 0.3s;
}
</style>

""", unsafe_allow_html=True)

# -----------------------------
# LOAD ENV & DB
# -----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_database"]
pred_col = db["predictions"]
pollution_col = db["historical_pollutants"]
weather_col = db["weather"]
model_col = db["model_registry"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_predictions():
    df = pd.DataFrame(list(pred_col.find({}, {"_id": 0})))
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")

def load_latest_weather():
    return weather_col.find_one({}, {"_id":0}, sort=[("timestamp",-1)])

def aqi_label(aqi):
    if aqi <= 50: return "Good 😄"
    if aqi <= 100: return "Moderate 🙂"
    if aqi <= 150: return "Unhealthy for Sensitive Groups 😷"
    if aqi <= 200: return "Unhealthy 🤒"
    if aqi <= 300: return "Very Unhealthy 😨"
    return "Hazardous 🚨"

# -----------------------------
# TITLE (Top-left)
# -----------------------------
st.markdown("<h1 style='text-align:left;color:white;'>🌤️ Karachi AQI Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_predictions()
if df is None:
    st.warning("⚠️ No Prediction Data Found")
    st.stop()

latest_aqi = int(df.iloc[-1]["predicted_aqi"])
category = aqi_label(latest_aqi)
today = date.today()

# -----------------------------
# TOP AQI CARDS
# -----------------------------
st.markdown("### 📊 AQI Overview")
col1, col2, col3, col4 = st.columns(4)

df["date"] = df["timestamp"].dt.date
daily = df.groupby("date")["predicted_aqi"].mean().reset_index()
next_days = daily[daily["date"] > today].head(3)

with col1:
    st.markdown(f"<div class='metric-card'><h3>Today</h3><h2>{latest_aqi}</h2><p>{category}</p></div>", unsafe_allow_html=True)

for i in range(3):
    if i < len(next_days):
        val = int(next_days.iloc[i]["predicted_aqi"])
        cat = aqi_label(val)
        with [col2, col3, col4][i]:
            st.markdown(f"<div class='card'><h4>{next_days.iloc[i]['date']}</h4><h2>{val}</h2><p>{cat}</p></div>", unsafe_allow_html=True)

# -----------------------------
# WEATHER INFO
# -----------------------------
st.markdown("### 🌦️ Weather Conditions")
weather = load_latest_weather()
if weather:
    w1, w2, w3 = st.columns(3)
    w1.metric("🌡️ Temp (°C)", weather.get("temp_c", "-"))
    w2.metric("💧 Humidity (%)", weather.get("humidity", "-"))
    w3.metric("🔴 Pressure (hPa)", weather.get("pressure", "-"))
else:
    st.info("No Weather Data Available")

# -----------------------------
# AQI BAR GRAPH + MAP
# -----------------------------
st.markdown("### 📊 AQI Forecast & Map")
col_left, col_right = st.columns([2,1])

with col_left:
    fig_bar_aqi = go.Figure()
    fig_bar_aqi.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["predicted_aqi"],
        marker=dict(
            color=df["predicted_aqi"],
            colorscale="Turbo",
            showscale=True
        ),
        name="AQI"
    ))
    fig_bar_aqi.update_layout(
        template="plotly_dark",
        height=350,
        title="AQI Forecast (Bar View)",
        xaxis_title="Time",
        yaxis_title="Predicted AQI",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_bar_aqi, use_container_width=True)

with col_right:
    map_fig = go.Figure(go.Scattermapbox(
        lat=[24.8607],
        lon=[67.0011],
        mode='markers+text',
        marker=dict(size=20, color="red"),
        text=[f"Karachi - AQI: {latest_aqi} 🌆"],
        textposition="top right"
    ))
    map_fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=10,
        mapbox_center={"lat":24.8607, "lon":67.0011},
        margin=dict(l=0,r=0,t=0,b=0),
        height=350
    )
    st.plotly_chart(map_fig, use_container_width=True)

# -----------------------------
# POLLUTANT PIE + AQI SPEEDOMETER
# -----------------------------
st.markdown("### 🏭 Pollutant Concentration & AQI Gauge")
pol_df = pd.DataFrame(list(pollution_col.find({}, {"_id":0})))

if not pol_df.empty:
    pol_df["timestamp"] = pd.to_datetime(pol_df["timestamp"])
    latest_pollutants = pol_df.iloc[-1][["pm2_5","pm10","no2","so2","co","o3","nh3"]].fillna(0)
    labels = [col.upper() for col in latest_pollutants.index]
    values = latest_pollutants.values

    col1, col2 = st.columns(2)

    # PIE CHART
    with col1:
        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(line=dict(color='white', width=2))
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            height=400,
            title="Pollutant Concentration 🏭",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # SPEEDOMETER GAUGE
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_aqi,
            number={'suffix': " AQI"},
            title={'text': "Current AQI 🌡️"},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0,50], 'color':'green'},
                    {'range':[51,100], 'color':'yellow'},
                    {'range':[101,150], 'color':'orange'},
                    {'range':[151,200], 'color':'red'},
                    {'range':[201,300], 'color':'purple'},
                    {'range':[301,500], 'color':'maroon'},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': latest_aqi
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------
# TOP MODEL COMPARISON (Slim Bars + Pattern for Best Model)
# -----------------------------
st.markdown("### 🤖 Top Model Comparison")

models_df = pd.DataFrame(list(model_col.find({"rank":{"$lte":2}},{"_id":0})))
if not models_df.empty:
    models_df = models_df.sort_values("rank")
    metrics = ["rmse","mae","r2"]
    best_model_name = models_df.iloc[0]["model_name"]
    
    col1, col2, col3 = st.columns(3)
    
    # RMSE chart
    with col1:
        fig_rmse = go.Figure()
        for i, row in models_df.iterrows():
            fig_rmse.add_trace(go.Bar(
                x=[row["model_name"]],
                y=[row["rmse"]],
                name=row["model_name"],
                marker=dict(
                    color="#14E2E9" if row["model_name"] != best_model_name else "#E7393F",
                    pattern_shape="\\"
                ),
                width=0.3  # slim bar
            ))
        fig_rmse.update_layout(
            template="plotly_dark",
            title="RMSE",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # MAE chart
    with col2:
        fig_mae = go.Figure()
        for i, row in models_df.iterrows():
            fig_mae.add_trace(go.Bar(
                x=[row["model_name"]],
                y=[row["mae"]],
                name=row["model_name"],
                marker=dict(
                    color="#14E2E9" if row["model_name"] != best_model_name else "#E7393F",
                    pattern_shape="\\"
                ),
                width=0.3
            ))
        fig_mae.update_layout(
            template="plotly_dark",
            title="MAE",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # R² chart
    with col3:
        fig_r2 = go.Figure()
        for i, row in models_df.iterrows():
            fig_r2.add_trace(go.Bar(
                x=[row["model_name"]],
                y=[row["r2"]],
                name=row["model_name"],
                marker=dict(
                    color="#14E2E9" if row["model_name"] != best_model_name else "#E7393F",
                    pattern_shape="\\"
                ),
                width=0.3
            ))
        fig_r2.update_layout(
            template="plotly_dark",
            title="R² Score",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    best = models_df.iloc[0]
    st.success(f"🏆 Best Model: {best['model_name']} | RMSE: {best['rmse']:.2f} | MAE: {best['mae']:.2f} | R²: {best['r2']:.2f}")



# -----------------------------
# AQI TIPS
# -----------------------------
st.markdown("### 🛡️ AQI Protection Tips")
st.markdown("""
<div class="tips">
<ul>
<li>😷 Wear a mask on high AQI days</li>
<li>🚴 Avoid outdoor activities during peak pollution</li>
<li>🏠 Use air purifiers indoors</li>
<li>💧 Stay hydrated & eat antioxidant-rich foods</li>
</ul>
</div>
""", unsafe_allow_html=True)
