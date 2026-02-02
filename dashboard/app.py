import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# -----------------------------
# LOAD ENV & DB
# -----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["aqi_database"]
pred_col = db["predictions"]
pollution_col = db["historical_pollutants"]
weather_col = db.get_collection("weather")

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
    latest = weather_col.find_one({}, {"_id":0}, sort=[("timestamp",-1)])
    if not latest:
        return None
    return latest

def aqi_label(aqi):
    if aqi <= 50: return "Good 😄"
    if aqi <= 100: return "Moderate 🙂"
    if aqi <= 150: return "Unhealthy for Sensitive Groups 😷"
    if aqi <= 200: return "Unhealthy 🤒"
    if aqi <= 300: return "Very Unhealthy 😨"
    return "Hazardous 🚨"

def gradient_card_style(start="#1e3c72", end="#2a5298", width="25%", height="120px", **kwargs):
    style = {
        "background": f"linear-gradient(135deg, {start}, {end}, 0.8)",  # slightly transparent
        "padding": "25px",
        "border-radius": "12px",
        "box-shadow": "0 0 12px rgba(255,255,255,0.2)",
        "border": "1px solid rgba(255,255,255,0.2)",
        "color": "white",
        "text-align": "left",
        "width": width,
        "height": height,
        "font-size": "1rem",
        "display":"flex",
        "flex-direction":"column",
        "justify-content":"center",
        "align-items":"flex-start",
        "backdrop-filter": "blur(6px)",   # adds slight blur behind card
        "background-color": "rgba(30,60,114,0.6)"  # fallback transparent background
    }
    style.update(kwargs)
    return style


# -----------------------------
# DASH APP
# -----------------------------
app = dash.Dash(__name__)
app.title = "Karachi AQI Dashboard"

# -----------------------------
# GLOBAL DARK THEME
# -----------------------------
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {background-color:#0F172A; color:#E5E7EB; font-family: Arial,sans-serif;}
            h1,h2,h3,h4 {color:#F8FAFC; margin:0;}
            .cards-row {
                display:flex;
                justify-content:space-between;
                margin-bottom:3px;
                gap:10px;
                flex-wrap: nowrap;
            }
            .chart-box {
                background:#111827;
                padding:10px;
                border-radius:12px;
                box-shadow:0 0 12px rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.2);
            }
        </style>
    </head>
    <body style="margin:0; padding:0; overflow-y: auto; height: 100vh;">

    <!-- Background Video -->
    <video autoplay muted loop id="bg-video" style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
    ">
        <source src="/assets/background.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>

    <!-- Dashboard content -->
    <div id="dash-container" style="position: relative; z-index: 1;">
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </div>
</body>


</html>
"""

# -----------------------------
# LAYOUT
# -----------------------------
app.layout = html.Div([
    html.Div([
        html.H1("🌤️ Karachi AQI Forecast", style={"text-align":"center", "margin-bottom":"10px"}),

        dcc.Interval(id="refresh", interval=60*1000, n_intervals=0),

        html.Div(id="iqair-cards", className="cards-row"),
        html.Div(id="top-cards", className="cards-row"),

        html.Div([
            html.Div(dcc.Graph(id="aqi-line"), className="chart-box", style={"width":"60%", "height":"250px"}),
            html.Div(dcc.Graph(id="karachi-map"), className="chart-box", style={"width":"38%", "height":"250px"}),
        ], className="cards-row"),

        html.Div([
    html.Div(
        dcc.Graph(id="pollutant-trends"),
        className="chart-box",
        style={"width":"60%", "height":"250px"}  # 3/5 part
    ),
    html.Div(
        id="info-card",
        className="chart-box",
        style={
            "width":"40%",   # 2/5 part
            "height":"266.5px",
            "padding":"0",
            "overflow":"hidden"
        }
    ),
], className="cards-row"),

    ], style={"padding":"10px"})
])

# -----------------------------
# CALLBACKS
# -----------------------------
@app.callback(
    [
        Output("top-cards","children"),
        Output("karachi-map","figure"),
        Output("aqi-line","figure"),
        Output("iqair-cards","children"),
        Output("pollutant-trends","figure"),
        Output("info-card","children"),
    ],
    Input("refresh","n_intervals")
)
def update_dashboard(_):
    df = load_predictions()
    if df is None:
        return ["No Data"]*6

    today = date.today()
    latest_aqi = int(df.iloc[0]["predicted_aqi"])
    category = aqi_label(latest_aqi)
    latest_weather = load_latest_weather()

    cards = []

    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["predicted_aqi"].mean().reset_index()

    iqair_cards = []
    iqair_cards.append(
        html.Div([html.H4("Today 📅"), html.H2(f"{latest_aqi}"), html.P(category)],
                 style=gradient_card_style("#11998e","#38ef7d"))
    )

    next_days = daily[daily["date"] > today].head(3)

    for i in range(3):
        if i < len(next_days):
            row = next_days.iloc[i]
            val = int(row["predicted_aqi"])
            cat = aqi_label(val)
            iqair_cards.append(
                html.Div([html.H4(str(row["date"])), html.H2(f"{val}"), html.P(cat)],
                         style=gradient_card_style("#6a11cb","#2575fc"))
            )
        else:
            iqair_cards.append(
                html.Div([html.H4("N/A"), html.H2("-"), html.P("-")],
                         style=gradient_card_style("#444","#777"))
            )

    line_fig = px.line(df, x="timestamp", y="predicted_aqi", markers=True, template="plotly_dark",
                       title="Hourly AQI Forecast")
    line_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), height=250)

    map_fig = px.scatter_mapbox(
        pd.DataFrame([{"lat":24.8607,"lon":67.0011,"AQI":latest_aqi}]),
        lat="lat", lon="lon", size="AQI", color="AQI",
        size_max=15, zoom=10, mapbox_style="carto-darkmatter",
        hover_name="AQI"
    )
    map_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), height=250)

    pol_trends = pd.DataFrame(list(pollution_col.find({}, {"_id":0})))
    if not pol_trends.empty:
        pol_trends["timestamp"] = pd.to_datetime(pol_trends["timestamp"])
        pol_fig = px.line(pol_trends, x="timestamp",
                          y=["pm2_5","pm10","no2","so2","co","o3","nh3"],
                          template="plotly_dark",
                          title="Pollutant Trends Over Time")
        pol_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), height=250)
    else:
        pol_fig = go.Figure()
        pol_fig.update_layout(template="plotly_dark", title="Pollutant Trends - No Data",
                              paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=250)

    info_children = []
    if latest_weather:
        if "temp_c" in latest_weather:
            info_children.append(html.H4(f"🌡️ Temp: {latest_weather['temp_c']} °C"))
        if "humidity" in latest_weather:
            info_children.append(html.H4(f"💧 Humidity: {latest_weather['humidity']} %"))
        if "wind_kph" in latest_weather:
            info_children.append(html.H4(f"💨 Wind: {latest_weather['wind_kph']} kph"))
        if "pressure" in latest_weather:
            info_children.append(html.H4(f"🧭 Pressure: {latest_weather['pressure']} hPa"))
    else:
        info_children.append(html.H4("No Data"))

    info_children = html.Div(
    children=[
        # Background Image (full card)
        html.Img(
            src="/assets/Info.jpg",
            style={
                "position": "absolute",
                "top": "10",
                "left": "0",
                "width": "100%",
                "height": "130%",
                "object-fit": "cover",
                "border-radius": "12px",
                "zIndex": "0"
            }
        ),

        # EVEN DARK LAYER (FULL CARD)
        html.Div(
            style={
                "position": "absolute",
                "top": "0",
                "left": "0",
                "width": "100%",
                "height": "100%",
                "background": "rgba(0,0,0,0.45)",
                "zIndex": "1",
                "border-radius": "12px"
            }
        ),

        # TEXT CONTENT ON TOP
        html.Div(
            info_children,
            style={
                "position": "relative",
                "zIndex": "2",
                "color": "white",
                "padding": "18px",
                "fontSize": "1rem",
                "textAlign": "left",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center"
            }
        )
    ],
    style={
        "position": "relative",
        "width": "100%",
        "height": "250px",
        "overflow": "hidden",
        "border-radius": "12px"
    }
)


    return cards, map_fig, line_fig, iqair_cards, pol_fig, info_children

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
