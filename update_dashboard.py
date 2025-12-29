import pandas as pd
import numpy as np
import pickle
import os
import requests
import concurrent.futures
from datetime import datetime
import pytz
import plotly.graph_objects as go

# --- CONFIGURATION ---
WUNDERGROUND_API_KEY = 'e1f10a1e78da46f5b10a1e78da96f525'
ENSEMBLE_FILE = "KLAX_Top3_Ensemble.pkl"
HISTORY_CSV = "klax_live_history.csv"
HTML_OUTPUT = "index.html"

def get_live_prediction():
    if not os.path.exists(ENSEMBLE_FILE):
        print("❌ Error: Ensemble file not found.")
        return None

    with open(ENSEMBLE_FILE, 'rb') as f:
        ensemble = pickle.load(f)

    all_needed_stations = set()
    for m in ensemble:
        all_needed_stations.update(m['stations'])
    needed_list = list(all_needed_stations)

    pws_vals = {}
    def fetch_pws(sid):
        url = f"https://api.weather.com/v2/pws/observations/current?stationId={sid}&format=json&units=e&apiKey={WUNDERGROUND_API_KEY}"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                val = r.json()['observations'][0]['imperial']['temp']
                return sid, val
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_pws, sid) for sid in needed_list]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                sid, val = res
                if val is not None and -20 < val < 130:
                    pws_vals[sid] = val

    if not pws_vals: return None

    live_mean = np.mean(list(pws_vals.values()))
    predictions = []
    rmse_values = []
    weights = []
    
    current_hour = datetime.now(pytz.utc).hour
    current_month = datetime.now(pytz.utc).month

    for m in ensemble:
        input_data = {}
        for feat in m['features']:
            if feat == 'Hour': input_data[feat] = current_hour
            elif feat == 'Month': input_data[feat] = current_month
            elif feat in pws_vals: input_data[feat] = pws_vals[feat]
            else: input_data[feat] = live_mean
        
        df_in = pd.DataFrame([input_data])[m['features']]
        pred = m['model'].predict(df_in)[0]
        predictions.append(pred)
        rmse_values.append(m['rmse'])
        weights.append(1 / (m['rmse']**2))

    weights = np.array(weights)
    weighted_mean = np.sum(np.array(predictions) * weights) / np.sum(weights)
    
    avg_variance = np.mean(np.array(rmse_values)**2)
    disagreement_variance = np.var(predictions)
    total_std_dev = np.sqrt(avg_variance + disagreement_variance)
    margin_of_error = 1.96 * total_std_dev

    return {
        'timestamp': datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': round(weighted_mean, 2),
        'ci_lower': round(weighted_mean - margin_of_error, 2),
        'ci_upper': round(weighted_mean + margin_of_error, 2),
        'raw_mean': round(live_mean, 2)
    }

def update_files():
    data = get_live_prediction()
    if not data:
        print("No data received.")
        return

    # 1. Update CSV
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, 'w') as f:
            f.write("timestamp,prediction,ci_lower,ci_upper,raw_mean\n")
    
    with open(HISTORY_CSV, 'a') as f:
        f.write(f"{data['timestamp']},{data['prediction']},{data['ci_lower']},{data['ci_upper']},{data['raw_mean']}\n")

    # 2. Update HTML Graph
    df = pd.read_csv(HISTORY_CSV)
    
    # Keep only last 24 hours of data (approx 288 records) to keep page load fast
    if len(df) > 288:
        df = df.tail(288)

    fig = go.Figure()
    
    # CI Band
    fig.add_trace(go.Scatter(
        x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
        y=df['ci_upper'].tolist() + df['ci_lower'].tolist()[::-1],
        fill='toself', fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name='95% Conf. Interval'
    ))

    # Prediction Line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['prediction'],
        mode='lines+markers', line=dict(color='#00b0f6', width=3),
        marker=dict(size=6), name='AI Prediction'
    ))

    # Raw Mean Line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['raw_mean'],
        mode='lines', line=dict(color='gray', width=1, dash='dot'),
        name='Raw Station Mean'
    ))

    fig.update_layout(
        title=f"KLAX Live Temp (Last 24h)<br><sup>Current: {data['prediction']}°F</sup>",
        template="plotly_dark", xaxis_title="Time (Pacific)", yaxis_title="Temp (°F)",
        hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1)
    )

    fig.write_html(HTML_OUTPUT)
    print(f"Updated {HTML_OUTPUT}")

if __name__ == "__main__":
    update_files()
