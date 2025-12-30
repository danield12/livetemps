import pandas as pd
import numpy as np
import pickle
import os
import requests
import concurrent.futures
from datetime import datetime
import time
import pytz
import plotly.graph_objects as go
import subprocess
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
WUNDERGROUND_API_KEY = 'e1f10a1e78da46f5b10a1e78da96f525'
ENSEMBLE_FILE = "KLAX_Top3_Ensemble.pkl"
HISTORY_CSV = "klax_live_history.csv"
HTML_OUTPUT = "index.html"
UPDATE_INTERVAL = 300  # Increased to 5 mins to allow GitHub Pages to build

# --- 1. FETCH NWS OFFICIAL DATA ---
def get_nws_temp():
    try:
        url = "https://api.weather.gov/stations/KLAX/observations/latest"
        headers = {'User-Agent': '(my-weather-app, contact@example.com)'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            temp_c = r.json()['properties']['temperature']['value']
            if temp_c is not None:
                return (temp_c * 9/5) + 32
    except Exception as e:
        print(f"   [NWS Error]: {e}")
    return None

# --- 2. RUN AI MODEL ---
def get_live_prediction():
    if not os.path.exists(ENSEMBLE_FILE):
        print(f"❌ Error: {ENSEMBLE_FILE} not found in {os.getcwd()}")
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

    print(f"-> Fetching data from {len(needed_list)} PWS stations...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_pws, sid) for sid in needed_list]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                sid, val = res
                if val is not None and -20 < val < 130:
                    pws_vals[sid] = val

    if not pws_vals:
        print("❌ No PWS data retrieved.")
        return None

    live_vals = list(pws_vals.values())
    live_mean = np.mean(live_vals)
    live_median = np.median(live_vals)

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

    nws_val = get_nws_temp()

    return {
        'timestamp': datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': round(weighted_mean, 2),
        'ci_lower': round(weighted_mean - margin_of_error, 2),
        'ci_upper': round(weighted_mean + margin_of_error, 2),
        'raw_mean': round(live_mean, 2),
        'raw_median': round(live_median, 2),
        'nws_temp': round(nws_val, 2) if nws_val else ""
    }

# --- 3. UPDATE CYCLE ---
def update_cycle():
    print(f"\n--- Cycle Start: {datetime.now().strftime('%H:%M:%S')} ---")
    
    # 1. Get Data
    data = get_live_prediction()
    if not data:
        return

    # 2. Update CSV
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, 'w') as f:
            f.write("timestamp,prediction,ci_lower,ci_upper,raw_mean,raw_median,nws_temp\n")
    
    with open(HISTORY_CSV, 'a') as f:
        f.write(f"{data['timestamp']},{data['prediction']},{data['ci_lower']},{data['ci_upper']},{data['raw_mean']},{data['raw_median']},{data['nws_temp']}\n")
    print("✓ CSV Updated.")

    # 3. Generate Graph
    df = pd.read_csv(HISTORY_CSV)
    if len(df) > 1440: df = df.tail(1440) 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ci_upper'], mode='lines', line=dict(color='rgba(0,255,255,0.2)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ci_lower'], mode='lines', fill='tonexty', fillcolor='rgba(0,255,255,0.1)', line=dict(color='rgba(0,255,255,0.2)', width=0), name='95% Confidence'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['raw_mean'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='PWS Mean'))
    
    df_nws = df[df['nws_temp'].notna()]
    if not df_nws.empty:
        fig.add_trace(go.Scatter(x=df_nws['timestamp'], y=df_nws['nws_temp'], mode='lines+markers', line=dict(color='#ff4d4d', width=2), name='NWS Official'))

    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['prediction'], mode='lines+markers', line=dict(color='#00b0f6', width=4), name='AI Prediction'))

    current_pred = data['prediction']
    current_nws = data['nws_temp'] if data['nws_temp'] != "" else "N/A"
    
    fig.update_layout(
        title=f"KLAX Live Temp Prediction<br><sup>AI: {current_pred}°F | NWS: {current_nws}°F | Last Updated: {data['timestamp']}</sup>",
        template="plotly_dark", xaxis_title="Time (Pacific)", yaxis_title="Temp (°F)",
        hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )

    # 4. Save HTML with Cache-Busting Meta Tags
    html_body = fig.to_html(full_html=False, include_plotlyjs='cdn')
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="{UPDATE_INTERVAL}">
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <title>KLAX AI Live Weather</title>
        <style>body {{ background-color: #111; margin: 0; padding: 20px; font-family: sans-serif; color: white; }}</style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    with open(HTML_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"✓ {HTML_OUTPUT} written to disk.")

    # 5. Git Push
    try:
        subprocess.run(["git", "add", HISTORY_CSV, HTML_OUTPUT], check=True)
        subprocess.run(["git", "commit", "-m", f"Update {data['timestamp']}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✓ Git Push Successful.")
    except Exception as e:
        print(f"⚠️ Git Push Failed: {e}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("--- STARTING CONTINUOUS MONITOR ---")
    print(f"Interval: {UPDATE_INTERVAL}s | Output: {HTML_OUTPUT}")
    
    while True:
        try:
            update_cycle()
        except Exception as e:
            print(f"Critical Cycle Error: {e}")

        # Check for Daily Exit
        now = datetime.now().time()
        if now.hour == 23 and now.minute >= 55:
            print("🕒 Maintenance window reached. Exiting.")
            break
        
        print(f"Sleeping for {UPDATE_INTERVAL}s...")
        time.sleep(UPDATE_INTERVAL)
