import pandas as pd
import numpy as np
import pickle
import os
import requests
import concurrent.futures
from datetime import datetime, time as dtime
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
UPDATE_INTERVAL = 60  # Run every 60 seconds

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
    except: pass
    return None

# --- 2. RUN AI MODEL ---
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
            r = requests.get(url, timeout=4)
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

# --- 3. UPDATE GRAPH & GIT ---
def update_cycle():
    data = get_live_prediction()
    if not data:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No data received.")
        return

    # Update CSV
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, 'w') as f:
            f.write("timestamp,prediction,ci_lower,ci_upper,raw_mean,raw_median,nws_temp\n")
    
    with open(HISTORY_CSV, 'a') as f:
        f.write(f"{data['timestamp']},{data['prediction']},{data['ci_lower']},{data['ci_upper']},{data['raw_mean']},{data['raw_median']},{data['nws_temp']}\n")

    # Generate Graph
    df = pd.read_csv(HISTORY_CSV)
    if len(df) > 1440: df = df.tail(1440) # Keep 24 hours

    fig = go.Figure()
    
    # CI Lines
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ci_upper'], mode='lines', line=dict(color='cyan', width=1, dash='dash'), name='95% CI High'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ci_lower'], mode='lines', line=dict(color='cyan', width=1, dash='dash'), name='95% CI Low'))

    # Raw Stats
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['raw_mean'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='Station Mean'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['raw_median'], mode='lines', line=dict(color='yellow', width=1), name='Station Median'))

    # NWS Official
    df_nws = df[df['nws_temp'].notna()]
    if not df_nws.empty:
        fig.add_trace(go.Scatter(x=df_nws['timestamp'], y=df_nws['nws_temp'], mode='lines+markers', line=dict(color='#ff4d4d', width=2), marker=dict(size=4), name='NWS Official'))

    # AI Prediction
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['prediction'], mode='lines+markers', line=dict(color='#00b0f6', width=4), marker=dict(size=6), name='AI Prediction'))

    current_pred = data['prediction']
    current_nws = data['nws_temp'] if data['nws_temp'] != "" else "N/A"
    
    fig.update_layout(
        title=f"KLAX Live Temp<br><sup>AI: {current_pred}°F | NWS: {current_nws}°F</sup>",
        template="plotly_dark", xaxis_title="Time (Pacific)", yaxis_title="Temp (°F)",
        hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )

    # 1. Write the standard Plotly file
    fig.write_html(HTML_OUTPUT)

    # ---------------------------------------------------------
    # 2. INJECT AUTO-REFRESH TAG
    # This reads the file back, adds a meta refresh tag to the <head>, and resaves it.
    # ---------------------------------------------------------
    try:
        with open(HTML_OUTPUT, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject the meta refresh tag (refresh every 60 seconds)
        # We replace the opening <head> tag with <head> + the meta tag
        refresh_tag = f'<head><meta http-equiv="refresh" content="{UPDATE_INTERVAL}">'
        updated_content = html_content.replace('<head>', refresh_tag)
        
        with open(HTML_OUTPUT, 'w', encoding='utf-8') as f:
            f.write(updated_content)
    except Exception as e:
        print(f"Error injecting refresh tag: {e}")
    # ---------------------------------------------------------
    
    # Git Push
    try:
        subprocess.run(["git", "add", HISTORY_CSV, HTML_OUTPUT], check=True)
        subprocess.run(["git", "commit", "-m", f"Update {datetime.now().strftime('%H:%M')}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated & Pushed.")
    except:
        print("Git push failed (connection error?)")

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("--- STARTING CONTINUOUS MONITOR ---")
    
    while True:
        # 1. Run the Update
        try:
            update_cycle()
        except Exception as e:
            print(f"Cycle Error: {e}")

        # 2. Check for Daily Exit (e.g., 23:58 PM)
        # This allows Cron to restart it fresh at 00:00 or 06:00
        now = datetime.now().time()
        if now.hour == 23 and now.minute >= 58:
            print("🕒 End of day reached. Exiting for scheduled restart.")
            break
        
        # 3. Sleep
        time.sleep(UPDATE_INTERVAL)
