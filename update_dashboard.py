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

# Force the script to stay in its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
WUNDERGROUND_API_KEY = 'e1f10a1e78da46f5b10a1e78da96f525'
ENSEMBLE_FILE = "KLAX_Top3_Ensemble.pkl"
HISTORY_CSV = "klax_live_history.csv"
HTML_OUTPUT = "index.html"
UPDATE_INTERVAL = 300 # 5 Minutes is best for GitHub Pages

def get_nws_temp():
    try:
        url = "https://api.weather.gov/stations/KLAX/observations/latest"
        headers = {'User-Agent': '(my-weather-app, contact@example.com)'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            temp_c = r.json()['properties']['temperature']['value']
            if temp_c is not None: return (temp_c * 9/5) + 32
    except: pass
    return None

def get_live_prediction():
    if not os.path.exists(ENSEMBLE_FILE):
        print("❌ Model file missing.")
        return None

    with open(ENSEMBLE_FILE, 'rb') as f:
        ensemble = pickle.load(f)

    all_stations = set()
    for m in ensemble: all_stations.update(m['stations'])
    
    pws_vals = {}
    def fetch_pws(sid):
        url = f"https://api.weather.com/v2/pws/observations/current?stationId={sid}&format=json&units=e&apiKey={WUNDERGROUND_API_KEY}"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return sid, r.json()['observations'][0]['imperial']['temp']
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_pws, list(all_stations)))
        for res in results:
            if res: pws_vals[res[0]] = res[1]

    if not pws_vals: return None

    live_mean = np.mean(list(pws_vals.values()))
    preds = []
    rmses = []
    
    curr_hour = datetime.now(pytz.utc).hour
    curr_month = datetime.now(pytz.utc).month

    for m in ensemble:
        in_data = {f: pws_vals.get(f, live_mean) for f in m['features']}
        in_data['Hour'] = curr_hour
        in_data['Month'] = curr_month
        df_in = pd.DataFrame([in_data])[m['features']]
        preds.append(m['model'].predict(df_in)[0])
        rmses.append(m['rmse'])

    weights = 1 / (np.array(rmses)**2)
    weighted_mean = np.sum(np.array(preds) * weights) / np.sum(weights)
    nws = get_nws_temp()

    return {
        'timestamp': datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': round(weighted_mean, 2),
        'nws_temp': round(nws, 2) if nws else ""
    }

def update_cycle():
    print(f"\n>> UPDATING: {datetime.now().strftime('%H:%M:%S')}")
    data = get_live_prediction()
    if not data: return

    # Save to CSV
    df_new = pd.DataFrame([data])
    df_new.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)

    # Load history for graph
    df = pd.read_csv(HISTORY_CSV).tail(288) # Last 24 hours (if 5-min intervals)

    # Plotly Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['prediction'], name="AI Prediction", line=dict(color='#00b0f6', width=4)))
    if 'nws_temp' in df:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['nws_temp'], name="NWS Official", line=dict(color='#ff4d4d', width=2)))

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=50, b=10))

    # Generate the Full HTML String
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="{UPDATE_INTERVAL}">
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <title>KLAX LIVE AI</title>
        <style>
            body {{ background-color: #111; color: white; font-family: sans-serif; margin: 0; text-align: center; }}
            .status {{ background: #222; padding: 10px; border-bottom: 2px solid #00b0f6; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <div class="status">
            LAST UPDATE: {data['timestamp']} (Pacific) | 
            AI: {data['prediction']}°F | 
            NWS: {data['nws_temp']}°F
        </div>
        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    </body>
    </html>
    """

    # ATOMIC WRITE: Write to tmp, then rename
    with open("tmp.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    os.replace("tmp.html", HTML_OUTPUT)
    print(f"✓ {HTML_OUTPUT} Updated Locally.")

    # Push to Git
    try:
        subprocess.run(["git", "add", HISTORY_CSV, HTML_OUTPUT], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto-update {data['timestamp']}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✓ Pushed to GitHub.")
    except Exception as e:
        print(f"❌ Git Error: {e}")

if __name__ == "__main__":
    while True:
        try:
            update_cycle()
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(UPDATE_INTERVAL)
