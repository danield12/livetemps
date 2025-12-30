import pandas as pd
import numpy as np
import pickle
import os
import requests
import concurrent.futures
from datetime import datetime
import pytz
import plotly.graph_objects as go
import warnings
import time  # Added for the sleep loop

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
WUNDERGROUND_API_KEY = 'e1f10a1e78da46f5b10a1e78da96f525'
ENSEMBLE_FILE = os.path.join(BASE_DIR, "KLAX_Top3_Ensemble.pkl")
HISTORY_CSV = os.path.join(BASE_DIR, "klax_live_history.csv")
HTML_OUTPUT = os.path.join(BASE_DIR, "index.html")

# UPDATED: Set to 60 seconds (1 minute) to match your loop
UPDATE_INTERVAL = 60 

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
        print(f"❌ Model file missing at: {ENSEMBLE_FILE}")
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

    if not pws_vals: 
        print("❌ No PWS data retrieved.")
        return None

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

def run_single_update():
    print(f">> STARTING UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    data = get_live_prediction()
    if not data: return

    # Save to CSV
    df_new = pd.DataFrame([data])
    df_new.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)

    # Load history for graph
    df = pd.read_csv(HISTORY_CSV).tail(288) 

    # Plotly Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['prediction'], name="AI Prediction", line=dict(color='#00b0f6', width=4)))
    if 'nws_temp' in df:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['nws_temp'], name="NWS Official", line=dict(color='#ff4d4d', width=2)))

    fig.update_layout(
        template="plotly_dark", 
        height=600, 
        margin=dict(l=10, r=10, t=50, b=10),
        title="KLAX Temperature: AI vs Official"
    )

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
            .status {{ background: #222; padding: 15px; border-bottom: 2px solid #00b0f6; font-size: 1em; font-weight: bold; }}
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

    with open(HTML_OUTPUT, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ {HTML_OUTPUT} generated successfully.")

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    # Settings for the run
    DURATION_HOURS = 6
    REFRESH_SECONDS = 60
    
    start_time = time.time()
    end_time = start_time + (DURATION_HOURS * 3600)
    
    print(f"--- Script will run for {DURATION_HOURS} hours (until approx {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}) ---")
    
    try:
        while time.time() < end_time:
            run_single_update()
            
            # Calculate remaining time to sleep
            # (Simple sleep is usually fine, but this handles drift slightly better)
            time.sleep(REFRESH_SECONDS)
            
    except KeyboardInterrupt:
        print("\n>> Script stopped by user.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    
    print("--- 6 Hour Run Complete ---")
