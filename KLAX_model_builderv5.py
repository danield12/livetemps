import pandas as pd
import numpy as np
import pickle
import os
import time
import requests
import concurrent.futures
from datetime import datetime
import pytz
import plotly.graph_objects as go
import subprocess

# --- CONFIGURATION ---
WUNDERGROUND_API_KEY = 'e1f10a1e78da46f5b10a1e78da96f525'
ENSEMBLE_FILE = "KLAX_Top3_Ensemble.pkl"
HISTORY_CSV = "klax_live_history.csv"
HTML_OUTPUT = "index.html"  # Name it index.html for GitHub Pages
UPDATE_INTERVAL = 60  # Run every 60 seconds

def get_live_prediction():
    """Runs the Ensemble Model on live PWS data."""
    if not os.path.exists(ENSEMBLE_FILE):
        print("❌ Error: Ensemble file not found. Run training script first.")
        return None

    with open(ENSEMBLE_FILE, 'rb') as f:
        ensemble = pickle.load(f)

    # 1. Identify Stations
    all_needed_stations = set()
    for m in ensemble:
        all_needed_stations.update(m['stations'])
    needed_list = list(all_needed_stations)

    # 2. Fetch Data
    pws_vals = {}
    def fetch_pws(sid):
        url = f"https://api.weather.com/v2/pws/observations/current?stationId={sid}&format=json&units=e&apiKey={WUNDERGROUND_API_KEY}"
        try:
            r = requests.get(url, timeout=2.5)
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

    # 3. Predict
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

    # 4. Aggregate & CI
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

def update_plot():
    """Reads history CSV and generates a Plotly HTML file."""
    if not os.path.exists(HISTORY_CSV): return

    df = pd.read_csv(HISTORY_CSV)
    
    fig = go.Figure()

    # 1. Confidence Interval (Shaded Area)
    # We draw the Upper bound, then the Lower bound (filling to next y) to create a band
    fig.add_trace(go.Scatter(
        x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
        y=df['ci_upper'].tolist() + df['ci_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))

    # 2. Raw PWS Mean (Dotted Line)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['raw_mean'],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        name='Raw Station Mean'
    ))

    # 3. Main Prediction (Solid Line)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['prediction'],
        mode='lines+markers',
        line=dict(color='#00b0f6', width=3),
        marker=dict(size=6),
        name='Ensemble Prediction'
    ))

    # Layout Styling
    current_temp = df['prediction'].iloc[-1]
    last_time = df['timestamp'].iloc[-1]
    
    fig.update_layout(
        title=f"KLAX Live Temperature Model<br><sup>Current: {current_temp}°F (Updated: {last_time})</sup>",
        template="plotly_dark",
        xaxis_title="Time (Pacific)",
        yaxis_title="Temperature (°F)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.write_html(HTML_OUTPUT)
    print(f"   [✓] Graph updated: {HTML_OUTPUT}")

def git_push():
    """Commits and pushes changes to GitHub."""
    try:
        subprocess.run(["git", "add", HISTORY_CSV, HTML_OUTPUT], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto-update: {datetime.now().strftime('%H:%M')}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("   [✓] Pushed to GitHub")
    except Exception as e:
        print(f"   [!] Git Error: {e}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("--- STARTING KLAX LIVE MONITOR ---")
    
    # Create CSV header if missing
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, 'w') as f:
            f.write("timestamp,prediction,ci_lower,ci_upper,raw_mean\n")

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running inference...")
        
        data = get_live_prediction()
        
        if data:
            print(f"   Pred: {data['prediction']}°F (±{(data['ci_upper']-data['prediction']):.2f})")
            
            # Append to CSV
            with open(HISTORY_CSV, 'a') as f:
                f.write(f"{data['timestamp']},{data['prediction']},{data['ci_lower']},{data['ci_upper']},{data['raw_mean']}\n")
            
            # Update Graph
            update_plot()
            
            # Git Push
            git_push()
        
        else:
            print("   [!] Failed to get prediction data.")

        print(f"   Sleeping for {UPDATE_INTERVAL}s...")
        time.sleep(UPDATE_INTERVAL)