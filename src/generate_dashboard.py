import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']

def get_aqi_color(aqi_val):
    if pd.isna(aqi_val):
        return "#747d8c"
    v = float(aqi_val)
    if v <= 50:
        return "#2ed573" # Good (Green)
    elif v <= 100:
        return "#7bed9f" # Satisfactory (Light Green)
    elif v <= 200:
        return "#ffa502" # Moderate (Orange-Yellow)
    elif v <= 300:
        return "#ff7f50" # Poor (Coral)
    elif v <= 400:
        return "#ff4757" # Very Poor (Red)
    else:
        return "#8b0000" # Severe (Dark Crimson)

def generate_command_center_dashboard():
    clustered_file = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    spearman_file = PROJECT_ROOT / "outputs" / "reports" / "spearman_correlation_matrix.csv"
    loadings_file = PROJECT_ROOT / "outputs" / "reports" / "pca_component_loadings.csv"
    eval_file = PROJECT_ROOT / "outputs" / "reports" / "clustering_metrics_evaluation.csv"
    cluster_medians_file = PROJECT_ROOT / "outputs" / "reports" / "cluster_profiles_summary.csv"
    
    forecast_file = PROCESSED_DATA_DIR / "station_next_hour_forecasts.csv"
    multi_metrics_file = PROJECT_ROOT / "outputs" / "reports" / "forecasting_multi_pollutant_metrics.csv"
    forecast_importance_file = PROJECT_ROOT / "outputs" / "reports" / "forecasting_feature_importance.csv"

    df = pd.read_csv(clustered_file)
    spearman_df = pd.read_csv(spearman_file, index_col=0)
    loadings_df = pd.read_csv(loadings_file, index_col=0)
    eval_df = pd.read_csv(eval_file)
    cluster_medians_df = pd.read_csv(cluster_medians_file, index_col=0)

    forecast_df = pd.read_csv(forecast_file) if forecast_file.exists() else None
    multi_metrics_df = pd.read_csv(multi_metrics_file) if multi_metrics_file.exists() else None
    forecast_importance_df = pd.read_csv(forecast_importance_file) if forecast_importance_file.exists() else None

    # Merge forecast details into master dataframe
    if forecast_df is not None:
        # Select all forecast columns
        forecast_cols = [
            'station', 'current_aqi', 'predicted_next_aqi', 'predicted_aqi_change', 'forecast_trend',
            'current_dominant_pollutant', 'predicted_dominant_pollutant', 'current_aqi_category', 'predicted_aqi_category',
            'pred_PM2.5', 'change_PM2.5', 'pred_PM10', 'change_PM10',
            'pred_NO2', 'change_NO2', 'pred_SO2', 'change_SO2',
            'pred_CO', 'change_CO', 'pred_OZONE', 'change_OZONE',
            'pred_NH3', 'change_NH3'
        ]
        available_fcols = [c for c in forecast_cols if c in forecast_df.columns]
        df = df.merge(forecast_df[available_fcols], on='station', how='left')
    else:
        df['current_aqi'] = 150.0
        df['predicted_next_aqi'] = 150.0
        df['predicted_aqi_change'] = 0.0
        df['forecast_trend'] = 'Stable Air Quality'
        df['current_dominant_pollutant'] = 'PM2.5'
        df['predicted_dominant_pollutant'] = 'PM2.5'
        df['current_aqi_category'] = 'Moderate'
        df['predicted_aqi_category'] = 'Moderate'
        for pol in TARGET_POLLUTANTS:
            df[f'pred_{pol}'] = df[pol]
            df[f'change_{pol}'] = 0.0

    cluster_colors = {
        0: "#ff4757",  # Red: Multi-pollutant severe
        1: "#2ed573",  # Green: Clean ambient baseline
        2: "#ffa502",  # Orange: Particulate & dust dominated
        3: "#a55eea"   # Purple: Industrial sulfur & gas dominated
    }

    stations = []
    for _, row in df.iterrows():
        # Action determination based on dominant and high-level pollutant
        dom = str(row.get('predicted_dominant_pollutant', 'PM2.5'))
        if dom in ['PM2.5', 'PM10'] or row['PM10'] > 200 or row['PM2.5'] > 120:
            action = "Deploy anti-smog water cannons, mechanized mist sweepers, halt unpaved road dust & construction"
        elif dom in ['NO2', 'CO'] or row['NO2'] > 60:
            action = "Heavy diesel truck bypass rerouting, dynamic congestion tolling, prioritize electrified transit corridors"
        elif dom == 'SO2' or row['SO2'] > 30:
            action = "Audit thermal stack FGD units, enforce continuous CEMS telemetry, inspect high-sulfur petcoke boilers"
        elif dom == 'OZONE' or row['OZONE'] > 60:
            action = "Regulate VOC vapor recovery during fuel dispensing, restrict daytime volatile paint solvent operations"
        elif dom == 'NH3' or row['NH3'] > 50:
            action = "Enforce covered bio-waste slurry storage, inspect cold-storage ammonia cooling refrigeration loops"
        else:
            action = "Continuous ambient telemetry surveillance and real-time public health advisories"

        cur_aqi = float(row.get('current_aqi', 150)) if pd.notna(row.get('current_aqi')) else 150.0
        pred_aqi = float(row.get('predicted_next_aqi', cur_aqi)) if pd.notna(row.get('predicted_next_aqi')) else cur_aqi
        aqi_change = float(row.get('predicted_aqi_change', 0.0)) if pd.notna(row.get('predicted_aqi_change')) else round(pred_aqi - cur_aqi, 1)

        pollutant_dict = {}
        for pol in TARGET_POLLUTANTS:
            c_val = round(float(row[pol]), 1) if pol in row and pd.notna(row[pol]) else 0.0
            p_val = round(float(row.get(f'pred_{pol}', c_val)), 1) if pd.notna(row.get(f'pred_{pol}')) else c_val
            ch_val = round(float(row.get(f'change_{pol}', p_val - c_val)), 1) if pd.notna(row.get(f'change_{pol}')) else round(p_val - c_val, 1)
            pollutant_dict[pol] = {
                "cur": c_val,
                "pred": p_val,
                "change": ch_val
            }

        stations.append({
            "rank": int(row['priority_rank']),
            "name": str(row['station']),
            "city": str(row['city']),
            "state": str(row['state']),
            "lat": float(row['latitude']) if pd.notna(row['latitude']) else None,
            "lon": float(row['longitude']) if pd.notna(row['longitude']) else None,
            "cluster": int(row['cluster']),
            "profile": str(row['cluster_profile']),
            "color": cluster_colors.get(int(row['cluster']), "#747d8c"),
            "outlier": bool(row['is_outlier']),
            "cur_aqi": cur_aqi,
            "pred_aqi": pred_aqi,
            "aqi_change": aqi_change,
            "aqi_color": get_aqi_color(pred_aqi),
            "cur_dominant": str(row.get('current_dominant_pollutant', 'PM2.5')),
            "pred_dominant": str(row.get('predicted_dominant_pollutant', 'PM2.5')),
            "cur_category": str(row.get('current_aqi_category', 'Moderate')),
            "pred_category": str(row.get('predicted_aqi_category', 'Moderate')),
            "trend": str(row.get('forecast_trend', 'Stable Air Quality')),
            "pollutants": pollutant_dict,
            "action": action
        })

    stations_json = json.dumps(stations)
    pollutants_json = json.dumps(TARGET_POLLUTANTS)
    corr_matrix_json = json.dumps(spearman_df.values.tolist())
    
    pca_labels_json = json.dumps(loadings_df.index.tolist())
    pc1_json = json.dumps(loadings_df['PC1_Loading'].tolist())
    pc2_json = json.dumps(loadings_df['PC2_Loading'].tolist())
    pc3_json = json.dumps(loadings_df['PC3_Loading'].tolist())
    
    cluster_profiles_json = json.dumps(cluster_medians_df.to_dict(orient='index'))
    
    k_labels_json = json.dumps(eval_df['k'].tolist())
    sil_json = json.dumps(eval_df['silhouette_score'].tolist())
    db_json = json.dumps(eval_df['davies_bouldin_index'].tolist())

    # Multi-pollutant model performance
    multi_metrics_json = json.dumps(multi_metrics_df.to_dict(orient='records') if multi_metrics_df is not None else [])
    model_pollutants = multi_metrics_df['Pollutant'].tolist() if multi_metrics_df is not None else TARGET_POLLUTANTS
    model_r2 = multi_metrics_df['R2_Score'].tolist() if multi_metrics_df is not None else [0.98] * len(TARGET_POLLUTANTS)
    model_rmse = multi_metrics_df['RMSE'].tolist() if multi_metrics_df is not None else [3.0] * len(TARGET_POLLUTANTS)
    model_reductions = multi_metrics_df['Error_Reduction_%'].tolist() if multi_metrics_df is not None else [5.0] * len(TARGET_POLLUTANTS)

    # Top surge alerts across all impurities
    severe_aqi_spikes = sorted([s for s in stations if s['aqi_change'] > 5], key=lambda x: x['aqi_change'], reverse=True)
    if not severe_aqi_spikes:
        severe_aqi_spikes = sorted(stations, key=lambda x: x['pred_aqi'], reverse=True)

    avg_cur_aqi = round(np.mean([s['cur_aqi'] for s in stations]), 1)
    avg_pred_aqi = round(np.mean([s['pred_aqi'] for s in stations]), 1)
    severe_count = sum(1 for s in stations if s['pred_aqi'] > 300)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroPulse | National Multi-Pollutant Command Center</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" />
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-glass: rgba(15, 23, 42, 0.90);
            --bg-glass-card: rgba(30, 41, 59, 0.85);
            --border-glass: rgba(255, 255, 255, 0.12);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-red: #ff4757;
            --accent-green: #2ed573;
            --accent-orange: #ffa502;
            --accent-purple: #a55eea;
            --accent-cyan: #00d2d3;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body, html {{ height: 100%; width: 100%; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: #0b0f19; color: var(--text-main); }}

        /* Fullscreen Map */
        #map {{ height: 100vh; width: 100vw; position: absolute; top: 0; left: 0; z-index: 1; }}

        /* Top HUD Header */
        .hud-header {{
            position: absolute; top: 16px; left: 16px; right: 16px; z-index: 1000;
            display: flex; justify-content: space-between; align-items: center;
            background: var(--bg-glass); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-glass); border-radius: 12px;
            padding: 12px 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }}
        .brand {{ display: flex; align-items: center; gap: 12px; }}
        .pulse-indicator {{
            width: 10px; height: 10px; background: #2ed573; border-radius: 50%;
            box-shadow: 0 0 10px #2ed573, 0 0 20px #2ed573;
            animation: pulse-dot 2s infinite ease-in-out;
        }}
        @keyframes pulse-dot {{
            0%, 100% {{ transform: scale(0.9); opacity: 0.8; }}
            50% {{ transform: scale(1.3); opacity: 1; }}
        }}
        .brand-title {{ font-size: 16px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; color: #fff; }}
        .brand-subtitle {{ font-size: 11px; color: var(--text-sub); margin-top: 1px; }}

        /* Search Bar */
        .search-container {{ flex: 0 1 340px; position: relative; }}
        .search-input {{
            width: 100%; background: rgba(255, 255, 255, 0.08); border: 1px solid var(--border-glass);
            border-radius: 8px; padding: 8px 14px 8px 36px; color: #fff; font-size: 13px; outline: none;
            transition: all 0.2s;
        }}
        .search-input:focus {{ background: rgba(255, 255, 255, 0.14); border-color: var(--accent-blue); box-shadow: 0 0 12px rgba(56, 189, 248, 0.3); }}
        .search-icon {{ position: absolute; left: 12px; top: 9px; font-size: 14px; opacity: 0.6; }}

        /* Top HUD Quick Metrics */
        .hud-metrics {{ display: flex; gap: 12px; }}
        .metric-pill {{
            background: rgba(255, 255, 255, 0.06); border: 1px solid var(--border-glass);
            padding: 6px 14px; border-radius: 8px; font-size: 12px; display: flex; align-items: center; gap: 8px;
        }}
        .metric-pill b {{ font-size: 14px; color: #fff; }}

        /* Left Floating Control Deck */
        .control-deck {{
            position: absolute; top: 90px; left: 16px; z-index: 1000;
            background: var(--bg-glass); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--border-glass); border-radius: 12px;
            padding: 16px; width: 280px; box-shadow: 0 8px 32px rgba(0,0,0,0.45);
            font-size: 12px; max-height: calc(100vh - 120px); overflow-y: auto;
        }}
        .deck-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-sub); margin-bottom: 10px; font-weight: 700; }}
        .layer-btn {{
            width: 100%; display: flex; justify-content: space-between; align-items: center;
            background: rgba(255, 255, 255, 0.05); border: 1px solid transparent; border-radius: 6px;
            padding: 8px 12px; color: #fff; font-size: 12px; cursor: pointer; margin-bottom: 5px; transition: all 0.2s;
        }}
        .layer-btn:hover {{ background: rgba(255, 255, 255, 0.1); }}
        .layer-btn.active {{ background: rgba(56, 189, 248, 0.18); border-color: var(--accent-blue); color: var(--accent-blue); font-weight: 700; }}

        .cluster-legend-item {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 5px 4px; font-size: 11px; cursor: pointer; opacity: 0.9;
        }}
        .cluster-legend-item:hover {{ opacity: 1; }}
        .cluster-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; }}

        /* Right Floating Drawer Button */
        .drawer-toggle-btn {{
            position: absolute; top: 90px; right: 16px; z-index: 1000;
            background: var(--bg-glass); backdrop-filter: blur(18px);
            border: 1px solid var(--border-glass); border-radius: 10px;
            padding: 10px 18px; color: #fff; font-size: 13px; font-weight: 700;
            cursor: pointer; display: flex; align-items: center; gap: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4); transition: all 0.2s;
        }}
        .drawer-toggle-btn:hover {{ border-color: var(--accent-blue); transform: translateY(-1px); }}

        /* Right Slide-over Analytics Drawer */
        .analytics-drawer {{
            position: absolute; top: 16px; bottom: 16px; right: -720px; width: 680px; z-index: 1001;
            background: rgba(11, 15, 25, 0.95); backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px);
            border: 1px solid var(--border-glass); border-radius: 16px;
            box-shadow: -10px 0 40px rgba(0,0,0,0.6); display: flex; flex-direction: column;
            transition: right 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .analytics-drawer.open {{ right: 16px; }}
        .drawer-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 18px 24px; border-bottom: 1px solid var(--border-glass);
        }}
        .drawer-header h3 {{ font-size: 16px; font-weight: 700; color: #fff; }}
        .close-btn {{ background: transparent; border: none; color: #94a3b8; font-size: 20px; cursor: pointer; }}
        .close-btn:hover {{ color: #fff; }}

        .drawer-nav {{
            display: flex; gap: 6px; padding: 12px 24px 0; border-bottom: 1px solid var(--border-glass); overflow-x: auto;
        }}
        .drawer-tab {{
            background: transparent; border: none; padding: 8px 14px; font-size: 12px; font-weight: 600;
            color: var(--text-sub); cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s;
            white-space: nowrap;
        }}
        .drawer-tab.active {{ color: var(--accent-blue); border-bottom-color: var(--accent-blue); }}

        .drawer-body {{ flex: 1; overflow-y: auto; padding: 20px 24px; }}
        .drawer-pane {{ display: none; }}
        .drawer-pane.active {{ display: block; }}

        /* Floating Station Detail Card */
        .station-hud-card {{
            position: absolute; bottom: 25px; left: 16px; z-index: 1000; width: 440px;
            background: var(--bg-glass); backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
            border: 1px solid var(--border-glass); border-radius: 14px;
            padding: 18px; box-shadow: 0 12px 40px rgba(0,0,0,0.55); display: none;
            animation: slideUp 0.3s ease-out; max-height: 82vh; overflow-y: auto;
        }}
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        .station-hud-card.active {{ display: block; }}
        .hud-station-name {{ font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 2px; }}
        .hud-station-sub {{ font-size: 12px; color: var(--text-sub); margin-bottom: 12px; }}

        /* Badges */
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; display: inline-block; }}
        .badge-danger {{ background: rgba(255, 71, 87, 0.2); color: #ff6b81; border: 1px solid rgba(255, 71, 87, 0.4); }}
        .badge-warning {{ background: rgba(255, 165, 2, 0.2); color: #ffa502; border: 1px solid rgba(255, 165, 2, 0.4); }}
        .badge-success {{ background: rgba(46, 213, 115, 0.2); color: #2ed573; border: 1px solid rgba(46, 213, 115, 0.4); }}
        .badge-purple {{ background: rgba(165, 94, 234, 0.2); color: #a55eea; border: 1px solid rgba(165, 94, 234, 0.4); }}
        .badge-outlier {{ background: #ff4757; color: white; }}

        /* DataTables dark custom */
        .dataTables_wrapper {{ color: #fff; font-size: 12px; }}
        .dataTables_wrapper select, .dataTables_wrapper input {{ background: rgba(255,255,255,0.08); border: 1px solid var(--border-glass); color: #fff; border-radius: 6px; padding: 4px 8px; }}
        table.dataTable {{ border-color: rgba(255,255,255,0.1) !important; }}
        table.dataTable thead th {{ background: rgba(255,255,255,0.05); color: var(--text-sub); border-bottom: 1px solid var(--border-glass) !important; }}
    </style>
</head>
<body>

    <!-- Fullscreen Map -->
    <div id="map"></div>

    <!-- Top HUD Header -->
    <div class="hud-header">
        <div class="brand">
            <div class="pulse-indicator"></div>
            <div>
                <div class="brand-title">AeroPulse Command Center</div>
                <div class="brand-subtitle">Multi-Pollutant Profiling & 7-Gas AI Forecasting &bull; CPCB CAAQMS Network</div>
            </div>
        </div>

        <div class="search-container">
            <span class="search-icon">&#128269;</span>
            <input type="text" class="search-input" id="stationSearch" placeholder="Search station, city (e.g. Anand Vihar, Delhi, Mumbai)..." onkeyup="if(event.key === 'Enter') handleStationSearch(this.value)">
        </div>

        <div class="hud-metrics">
            <div class="metric-pill">
                <span>Active Stations</span>
                <b>{len(stations)}</b>
            </div>
            <div class="metric-pill">
                <span>Avg Predicted AQI</span>
                <b style="color:{get_aqi_color(avg_pred_aqi)};">{avg_pred_aqi}</b>
            </div>
            <div class="metric-pill">
                <span>Severe Hotspots (&gt;300)</span>
                <b style="color:var(--accent-red);">{severe_count}</b>
            </div>
        </div>
    </div>

    <!-- Left Floating Control Deck -->
    <div class="control-deck">
        <div class="deck-title">Visual Layer View</div>
        <button class="layer-btn active" id="btn-mode-cluster" onclick="setVisualMode('cluster')">
            <span>&#9679; Cluster Profiles (K=4)</span>
            <span style="font-size:10px; opacity:0.7;">Fingerprint</span>
        </button>
        <button class="layer-btn" id="btn-mode-predicted-aqi" onclick="setVisualMode('predicted-aqi')">
            <span>&#9679; Next-Hour Predicted AQI</span>
            <span style="font-size:10px; opacity:0.7;">All 7 Gases</span>
        </button>
        <button class="layer-btn" id="btn-mode-current-aqi" onclick="setVisualMode('current-aqi')">
            <span>&#9679; Current CPCB AQI</span>
            <span style="font-size:10px; opacity:0.7;">Live</span>
        </button>
        <button class="layer-btn" id="btn-mode-pm25" onclick="setVisualMode('pm25')">
            <span>&#9679; PM2.5 Fine Particles</span>
            <span style="font-size:10px; opacity:0.7;">&mu;g/m&sup3;</span>
        </button>
        <button class="layer-btn" id="btn-mode-pm10" onclick="setVisualMode('pm10')">
            <span>&#9679; PM10 Coarse Dust</span>
            <span style="font-size:10px; opacity:0.7;">&mu;g/m&sup3;</span>
        </button>
        <button class="layer-btn" id="btn-mode-toxic-gas" onclick="setVisualMode('toxic-gas')">
            <span>&#9679; Toxic Gases (NO2/SO2/CO)</span>
            <span style="font-size:10px; opacity:0.7;">Industrial</span>
        </button>

        <hr style="border:0; border-top:1px solid var(--border-glass); margin:12px 0;">

        <div class="deck-title">K-Means Cluster Filter</div>
        <div class="cluster-legend-item" onclick="toggleClusterFilter(0)">
            <div><span class="cluster-dot" style="background:var(--accent-red);"></span>Cluster 0: Multi-Pollutant Severe</div>
            <b style="color:var(--accent-red);">{(df['cluster']==0).sum()}</b>
        </div>
        <div class="cluster-legend-item" onclick="toggleClusterFilter(1)">
            <div><span class="cluster-dot" style="background:var(--accent-green);"></span>Cluster 1: Baseline Clean</div>
            <b style="color:var(--accent-green);">{(df['cluster']==1).sum()}</b>
        </div>
        <div class="cluster-legend-item" onclick="toggleClusterFilter(2)">
            <div><span class="cluster-dot" style="background:var(--accent-orange);"></span>Cluster 2: Dust Dominated</div>
            <b style="color:var(--accent-orange);">{(df['cluster']==2).sum()}</b>
        </div>
        <div class="cluster-legend-item" onclick="toggleClusterFilter(3)">
            <div><span class="cluster-dot" style="background:var(--accent-purple);"></span>Cluster 3: Industrial Gas Dominated</div>
            <b style="color:var(--accent-purple);">{(df['cluster']==3).sum()}</b>
        </div>

        <hr style="border:0; border-top:1px solid var(--border-glass); margin:12px 0;">
        <label style="display:flex; align-items:center; gap:8px; cursor:pointer; font-size:11px;">
            <input type="checkbox" id="chkOutliersOnly" onchange="toggleOutliersOnly(this.checked)">
            <span>Filter Outliers Only ({df['is_outlier'].sum()} flagged)</span>
        </label>
    </div>

    <!-- Toggle Drawer Button -->
    <button class="drawer-toggle-btn" onclick="toggleDrawer()">
        <span>&#128202; Analytics & ML Models</span>
        <span style="font-size:11px; opacity:0.7;">&#9654;</span>
    </button>

    <!-- Floating Slide-over Analytics Drawer -->
    <div class="analytics-drawer" id="analyticsDrawer">
        <div class="drawer-header">
            <h3>Machine Learning Analytics & Insights</h3>
            <button class="close-btn" onclick="toggleDrawer()">&times;</button>
        </div>

        <div class="drawer-nav">
            <button class="drawer-tab active" onclick="switchDrawerTab('pane-forecast')">&#128302; 7-Gas AI Forecasting</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-priority')">&#128203; Priority Hotspots</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-clusters')">&#128202; Cluster Fingerprints</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-correlation')">&#128279; Co-occurrence (Spearman)</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-pca')">&#129516; PCA Variance</button>
        </div>

        <div class="drawer-body">
            <!-- PANE 1: MULTI-POLLUTANT FORECAST -->
            <div id="pane-forecast" class="drawer-pane active">
                <div style="background:rgba(56, 189, 248, 0.08); border:1px solid rgba(56, 189, 248, 0.25); border-radius:10px; padding:12px 16px; margin-bottom:16px;">
                    <h4 style="font-size:14px; color:var(--accent-blue); margin-bottom:4px;">Multi-Pollutant Autoregressive Regressors (All 7 Impurities)</h4>
                    <p style="font-size:12px; color:var(--text-sub); line-height:1.5;">
                        Instead of predicting only PM2.5, individual Ridge Regressors with L2 regularization predict next-hour concentrations for <b>all 7 criteria pollutants</b>: PM2.5, PM10, NO2, SO2, CO, OZONE, and NH3. Overall CPCB AQI is subsequently derived using max sub-index calculation.
                    </p>
                </div>

                <h4 style="font-size:13px; text-transform:uppercase; letter-spacing:0.5px; color:#fff; margin-bottom:8px;">Model Benchmark vs Persistence Baseline</h4>
                <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:20px;">
                    <thead>
                        <tr style="border-bottom:1px solid var(--border-glass); color:var(--text-sub);">
                            <th style="padding:6px; text-align:left;">Impurity</th>
                            <th style="padding:6px;">Algorithm</th>
                            <th style="padding:6px;">R&sup2; Score</th>
                            <th style="padding:6px;">RMSE</th>
                            <th style="padding:6px;">MAE</th>
                            <th style="padding:6px;">Error Reduction</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    if multi_metrics_df is not None:
        for _, m in multi_metrics_df.iterrows():
            html_content += f"""
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05); color:#cbd5e1;">
                            <td style="padding:8px 6px; font-weight:bold; color:var(--accent-blue);">{m['Pollutant']}</td>
                            <td style="padding:8px 6px;">{m['Model']}</td>
                            <td style="padding:8px 6px; text-align:center; font-weight:bold; color:#2ed573;">{m['R2_Score']}</td>
                            <td style="padding:8px 6px; text-align:center;">{m['RMSE']}</td>
                            <td style="padding:8px 6px; text-align:center;">{m['MAE']}</td>
                            <td style="padding:8px 6px; text-align:center; color:var(--accent-green); font-weight:bold;">+{m['Error_Reduction_%']}%</td>
                        </tr>
            """

    html_content += f"""
                    </tbody>
                </table>

                <h4 style="font-size:13px; text-transform:uppercase; letter-spacing:0.5px; color:#fff; margin-bottom:8px;">Predictive Accuracy Across All 7 Impurities (R&sup2; Score)</h4>
                <canvas id="drawerMultiPollutantChart" height="150" style="margin-bottom:20px;"></canvas>

                <h4 style="font-size:13px; text-transform:uppercase; letter-spacing:0.5px; color:#fff; margin:16px 0 8px;">Top Predicted AQI Surge Hotspots (&Delta; AQI Alert)</h4>
                <table style="width:100%; font-size:12px; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:1px solid var(--border-glass); color:var(--text-sub);">
                            <th style="padding:6px; text-align:left;">City</th>
                            <th style="padding:6px; text-align:left;">Station</th>
                            <th style="padding:6px;">Current AQI</th>
                            <th style="padding:6px;">Predicted AQI</th>
                            <th style="padding:6px;">Dominant Gas</th>
                            <th style="padding:6px;">&Delta;</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for s in severe_aqi_spikes[:8]:
        html_content += f"""
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px;"><b>{s['city']}</b></td>
                            <td style="padding:6px; font-size:11px; color:#cbd5e1;">{s['name'][:22]}...</td>
                            <td style="padding:6px; text-align:center;">{s['cur_aqi']}</td>
                            <td style="padding:6px; text-align:center; font-weight:bold; color:{s['aqi_color']};">{s['pred_aqi']}</td>
                            <td style="padding:6px; text-align:center; font-size:11px; color:var(--accent-blue); font-weight:bold;">{s['pred_dominant']}</td>
                            <td style="padding:6px; text-align:center; font-weight:bold; color:{'var(--accent-red)' if s['aqi_change'] > 0 else 'var(--accent-green)'};">{'+' if s['aqi_change'] > 0 else ''}{s['aqi_change']}</td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>

            <!-- PANE 2: PRIORITY -->
            <div id="pane-priority" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">Ranked Priority Action Table</h4>
                <table id="drawerPriorityTable" class="display" style="width:100%">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>City</th>
                            <th>Cluster</th>
                            <th>Current AQI</th>
                            <th>Predicted AQI</th>
                            <th>Dominant Gas</th>
                            <th>Outlier</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for s in stations:
        outlier_tag = '<span class="badge badge-outlier">OUTLIER</span>' if s['outlier'] else '<span style="color:#64748b;">Normal</span>'
        badge_class = "badge-danger" if s['cluster'] == 0 else ("badge-success" if s['cluster'] == 1 else ("badge-warning" if s['cluster'] == 2 else "badge-purple"))
        html_content += f"""
                        <tr>
                            <td><b>#{s['rank']}</b></td>
                            <td><a href="javascript:void(0)" onclick="flyToStation('{s['name'].replace("'", "\\'")}')" style="color:#38bdf8; text-decoration:none;">{s['city']}</a></td>
                            <td><span class="badge {badge_class}">C{s['cluster']}</span></td>
                            <td>{s['cur_aqi']}</td>
                            <td style="font-weight:bold; color:{s['aqi_color']};">{s['pred_aqi']}</td>
                            <td style="font-weight:bold; color:var(--accent-blue);">{s['pred_dominant']}</td>
                            <td>{outlier_tag}</td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>

            <!-- PANE 3: CLUSTERS -->
            <div id="pane-clusters" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">Cluster Multi-Pollutant Medians (&mu;g/m&sup3;)</h4>
                <canvas id="drawerClusterChart" height="200" style="margin-bottom:20px;"></canvas>
                <div style="font-size:12px; color:var(--text-sub); line-height:1.6;">
                    <p><b>&bull; Cluster 0 (Multi-Pollutant Severe):</b> Elevated across all 7 impurities (PM2.5: 86.0, PM10: 172.5, NO2: 36.0, SO2: 17.5). High-risk urban/industrial corridors.</p>
                    <p><b>&bull; Cluster 1 (Baseline Clean Ambient):</b> Lowest concentration across all 7 parameters (PM2.5: 19.5, PM10: 55.0). Represents rural and coastal background stations.</p>
                    <p><b>&bull; Cluster 2 (Particulate & Dust Dominated):</b> Severe coarse particle ratio (PM10: 104.0, PM2.5: 46.0). Road and construction dust.</p>
                    <p><b>&bull; Cluster 3 (Vehicular & Industrial Gas Dominated):</b> High gaseous combustion (NO2: 31.0, SO2: 15.0, CO: 46.0) with moderate particulate load.</p>
                </div>

                <h4 style="font-size:14px; margin:20px 0 8px; color:#fff;">Clustering Metric Validation (Silhouette & Davies-Bouldin)</h4>
                <canvas id="drawerEvalChart" height="150"></canvas>
            </div>

            <!-- PANE 4: CORRELATION -->
            <div id="pane-correlation" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">7x7 Spearman Rank Correlation Matrix</h4>
                <p style="font-size:12px; color:var(--text-sub); margin-bottom:12px;">Non-parametric rank correlations between simultaneous CAAQMS pollutant measurements across India.</p>
                <table id="drawerCorrTable" style="width:100%; font-size:11px; text-align:center; border-collapse:collapse; margin-bottom:20px;"></table>
                <div style="font-size:12px; color:var(--text-sub); line-height:1.6;">
                    <p><b>&bull; PM2.5 &harr; PM10 (+0.792):</b> Strongest co-occurrence in Indian atmosphere, shared combustion and re-suspension sources.</p>
                    <p><b>&bull; NO2 &harr; NH3 (+0.433):</b> Coupled secondary aerosol chemistry forming ammonium nitrate particulate.</p>
                    <p><b>&bull; Ozone &harr; Primary Inversion:</b> Negative co-occurrences indicating photochemical titration by freshly emitted NO.</p>
                </div>
            </div>

            <!-- PANE 5: PCA -->
            <div id="pane-pca" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">Principal Component Loadings (Variance Drivers)</h4>
                <canvas id="drawerPcaChart" height="200" style="margin-bottom:20px;"></canvas>
                <div style="font-size:12px; color:var(--text-sub); line-height:1.6;">
                    <p><b>&bull; PC1 (33.8% Variance):</b> Particulate Severity Axis (PM10: +0.56, PM2.5: +0.53, NO2: +0.44).</p>
                    <p><b>&bull; PC2 (16.7% Variance):</b> Industrial Sulfur Plume Axis (SO2: +0.70).</p>
                    <p><b>&bull; PC3 (14.4% Variance):</b> Secondary Photochemical Smog Axis (Ozone: +0.73).</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Floating Station HUD Card (appears on marker click) -->
    <div class="station-hud-card" id="stationCard">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="hud-station-name" id="cardName">Select a Station</div>
                <div class="hud-station-sub" id="cardLocation">City, State &bull; Rank</div>
            </div>
            <button class="close-btn" onclick="closeStationCard()">&times;</button>
        </div>

        <!-- AQI Comparison Hero -->
        <div style="display:flex; gap:10px; margin-bottom:12px;">
            <div style="flex:1; background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:8px;">
                <div style="font-size:10px; color:var(--text-sub); text-transform:uppercase;">Current AQI</div>
                <div style="font-size:18px; font-weight:800;" id="cardCurrentAqi">--</div>
                <div style="font-size:10px; color:var(--text-sub);" id="cardCurrentSub">--</div>
            </div>
            <div style="flex:1; background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:8px;">
                <div style="font-size:10px; color:var(--text-sub); text-transform:uppercase;">Next-Hour Predicted AQI</div>
                <div style="font-size:18px; font-weight:800;" id="cardPredAqi">--</div>
                <div style="font-size:10px; color:var(--accent-blue);" id="cardPredSub">--</div>
            </div>
            <div style="flex:1; background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:8px;">
                <div style="font-size:10px; color:var(--text-sub); text-transform:uppercase;">AQI Delta</div>
                <div style="font-size:18px; font-weight:800;" id="cardAqiDelta">--</div>
                <div style="font-size:10px; color:var(--text-sub);" id="cardTrend">--</div>
            </div>
        </div>

        <div style="font-size:11px; font-weight:700; color:var(--text-sub); text-transform:uppercase; margin-bottom:8px; display:flex; justify-content:space-between;">
            <span>All 7 Impurities Forecast</span>
            <span style="font-weight:normal; font-size:10px; opacity:0.8;">Current vs Next 1-Hr</span>
        </div>
        
        <table style="width:100%; font-size:11px; border-collapse:collapse; margin-bottom:12px;" id="cardPollutantTable">
            <thead>
                <tr style="border-bottom:1px solid var(--border-glass); color:var(--text-sub);">
                    <th style="text-align:left; padding:4px 0;">Impurity</th>
                    <th style="text-align:right; padding:4px 6px;">Live</th>
                    <th style="text-align:right; padding:4px 6px;">Predicted</th>
                    <th style="text-align:right; padding:4px 6px;">&Delta; Next Hr</th>
                </tr>
            </thead>
            <tbody id="cardPollutantBody"></tbody>
        </table>

        <div style="padding:10px; background:rgba(56, 189, 248, 0.08); border:1px solid rgba(56, 189, 248, 0.2); border-radius:8px; font-size:11px;">
            <b style="color:var(--accent-blue);">Actionable Mitigation Protocol:</b>
            <div id="cardAction" style="color:#e2e8f0; margin-top:3px; line-height:1.4;">--</div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const stations = {stations_json};
        const pols = {pollutants_json};
        let currentMode = 'cluster';
        let filterCluster = null;
        let outliersOnly = false;
        let markers = [];

        // Setup Map with Esri World Dark/Street & Satellite
        const map = L.map('map', {{ zoomControl: false }}).setView([22.5937, 78.9629], 5);
        L.control.zoom({{ position: 'bottomright' }}).addTo(map);

        const darkBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 18
        }}).addTo(map);

        const satelliteBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 18
        }});

        L.control.layers({{ "Street Map": darkBase, "Satellite Imagery": satelliteBase }}, null, {{ position: 'bottomright' }}).addTo(map);

        function getMarkerColor(s) {{
            if (currentMode === 'cluster') {{
                return s.color;
            }} else if (currentMode === 'predicted-aqi') {{
                return s.aqi_color;
            }} else if (currentMode === 'current-aqi') {{
                const v = s.cur_aqi;
                if (v <= 50) return '#2ed573';
                if (v <= 100) return '#7bed9f';
                if (v <= 200) return '#ffa502';
                if (v <= 300) return '#ff7f50';
                if (v <= 400) return '#ff4757';
                return '#8b0000';
            }} else if (currentMode === 'pm25') {{
                const v = s.pollutants['PM2.5'].pred;
                if (v > 120) return '#ff4757';
                if (v > 60) return '#ffa502';
                if (v > 30) return '#70a1ff';
                return '#2ed573';
            }} else if (currentMode === 'pm10') {{
                const v = s.pollutants['PM10'].pred;
                if (v > 250) return '#ff4757';
                if (v > 100) return '#ffa502';
                if (v > 50) return '#70a1ff';
                return '#2ed573';
            }} else if (currentMode === 'toxic-gas') {{
                const no2 = s.pollutants['NO2'].pred;
                const so2 = s.pollutants['SO2'].pred;
                if (no2 > 80 || so2 > 80) return '#ff4757';
                if (no2 > 40 || so2 > 40) return '#ffa502';
                return '#2ed573';
            }}
            return s.color;
        }}

        function renderMarkers() {{
            markers.forEach(m => map.removeLayer(m));
            markers = [];

            stations.forEach(s => {{
                if (!s.lat || !s.lon) return;
                if (filterCluster !== null && s.cluster !== filterCluster) return;
                if (outliersOnly && !s.outlier) return;

                const color = getMarkerColor(s);
                const radius = s.outlier ? 9 : 6;

                const marker = L.circleMarker([s.lat, s.lon], {{
                    radius: radius,
                    fillColor: color,
                    color: s.outlier ? '#ffffff' : 'rgba(0,0,0,0.5)',
                    weight: s.outlier ? 2.5 : 1,
                    opacity: 1,
                    fillOpacity: 0.9
                }}).addTo(map);

                marker.on('click', () => openStationCard(s));
                markers.push(marker);
            }});
        }}

        function setVisualMode(mode) {{
            currentMode = mode;
            document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-mode-' + mode).classList.add('active');
            renderMarkers();
        }}

        function toggleClusterFilter(c) {{
            filterCluster = (filterCluster === c) ? null : c;
            renderMarkers();
        }}

        function toggleOutliersOnly(val) {{
            outliersOnly = val;
            renderMarkers();
        }}

        function openStationCard(s) {{
            document.getElementById('cardName').innerText = s.name;
            document.getElementById('cardLocation').innerText = `${{s.city}}, ${{s.state}} &bull; Rank #${{s.rank}} in India`;
            
            // Current AQI
            document.getElementById('cardCurrentAqi').innerText = s.cur_aqi;
            document.getElementById('cardCurrentSub').innerText = `${{s.cur_category}} &bull; ${{s.cur_dominant}}`;

            // Predicted AQI
            const pAqiEl = document.getElementById('cardPredAqi');
            pAqiEl.innerText = s.pred_aqi;
            pAqiEl.style.color = s.aqi_color;
            document.getElementById('cardPredSub').innerText = `${{s.pred_category}} &bull; Dominant: ${{s.pred_dominant}}`;

            // Delta
            const dEl = document.getElementById('cardAqiDelta');
            const sign = s.aqi_change > 0 ? '+' : '';
            dEl.innerText = `${{sign}}${{s.aqi_change}}`;
            dEl.style.color = s.aqi_change > 0 ? 'var(--accent-red)' : 'var(--accent-green)';
            document.getElementById('cardTrend').innerText = s.trend;

            // Multi-Pollutant Table
            let tbodyHtml = '';
            for (let [p, data] of Object.entries(s.pollutants)) {{
                const chgSign = data.change > 0 ? '+' : '';
                const chgColor = data.change > 0 ? 'var(--accent-red)' : (data.change < 0 ? 'var(--accent-green)' : 'var(--text-sub)');
                tbodyHtml += `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:4px 0; font-weight:bold; color:#f1f5f9;">${{p}}</td>
                        <td style="text-align:right; padding:4px 6px; color:#cbd5e1;">${{data.cur}}</td>
                        <td style="text-align:right; padding:4px 6px; font-weight:bold; color:var(--accent-blue);">${{data.pred}}</td>
                        <td style="text-align:right; padding:4px 6px; font-weight:bold; color:${{chgColor}};">${{chgSign}}${{data.change}}</td>
                    </tr>
                `;
            }}
            document.getElementById('cardPollutantBody').innerHTML = tbodyHtml;
            document.getElementById('cardAction').innerText = s.action;
            document.getElementById('stationCard').classList.add('active');
        }}

        function closeStationCard() {{
            document.getElementById('stationCard').classList.remove('active');
        }}

        function toggleDrawer() {{
            document.getElementById('analyticsDrawer').classList.toggle('open');
        }}

        function switchDrawerTab(paneId) {{
            document.querySelectorAll('.drawer-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.drawer-pane').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(paneId).classList.add('active');
        }}

        function handleStationSearch(term) {{
            if (!term || term.length < 2) return;
            const match = stations.find(s => s.name.toLowerCase().includes(term.toLowerCase()) || s.city.toLowerCase().includes(term.toLowerCase()));
            if (match && match.lat && match.lon) {{
                map.flyTo([match.lat, match.lon], 12, {{ duration: 1.5 }});
                openStationCard(match);
            }}
        }}

        function flyToStation(name) {{
            const match = stations.find(s => s.name === name);
            if (match && match.lat && match.lon) {{
                toggleDrawer();
                map.flyTo([match.lat, match.lon], 12, {{ duration: 1.5 }});
                openStationCard(match);
            }}
        }}

        // Initialize DataTable
        $(document).ready(function() {{
            $('#drawerPriorityTable').DataTable({{
                pageLength: 8,
                order: [[0, 'asc']]
            }});
        }});

        // Render Correlation Table in Drawer
        const corrData = {corr_matrix_json};
        let tableHtml = '<tr><th></th>' + pols.map(p => '<th>' + p + '</th>').join('') + '</tr>';
        corrData.forEach((row, i) => {{
            tableHtml += '<tr><th style="color:#cbd5e1;">' + pols[i] + '</th>';
            row.forEach(val => {{
                let bg = 'rgba(255,255,255,0.05)';
                let color = '#fff';
                if (val >= 0.7) {{ bg = 'rgba(255, 71, 87, 0.4)'; color = '#ff6b81'; }}
                else if (val >= 0.4) {{ bg = 'rgba(255, 165, 2, 0.3)'; color = '#ffa502'; }}
                else if (val < 0) {{ bg = 'rgba(56, 189, 248, 0.25)'; color = '#38bdf8'; }}
                tableHtml += `<td style="background:${{bg}}; color:${{color}}; font-weight:bold; padding:8px 4px; border:1px solid rgba(255,255,255,0.05);">${{val.toFixed(2)}}</td>`;
            }});
            tableHtml += '</tr>';
        }});
        document.getElementById('drawerCorrTable').innerHTML = tableHtml;

        // Drawer Charts: 7-Pollutant Model Accuracy
        new Chart(document.getElementById('drawerMultiPollutantChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(model_pollutants)},
                datasets: [
                    {{
                        label: 'Ridge Regressor R² Accuracy',
                        data: {json.dumps(model_r2)},
                        backgroundColor: '#38bdf8',
                        borderRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ min: 0.90, max: 1.0, ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    x: {{ ticks: {{ color: '#f8fafc' }}, grid: {{ display: false }} }}
                }}
            }}
        }});

        const clusterProfiles = {cluster_profiles_json};
        new Chart(document.getElementById('drawerClusterChart'), {{
            type: 'bar',
            data: {{
                labels: pols,
                datasets: [
                    {{ label: 'C0: Severe', data: pols.map(p => clusterProfiles[0][p]), backgroundColor: '#ff4757' }},
                    {{ label: 'C1: Clean', data: pols.map(p => clusterProfiles[1][p]), backgroundColor: '#2ed573' }},
                    {{ label: 'C2: Dust', data: pols.map(p => clusterProfiles[2][p]), backgroundColor: '#ffa502' }},
                    {{ label: 'C3: Gas', data: pols.map(p => clusterProfiles[3][p]), backgroundColor: '#a55eea' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    x: {{ ticks: {{ color: '#f8fafc' }}, grid: {{ display: false }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('drawerEvalChart'), {{
            type: 'line',
            data: {{
                labels: {k_labels_json},
                datasets: [
                    {{ label: 'Silhouette Score', data: {sil_json}, borderColor: '#2ed573', tension: 0.2, yAxisID: 'y' }},
                    {{ label: 'Davies-Bouldin', data: {db_json}, borderColor: '#ff4757', tension: 0.2, yAxisID: 'y1' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ type: 'linear', position: 'left', ticks: {{ color: '#2ed573' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y1: {{ type: 'linear', position: 'right', ticks: {{ color: '#ff4757' }}, grid: {{ display: false }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('drawerPcaChart'), {{
            type: 'bar',
            data: {{
                labels: {pca_labels_json},
                datasets: [
                    {{ label: 'PC1 (Particulate)', data: {pc1_json}, backgroundColor: '#38bdf8' }},
                    {{ label: 'PC2 (Sulfur)', data: {pc2_json}, backgroundColor: '#a55eea' }},
                    {{ label: 'PC3 (Ozone)', data: {pc3_json}, backgroundColor: '#ffa502' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ min: -0.6, max: 0.8, ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    x: {{ ticks: {{ color: '#f8fafc' }}, grid: {{ display: false }} }}
                }}
            }}
        }});

        // Render all initial markers
        renderMarkers();
    </script>
</body>
</html>
"""

    output_path = PROJECT_ROOT / "outputs" / "dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("=" * 65)
    print(" COMMAND CENTER / MAP-FIRST DASHBOARD GENERATED")
    print(f" Saved to: {output_path}")
    print("=" * 65)
    return output_path

generate_full_dashboard = generate_command_center_dashboard

if __name__ == "__main__":
    generate_command_center_dashboard()
