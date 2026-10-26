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

def generate_command_center_dashboard():
    clustered_file = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    spearman_file = PROJECT_ROOT / "outputs" / "reports" / "spearman_correlation_matrix.csv"
    loadings_file = PROJECT_ROOT / "outputs" / "reports" / "pca_component_loadings.csv"
    eval_file = PROJECT_ROOT / "outputs" / "reports" / "clustering_metrics_evaluation.csv"
    cluster_medians_file = PROJECT_ROOT / "outputs" / "reports" / "cluster_profiles_summary.csv"
    
    forecast_file = PROCESSED_DATA_DIR / "station_next_hour_forecasts.csv"
    forecast_metrics_file = PROJECT_ROOT / "outputs" / "reports" / "forecasting_model_metrics.csv"
    forecast_importance_file = PROJECT_ROOT / "outputs" / "reports" / "forecasting_feature_importance.csv"

    df = pd.read_csv(clustered_file)
    spearman_df = pd.read_csv(spearman_file, index_col=0)
    loadings_df = pd.read_csv(loadings_file, index_col=0)
    eval_df = pd.read_csv(eval_file)
    cluster_medians_df = pd.read_csv(cluster_medians_file, index_col=0)

    forecast_df = pd.read_csv(forecast_file) if forecast_file.exists() else None
    forecast_metrics_df = pd.read_csv(forecast_metrics_file) if forecast_metrics_file.exists() else None
    forecast_importance_df = pd.read_csv(forecast_importance_file) if forecast_importance_file.exists() else None

    # Merge forecast details into master dataframe
    if forecast_df is not None:
        f_sub = forecast_df[['station', 'predicted_next_pm25', 'predicted_change', 'forecast_trend', 'predicted_aqi_category']]
        df = df.merge(f_sub, on='station', how='left')
    else:
        df['predicted_next_pm25'] = df['PM2.5']
        df['predicted_change'] = 0.0
        df['forecast_trend'] = 'Stable'
        df['predicted_aqi_category'] = 'Moderate'

    cluster_colors = {
        0: "#ff4757",  # Red: Multi-pollutant severe
        1: "#2ed573",  # Green: Clean ambient baseline
        2: "#ffa502",  # Orange: Particulate & dust dominated
        3: "#a55eea"   # Purple: Industrial sulfur & gas dominated
    }

    stations = []
    for _, row in df.iterrows():
        action = "General air quality monitoring & public awareness"
        if row['PM10'] > 200 or row['PM2.5'] > 150:
            action = "Deploy anti-smog guns, mechanized road sweeping, enforce construction dust cover"
        elif row['NO2'] > 60 or row['CO'] > 50:
            action = "Heavy diesel vehicle bypass rerouting, congestion pricing, EV transit priority"
        elif row['SO2'] > 30:
            action = "Industrial stack emission audit, flue-gas desulfurization (FGD) verification, PNG switch"
        elif row['OZONE'] > 50:
            action = "Regulate volatile organic solvent storage and refueling vapor emissions during midday"

        pred_pm25 = float(row['predicted_next_pm25']) if pd.notna(row['predicted_next_pm25']) else float(row['PM2.5'])
        pred_change = float(row['predicted_change']) if pd.notna(row['predicted_change']) else 0.0

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
            "pm25": float(row['PM2.5']),
            "pm10": float(row['PM10']),
            "no2": float(row['NO2']),
            "so2": float(row['SO2']),
            "co": float(row['CO']),
            "o3": float(row['OZONE']),
            "nh3": float(row['NH3']),
            "pred_pm25": pred_pm25,
            "pred_change": pred_change,
            "trend": str(row.get('forecast_trend', 'Stable')),
            "pred_category": str(row.get('predicted_aqi_category', 'Moderate')),
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

    forecast_importance_labels_json = json.dumps(forecast_importance_df['Feature'].head(8).tolist() if forecast_importance_df is not None else [])
    forecast_importance_vals_json = json.dumps(forecast_importance_df['Importance'].head(8).tolist() if forecast_importance_df is not None else [])

    severe_spikes = [s for s in stations if s['pred_change'] >= 15]
    severe_category_count = sum(1 for s in stations if s['pred_pm25'] > 120)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroPulse | National Air Pollution Command Center</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" />
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-glass: rgba(15, 23, 42, 0.88);
            --bg-glass-card: rgba(30, 41, 59, 0.82);
            --border-glass: rgba(255, 255, 255, 0.12);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-red: #ff4757;
            --accent-green: #2ed573;
            --accent-orange: #ffa502;
            --accent-purple: #a55eea;
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
            padding: 12px 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.45);
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
        .search-container {{ flex: 0 1 380px; position: relative; }}
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
            padding: 16px; width: 270px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            font-size: 12px;
        }}
        .deck-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-sub); margin-bottom: 10px; font-weight: 700; }}
        .layer-btn {{
            width: 100%; display: flex; justify-content: space-between; align-items: center;
            background: rgba(255, 255, 255, 0.05); border: 1px solid transparent; border-radius: 6px;
            padding: 9px 12px; color: #fff; font-size: 12px; cursor: pointer; margin-bottom: 6px; transition: all 0.2s;
        }}
        .layer-btn:hover {{ background: rgba(255, 255, 255, 0.1); }}
        .layer-btn.active {{ background: rgba(56, 189, 248, 0.18); border-color: var(--accent-blue); color: var(--accent-blue); font-weight: 700; }}

        .cluster-legend-item {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 6px 4px; font-size: 11px; cursor: pointer; opacity: 0.9;
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
            box-shadow: 0 4px 20px rgba(0,0,0,0.35); transition: all 0.2s;
        }}
        .drawer-toggle-btn:hover {{ border-color: var(--accent-blue); transform: translateY(-1px); }}

        /* Right Slide-over Analytics Drawer */
        .analytics-drawer {{
            position: absolute; top: 16px; bottom: 16px; right: -650px; width: 620px; z-index: 1001;
            background: rgba(11, 15, 25, 0.94); backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px);
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
            position: absolute; bottom: 25px; left: 16px; z-index: 1000; width: 380px;
            background: var(--bg-glass); backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
            border: 1px solid var(--border-glass); border-radius: 14px;
            padding: 18px; box-shadow: 0 12px 40px rgba(0,0,0,0.5); display: none;
            animation: slideUp 0.3s ease-out;
        }}
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        .station-hud-card.active {{ display: block; }}
        .hud-station-name {{ font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 2px; }}
        .hud-station-sub {{ font-size: 12px; color: var(--text-sub); margin-bottom: 12px; }}
        .gauge-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 11px; }}
        .gauge-bar {{ flex: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; margin: 0 10px; overflow: hidden; }}
        .gauge-fill {{ height: 100%; border-radius: 3px; }}

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
        table.dataTable tbody tr {{ background: transparent !important; color: #e2e8f0; }}
        table.dataTable tbody tr:hover {{ background: rgba(255,255,255,0.05) !important; }}
        table.dataTable tbody td {{ border-top: 1px solid rgba(255,255,255,0.05) !important; }}

        /* Custom Leaflet Dark adjustments */
        .leaflet-container {{ background: #0b0f19 !important; }}
        .leaflet-control-layers {{ background: var(--bg-glass) !important; color: #fff !important; border: 1px solid var(--border-glass) !important; border-radius: 8px !important; }}
    </style>
</head>
<body>

    <!-- Fullscreen Map Canvas -->
    <div id="map"></div>

    <!-- Top Floating HUD -->
    <div class="hud-header">
        <div class="brand">
            <div class="pulse-indicator"></div>
            <div>
                <div class="brand-title">AeroPulse &bull; CPCB Command Center</div>
                <div class="brand-subtitle">508 Stations &bull; Machine Learning Airshed Intelligence</div>
            </div>
        </div>

        <div class="search-container">
            <span class="search-icon">🔍</span>
            <input type="text" id="stationSearch" class="search-input" placeholder="Search station, city, or state (e.g. Delhi, Ahmedabad)..." onkeyup="handleStationSearch(this.value)">
        </div>

        <div class="hud-metrics">
            <div class="metric-pill">
                <span>⚡ Severe Hotspots:</span>
                <b style="color:var(--accent-red);">{(df['cluster']==0).sum()}</b>
            </div>
            <div class="metric-pill">
                <span>🔮 Predicted Spikes:</span>
                <b style="color:var(--accent-orange);">{len(severe_spikes)}</b>
            </div>
            <div class="metric-pill">
                <span>🚨 Outliers:</span>
                <b style="color:#ff6b81;">{df['is_outlier'].sum()}</b>
            </div>
        </div>
    </div>

    <!-- Left Control Deck -->
    <div class="control-deck">
        <div class="deck-title">Visualization Mode</div>
        <button class="layer-btn active" id="btn-mode-cluster" onclick="setVisualMode('cluster')">
            <span>🏷️ ML Cluster Profile</span>
            <span style="font-size:10px;">4 Profiles</span>
        </button>
        <button class="layer-btn" id="btn-mode-current" onclick="setVisualMode('current')">
            <span>💨 Current PM2.5 Level</span>
            <span style="font-size:10px;">Recorded</span>
        </button>
        <button class="layer-btn" id="btn-mode-predicted" onclick="setVisualMode('predicted')">
            <span>🔮 Predicted Next-Hour</span>
            <span style="font-size:10px;">ML Forecast</span>
        </button>

        <hr style="border:0; border-top:1px solid var(--border-glass); margin:12px 0;">

        <div class="deck-title">Pollution Clusters</div>
        <div class="cluster-legend-item" onclick="toggleClusterFilter(0)">
            <div><span class="cluster-dot" style="background:var(--accent-red);"></span>Cluster 0: Severe Hotspot</div>
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
            <div><span class="cluster-dot" style="background:var(--accent-purple);"></span>Cluster 3: Industrial Gas</div>
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
        <span>📊 Analytics & ML Models</span>
        <span style="font-size:11px; opacity:0.7;">▶</span>
    </button>

    <!-- Floating Slide-over Analytics Drawer -->
    <div class="analytics-drawer" id="analyticsDrawer">
        <div class="drawer-header">
            <h3>Machine Learning Analytics & Insights</h3>
            <button class="close-btn" onclick="toggleDrawer()">&times;</button>
        </div>

        <div class="drawer-nav">
            <button class="drawer-tab active" onclick="switchDrawerTab('pane-forecast')">🔮 Next-Hour Forecast</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-priority')">📋 Priority Hotspots</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-clusters')">📊 Cluster Fingerprints</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-correlation')">🔗 Co-occurrence (Spearman)</button>
            <button class="drawer-tab" onclick="switchDrawerTab('pane-pca')">🧬 PCA Variance</button>
        </div>

        <div class="drawer-body">
            <!-- PANE 1: FORECAST -->
            <div id="pane-forecast" class="drawer-pane active">
                <h4 style="font-size:14px; margin-bottom:8px; color:var(--accent-blue);">Model Performance on Live CAAQMS Transitions</h4>
                <p style="font-size:12px; color:var(--text-sub); margin-bottom:12px;">
                    Autoregressive time-series regression with lag features ($t, t-1$), momentum ($\Delta PM_{{2.5}}$), particulate ratios, and co-occurring toxic gases.
                </p>

                <table style="width:100%; font-size:12px; border-collapse:collapse; margin-bottom:20px;">
                    <thead>
                        <tr style="border-bottom:1px solid var(--border-glass); color:var(--text-sub);">
                            <th style="padding:6px; text-align:left;">Model</th>
                            <th style="padding:6px;">R&sup2; Score</th>
                            <th style="padding:6px;">RMSE</th>
                            <th style="padding:6px;">MAE</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    if forecast_metrics_df is not None:
        for _, m in forecast_metrics_df.iterrows():
            is_best = m['Model'].startswith("Ridge") or m['Model'].startswith("Hist")
            row_color = "color:#38bdf8; font-weight:bold;" if is_best else "color:#cbd5e1;"
            html_content += f"""
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05); {row_color}">
                            <td style="padding:8px 6px;">{m['Model']}</td>
                            <td style="padding:8px 6px; text-align:center;">{m['R2_Score']}</td>
                            <td style="padding:8px 6px; text-align:center;">{m['RMSE (ug/m3)']}</td>
                            <td style="padding:8px 6px; text-align:center;">{m['MAE (ug/m3)']}</td>
                        </tr>
            """

    html_content += f"""
                    </tbody>
                </table>

                <h4 style="font-size:14px; margin:16px 0 8px; color:#fff;">Feature Importance in Next-Hour Prediction</h4>
                <canvas id="drawerFeatureChart" height="150" style="margin-bottom:20px;"></canvas>

                <h4 style="font-size:14px; margin:16px 0 8px; color:#fff;">Top Next-Hour Surge Alerts (&Delta; &ge; +15 &mu;g/m&sup3;)</h4>
                <table style="width:100%; font-size:12px; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:1px solid var(--border-glass); color:var(--text-sub);">
                            <th style="padding:6px; text-align:left;">City</th>
                            <th style="padding:6px; text-align:left;">Station</th>
                            <th style="padding:6px;">Current</th>
                            <th style="padding:6px;">Next Hour</th>
                            <th style="padding:6px;">&Delta;</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for s in severe_spikes[:8]:
        html_content += f"""
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px;"><b>{s['city']}</b></td>
                            <td style="padding:6px; font-size:11px; color:#cbd5e1;">{s['name'][:24]}...</td>
                            <td style="padding:6px; text-align:center;">{s['pm25']}</td>
                            <td style="padding:6px; text-align:center; font-weight:bold; color:var(--accent-red);">{s['pred_pm25']}</td>
                            <td style="padding:6px; text-align:center; font-weight:bold; color:var(--accent-red);">+{s['pred_change']}</td>
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
                            <th>PM2.5</th>
                            <th>PM10</th>
                            <th>Outlier</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    for s in stations:
        outlier_tag = '<span class="badge badge-outlier">OUTLIER</span>' if s['outlier'] else '<span style="color:#64748b;">Normal</span>'
        badge_class = "badge-danger" if s['cluster'] == 0 else ("badge-success" if s['cluster'] == 1 else ("badge-warning" if s['cluster'] == 2 else "badge-purple"))
        html_content += f"""
                        <tr onclick="flyToStation('{s['name']}')" style="cursor:pointer;">
                            <td><b>#{s['rank']}</b></td>
                            <td><b>{s['city']}</b></td>
                            <td><span class="badge {badge_class}">C{s['cluster']}</span></td>
                            <td>{s['pm25']}</td>
                            <td>{s['pm10']}</td>
                            <td>{outlier_tag}</td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>

            <!-- PANE 3: CLUSTERS -->
            <div id="pane-clusters" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">Cluster Pollutant Fingerprints (Median &mu;g/m&sup3;)</h4>
                <canvas id="drawerClusterChart" height="200" style="margin-bottom:20px;"></canvas>

                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">K-Means Evaluation (Elbow & Silhouette)</h4>
                <canvas id="drawerEvalChart" height="150"></canvas>
            </div>

            <!-- PANE 4: CORRELATION -->
            <div id="pane-correlation" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">Spearman Rank Correlation Heatmap</h4>
                <table class="corr-table" id="drawerCorrTable" style="margin-bottom:15px;"></table>
                <div style="font-size:12px; color:var(--text-sub); line-height:1.6;">
                    <p><b>&bull; PM2.5 & PM10 (r = +0.804):</b> Extremely high particulate coupling across Indian airsheds.</p>
                    <p><b>&bull; NO2 & NH3 (r = +0.446):</b> Urban traffic, catalytic exhaust, and domestic combustion co-occurrence.</p>
                    <p><b>&bull; CO & OZONE (r = -0.106):</b> Photochemical dissociation and precursor consumption in sunlight.</p>
                </div>
            </div>

            <!-- PANE 5: PCA -->
            <div id="pane-pca" class="drawer-pane">
                <h4 style="font-size:14px; margin-bottom:8px; color:#fff;">Principal Component Loadings (Variance Drivers)</h4>
                <canvas id="drawerPcaChart" height="200" style="margin-bottom:20px;"></canvas>
                <div style="font-size:12px; color:var(--text-sub); line-height:1.6;">
                    <p><b>&bull; PC1 (33.5% Variance):</b> Particulate Severity Axis (PM10: +0.56, PM2.5: +0.53, NO2: +0.44).</p>
                    <p><b>&bull; PC2 (16.6% Variance):</b> Industrial Sulfur Plume Axis (SO2: +0.70).</p>
                    <p><b>&bull; PC3 (14.8% Variance):</b> Secondary Photochemical Smog Axis (Ozone: +0.73).</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Floating Station HUD Card (appears on marker click) -->
    <div class="station-hud-card" id="stationCard">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="hud-station-name" id="cardName">Select a Station</div>
                <div class="hud-station-sub" id="cardLocation">City, State</div>
            </div>
            <button class="close-btn" onclick="closeStationCard()">&times;</button>
        </div>

        <div style="display:flex; gap:10px; margin-bottom:12px;">
            <div style="flex:1; background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:8px;">
                <div style="font-size:10px; color:var(--text-sub); text-transform:uppercase;">Current PM2.5</div>
                <div style="font-size:18px; font-weight:800;" id="cardCurrentPm25">--</div>
            </div>
            <div style="flex:1; background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:8px;">
                <div style="font-size:10px; color:var(--text-sub); text-transform:uppercase;">Next-Hour Forecast</div>
                <div style="font-size:18px; font-weight:800; color:var(--accent-blue);" id="cardPredPm25">--</div>
            </div>
            <div style="flex:1; background:rgba(255,255,255,0.06); padding:8px 10px; border-radius:8px;">
                <div style="font-size:10px; color:var(--text-sub); text-transform:uppercase;">Expected Change</div>
                <div style="font-size:18px; font-weight:800;" id="cardChange">--</div>
            </div>
        </div>

        <div style="font-size:11px; font-weight:700; color:var(--text-sub); text-transform:uppercase; margin-bottom:8px;">Multi-Pollutant Fingerprint (&mu;g/m&sup3;)</div>
        <div id="cardGauges"></div>

        <div style="margin-top:12px; padding:10px; background:rgba(56, 189, 248, 0.08); border:1px solid rgba(56, 189, 248, 0.2); border-radius:8px; font-size:11px;">
            <b style="color:var(--accent-blue);">Recommended Action:</b>
            <div id="cardAction" style="color:#e2e8f0; margin-top:3px;">--</div>
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
            }} else if (currentMode === 'current') {{
                const val = s.pm25;
                if (val > 150) return '#ff4757';
                if (val > 90) return '#ffa502';
                if (val > 60) return '#eccc68';
                if (val > 30) return '#70a1ff';
                return '#2ed573';
            }} else {{
                // Predicted mode
                const val = s.pred_pm25;
                if (val > 150) return '#ff4757';
                if (val > 90) return '#ffa502';
                if (val > 60) return '#eccc68';
                if (val > 30) return '#70a1ff';
                return '#2ed573';
            }}
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
            document.getElementById('cardCurrentPm25').innerText = s.pm25;
            document.getElementById('cardPredPm25').innerText = s.pred_pm25;
            
            const sign = s.pred_change > 0 ? '+' : '';
            const changeColor = s.pred_change > 0 ? 'var(--accent-red)' : 'var(--accent-green)';
            const chgEl = document.getElementById('cardChange');
            chgEl.innerText = `${{sign}}${{s.pred_change}}`;
            chgEl.style.color = changeColor;

            // Gauges
            const limits = {{ 'PM2.5': 60, 'PM10': 100, 'NO2': 80, 'SO2': 80, 'CO': 4, 'O3': 100, 'NH3': 400 }};
            let gaugeHtml = '';
            const values = {{ 'PM2.5': s.pm25, 'PM10': s.pm10, 'NO2': s.no2, 'SO2': s.so2, 'CO': s.co, 'O3': s.o3, 'NH3': s.nh3 }};
            
            for (let [p, val] of Object.entries(values)) {{
                const lim = limits[p];
                const pct = Math.min((val / (lim * 2)) * 100, 100);
                const barColor = val > lim ? 'var(--accent-red)' : (val > lim * 0.7 ? 'var(--accent-orange)' : 'var(--accent-green)');
                gaugeHtml += `
                    <div class="gauge-row">
                        <span style="width:45px; font-weight:700;">${{p}}</span>
                        <div class="gauge-bar"><div class="gauge-fill" style="width:${{pct}}%; background:${{barColor}};"></div></div>
                        <span style="width:50px; text-align:right;">${{val}}</span>
                    </div>
                `;
            }}
            document.getElementById('cardGauges').innerHTML = gaugeHtml;
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

        // Drawer Charts
        new Chart(document.getElementById('drawerFeatureChart'), {{
            type: 'bar',
            data: {{
                labels: {forecast_importance_labels_json},
                datasets: [{{ label: 'Relative Feature Importance', data: {forecast_importance_vals_json}, backgroundColor: '#38bdf8' }}]
            }},
            options: {{
                indexAxis: 'y', responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ ticks: {{ color: '#f8fafc' }}, grid: {{ display: false }} }}
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

if __name__ == "__main__":
    generate_command_center_dashboard()
