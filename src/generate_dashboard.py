import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']

def generate_full_dashboard():
    clustered_file = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    spearman_file = PROJECT_ROOT / "outputs" / "reports" / "spearman_correlation_matrix.csv"
    loadings_file = PROJECT_ROOT / "outputs" / "reports" / "pca_component_loadings.csv"
    eval_file = PROJECT_ROOT / "outputs" / "reports" / "clustering_metrics_evaluation.csv"
    cluster_medians_file = PROJECT_ROOT / "outputs" / "reports" / "cluster_profiles_summary.csv"
    
    df = pd.read_csv(clustered_file)
    spearman_df = pd.read_csv(spearman_file, index_col=0)
    loadings_df = pd.read_csv(loadings_file, index_col=0)
    eval_df = pd.read_csv(eval_file)
    cluster_medians_df = pd.read_csv(cluster_medians_file, index_col=0)

    # Prepare station data for Map & Table
    cluster_colors = {0: "#e74c3c", 1: "#2ecc71", 2: "#f39c12", 3: "#9b59b6"}
    
    stations = []
    for _, row in df.iterrows():
        # Policy action rule
        action = "General air quality monitoring"
        if row['PM10'] > 200 or row['PM2.5'] > 150:
            action = "Deploy anti-smog guns, mechanised road sweeping, construction dust ban"
        elif row['NO2'] > 60 or row['CO'] > 50:
            action = "Heavy vehicle entry restriction, odd-even traffic rationing, EV corridor"
        elif row['SO2'] > 30:
            action = "Industrial emission audit, stack scrubber verification, fuel switch to PNG"
        elif row['OZONE'] > 50:
            action = "Regulate VOC and fuel vapor emissions during peak sunlight hours"

        stations.append({
            "rank": int(row['priority_rank']),
            "name": str(row['station']),
            "city": str(row['city']),
            "state": str(row['state']),
            "lat": float(row['latitude']) if pd.notna(row['latitude']) else None,
            "lon": float(row['longitude']) if pd.notna(row['longitude']) else None,
            "cluster": int(row['cluster']),
            "profile": str(row['cluster_profile']),
            "color": cluster_colors.get(int(row['cluster']), "#34495e"),
            "outlier": bool(row['is_outlier']),
            "pm25": float(row['PM2.5']),
            "pm10": float(row['PM10']),
            "no2": float(row['NO2']),
            "so2": float(row['SO2']),
            "co": float(row['CO']),
            "o3": float(row['OZONE']),
            "nh3": float(row['NH3']),
            "action": action
        })

    # Prepare JSON serializations
    stations_json = json.dumps(stations)
    pollutants_json = json.dumps(TARGET_POLLUTANTS)
    
    # Spearman matrix serialization
    corr_matrix_json = json.dumps(spearman_df.values.tolist())
    
    # PCA Loadings
    pca_labels_json = json.dumps(loadings_df.index.tolist())
    pc1_json = json.dumps(loadings_df['PC1_Loading'].tolist())
    pc2_json = json.dumps(loadings_df['PC2_Loading'].tolist())
    pc3_json = json.dumps(loadings_df['PC3_Loading'].tolist())
    
    # Cluster Medians
    cluster_profiles_json = json.dumps(cluster_medians_df.to_dict(orient='index'))
    
    # Evaluation Metrics
    k_labels_json = json.dumps(eval_df['k'].tolist())
    sil_json = json.dumps(eval_df['silhouette_score'].tolist())
    db_json = json.dumps(eval_df['davies_bouldin_index'].tolist())
    inertia_json = json.dumps(eval_df['inertia'].tolist())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Air Pollution Hotspot Profiling Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" />
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #1e3799;
            --accent: #4a69bd;
            --dark: #2c3e50;
            --light: #f8f9fa;
            --danger: #e74c3c;
            --warning: #f39c12;
            --success: #2ecc71;
            --purple: #9b59b6;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f1f2f6; color: #2f3542; }}
        
        /* Header */
        header {{ background: linear-gradient(135deg, #1e3799, #0c2461); color: white; padding: 22px 35px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        header h1 {{ margin: 0 0 6px; font-size: 24px; font-weight: 700; }}
        header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .header-meta {{ margin-top: 10px; font-size: 12px; opacity: 0.8; display: flex; gap: 20px; }}

        /* KPI Cards */
        .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 15px; padding: 20px 35px 0; }}
        .kpi-card {{ background: white; padding: 18px 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 5px solid var(--primary); }}
        .kpi-card h4 {{ margin: 0 0 5px; font-size: 12px; color: #747d8c; text-transform: uppercase; letter-spacing: 0.5px; }}
        .kpi-card .value {{ font-size: 26px; font-weight: bold; color: var(--dark); }}
        .kpi-card .subtext {{ font-size: 11px; color: #a4b0be; margin-top: 3px; }}

        /* Tab Navigation */
        .nav-tabs {{ display: flex; gap: 8px; padding: 20px 35px 0; border-bottom: 2px solid #dcdde1; }}
        .tab-btn {{ background: transparent; border: none; padding: 12px 20px; font-size: 14px; font-weight: 600; color: #57606f; cursor: pointer; border-radius: 8px 8px 0 0; transition: all 0.2s; }}
        .tab-btn.active {{ background: white; color: var(--primary); border-bottom: 3px solid var(--primary); box-shadow: 0 -2px 6px rgba(0,0,0,0.04); }}
        .tab-btn:hover:not(.active) {{ background: #e4e7eb; }}

        /* Content Container */
        .tab-content {{ display: none; padding: 20px 35px 40px; }}
        .tab-content.active {{ display: block; }}

        /* Card panels */
        .panel {{ background: white; border-radius: 10px; padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; }}
        .panel h3 {{ margin: 0 0 15px; font-size: 17px; color: var(--dark); border-bottom: 2px solid #f1f2f6; padding-bottom: 8px; }}

        /* Map styling */
        #map {{ height: 580px; width: 100%; border-radius: 8px; }}

        /* Grid */
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        /* Badges */
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block; }}
        .badge-danger {{ background: #ffebee; color: #c62828; }}
        .badge-warning {{ background: #fff8e1; color: #f57f17; }}
        .badge-success {{ background: #e8f8f5; color: #27ae60; }}
        .badge-purple {{ background: #f3e5f5; color: #8e24aa; }}
        .badge-outlier {{ background: #c0392b; color: white; }}

        /* Correlation Heatmap */
        .corr-table {{ width: 100%; border-collapse: collapse; text-align: center; font-size: 13px; }}
        .corr-table th, .corr-table td {{ padding: 12px; border: 1px solid #dfe4ea; }}
        .corr-table th {{ background: #f1f2f6; font-weight: 600; }}

        /* Table styling */
        #priorityTable {{ font-size: 12px; }}
        #priorityTable th {{ background: #f8f9fa; }}
    </style>
</head>
<body>

    <header>
        <h1>Air Pollution Hotspot Profiling & Co-occurrence Analysis</h1>
        <p>Nation-wide Continuous Ambient Air Quality Monitoring (CPCB Data) &bull; Machine Learning Analytics Dashboard</p>
        <div class="header-meta">
            <span><b>Project by:</b> Aditya (24PT27) & Mukunthan (24PT19)</span>
            <span><b>Model:</b> K-Means Clustering + PCA + Outlier Profiling</span>
            <span><b>Coverage:</b> 507 Stations &bull; 270 Cities &bull; 31 States</span>
        </div>
    </header>

    <!-- Key Performance Indicators -->
    <div class="kpi-row">
        <div class="kpi-card" style="border-left-color: var(--primary);">
            <h4>Active Stations</h4>
            <div class="value">{len(stations)}</div>
            <div class="subtext">Across 31 States & Union Territories</div>
        </div>
        <div class="kpi-card" style="border-left-color: var(--danger);">
            <h4>Critical Hotspots</h4>
            <div class="value">{(df['cluster'] == 0).sum()}</div>
            <div class="subtext">Cluster 0: Multi-Pollutant Severe</div>
        </div>
        <div class="kpi-card" style="border-left-color: #c0392b;">
            <h4>Statistical Outliers</h4>
            <div class="value">{df['is_outlier'].sum()}</div>
            <div class="subtext">Exceeding 95th Percentile Deviation</div>
        </div>
        <div class="kpi-card" style="border-left-color: var(--warning);">
            <h4>Optimal Clusters</h4>
            <div class="value">K = 4</div>
            <div class="subtext">Verified by Silhouette & Davies-Bouldin</div>
        </div>
        <div class="kpi-card" style="border-left-color: var(--success);">
            <h4>Baseline Clean Stations</h4>
            <div class="value">{(df['cluster'] == 1).sum()}</div>
            <div class="subtext">Cluster 1: Relatively Low Ambient Air</div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
        <button class="tab-btn active" onclick="openTab('tab-map')"> Hotspot Map</button>
        <button class="tab-btn" onclick="openTab('tab-priority')"> Priority Ranking & Policy Actions</button>
        <button class="tab-btn" onclick="openTab('tab-clusters')"> Cluster Profiles & Fingerprints</button>
        <button class="tab-btn" onclick="openTab('tab-correlation')"> Co-occurrence (Spearman)</button>
        <button class="tab-btn" onclick="openTab('tab-pca')"> PCA Drivers (Variance)</button>
    </div>

    <!-- TAB 1: MAP -->
    <div id="tab-map" class="tab-content active">
        <div class="panel">
            <h3>Geospatial Pollution Hotspots & Outlier Distribution</h3>
            <p style="font-size: 13px; color: #666; margin-top: -8px; margin-bottom: 12px;">
                Stations are colored by their Machine Learning Cluster Profile. <b>Thick black outlines</b> indicate statistical outlier stations. Click any station marker to inspect multi-pollutant concentrations.
            </p>
            <div id="map"></div>
        </div>
    </div>

    <!-- TAB 2: PRIORITY RANKING TABLE -->
    <div id="tab-priority" class="tab-content">
        <div class="panel">
            <h3>Ranked Priority List & Targeted Administrative Interventions</h3>
            <p style="font-size: 13px; color: #666; margin-top: -8px; margin-bottom: 15px;">
                Stations are prioritized by a composite severity score (particulates + toxic gases + outlier distance). Authorities can search by city/state or filter by policy action.
            </p>
            <table id="priorityTable" class="display" style="width:100%">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>City</th>
                        <th>State</th>
                        <th>Station Name</th>
                        <th>Cluster Profile</th>
                        <th>PM2.5</th>
                        <th>PM10</th>
                        <th>NO2</th>
                        <th>SO2</th>
                        <th>Outlier</th>
                        <th>Recommended Intervention</th>
                    </tr>
                </thead>
                <tbody>
    """

    for s in stations:
        outlier_tag = '<span class="badge badge-outlier">OUTLIER</span>' if s['outlier'] else '<span style="color:#aaa;">Normal</span>'
        badge_class = "badge-danger" if s['cluster'] == 0 else ("badge-success" if s['cluster'] == 1 else ("badge-warning" if s['cluster'] == 2 else "badge-purple"))
        
        html_content += f"""
                    <tr>
                        <td><b>#{s['rank']}</b></td>
                        <td><b>{s['city']}</b></td>
                        <td>{s['state']}</td>
                        <td>{s['name']}</td>
                        <td><span class="badge {badge_class}">{s['profile']}</span></td>
                        <td>{s['pm25']}</td>
                        <td>{s['pm10']}</td>
                        <td>{s['no2']}</td>
                        <td>{s['so2']}</td>
                        <td>{outlier_tag}</td>
                        <td style="font-size:11px; font-weight:600; color:#2c3e50;">{s['action']}</td>
                    </tr>
        """

    html_content += f"""
                </tbody>
            </table>
        </div>
    </div>

    <!-- TAB 3: CLUSTERS -->
    <div id="tab-clusters" class="tab-content">
        <div class="grid-2">
            <div class="panel">
                <h3>Cluster Pollutant Fingerprints (Median &mu;g/m&sup3;)</h3>
                <canvas id="clusterBarChart" height="230"></canvas>
            </div>
            <div class="panel">
                <h3>Cluster Distribution & Characteristics</h3>
                <div style="font-size: 13px; line-height: 1.6;">
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #e74c3c; background: #fff5f5;">
                        <b style="color:#e74c3c;">Cluster 0: Multi-Pollutant Severe Hotspot ({(df['cluster']==0).sum()} stations)</b><br>
                        Severe particulates combined with toxic vehicular gases ($PM_{{2.5}}: 135$, $PM_{{10}}: 146$, $NO_2: 57$). Predominant in industrial corridors and congested urban centres (Delhi NCR, Ahmedabad, Meerut).
                    </div>
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #2ecc71; background: #f0fff4;">
                        <b style="color:#2ecc71;">Cluster 1: Baseline Ambient / Cleaner Air ({(df['cluster']==1).sum()} stations)</b><br>
                        Stations meeting relatively clean background ambient levels across India ($PM_{{2.5}}: 53$, $PM_{{10}}: 71$, $NO_2: 19$). Coastal belts and southern regions.
                    </div>
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #f39c12; background: #fffaf0;">
                        <b style="color:#f39c12;">Cluster 2: Particulate & Dust Dominated ({(df['cluster']==2).sum()} stations)</b><br>
                        Elevated coarse dust and particulate matter, but moderate combustion gases ($PM_{{2.5}}: 115$, $PM_{{10}}: 124$). Typical of construction belts and semi-arid terrain.
                    </div>
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid #9b59b6; background: #faf5ff;">
                        <b style="color:#9b59b6;">Cluster 3: Industrial & Gas Dominated ({(df['cluster']==3).sum()} stations)</b><br>
                        Marked by elevated sulfur dioxide ($SO_2: 39 \ \mu g/m^3$, $3.5\times$ national median) indicating coal, chemical, or power generation plumes.
                    </div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h3>K-Means Model Evaluation Across Cluster Counts (K = 2 to 8)</h3>
            <canvas id="evalMetricsChart" height="100"></canvas>
        </div>
    </div>

    <!-- TAB 4: CORRELATION -->
    <div id="tab-correlation" class="tab-content">
        <div class="grid-2">
            <div class="panel">
                <h3>Spearman Rank Correlation Heatmap</h3>
                <table class="corr-table" id="corrTable"></table>
            </div>
            <div class="panel">
                <h3>Key Co-occurrence Findings</h3>
                <div style="font-size: 13px; line-height: 1.7;">
                    <p><b>1. Particulate Coupling (PM2.5 & PM10: r = +0.804):</b><br>
                    Extreme co-occurrence confirms that when fine combustion particles rise, coarse fugitive dust rises in tandem across Indian urban airsheds.</p>

                    <p><b>2. Vehicular Exhaust Coupling (NO2 & NH3: r = +0.446):</b><br>
                    Nitrogen dioxide and ammonia show moderate positive co-occurrence, pointing to urban traffic, diesel catalytic reduction byproducts, and localized combustion.</p>

                    <p><b>3. Roadside Corridor Signature (PM10 & NO2: r = +0.425):</b><br>
                    Combines vehicle tailpipe emissions with mechanical tyre and brake abrasion and resuspension of road dust.</p>

                    <p><b>4. Photochemical Ozone Decoupling (CO & Ozone: r = -0.106):</b><br>
                    Negative correlation reflects daytime photochemical smog chemistry: carbon monoxide and precursor VOCs are consumed during solar irradiation to form ground-level ozone.</p>
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 5: PCA -->
    <div id="tab-pca" class="tab-content">
        <div class="grid-2">
            <div class="panel">
                <h3>Principal Component Factor Loadings</h3>
                <canvas id="pcaLoadingsChart" height="230"></canvas>
            </div>
            <div class="panel">
                <h3>Variance Explained & Component Interpretations</h3>
                <div style="font-size: 13px; line-height: 1.6;">
                    <p><b>PC1 (33.5% Variance) &mdash; Overall Particulate & Combustion Severity:</b><br>
                    Heaviest positive weights on $PM_{{10}}$ (+0.567), $PM_{{2.5}}$ (+0.538), and $NO_2$ (+0.450). Captures overall airshed pollution load.</p>

                    <p><b>PC2 (16.6% Variance) &mdash; Industrial Sulfur Signature:</b><br>
                    Dominant positive loading on $SO_2$ (+0.687), negative on Ozone and $NH_3$. Isolates thermal power plants, smelting, and heavy fuel-oil combustion.</p>

                    <p><b>PC3 (14.8% Variance) &mdash; Secondary Ozone Photochemistry:</b><br>
                    Dominant positive loading on $OZONE$ (+0.619) and negative on $CO$ (-0.439). Differentiates daytime photochemical secondary smog from fresh primary emissions.</p>

                    <p><b>Total Cumulative Variance:</b> The top 3 components capture <b>65.0%</b> of all multi-pollutant variance across India.</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        function openTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            if (tabId === 'tab-map') {{
                setTimeout(() => map.invalidateSize(), 200);
            }}
        }}

        // Initialize DataTable
        $(document).ready(function() {{
            $('#priorityTable').DataTable({{
                pageLength: 15,
                order: [[0, 'asc']]
            }});
        }});

        // Map setup
        const map = L.map('map').setView([22.5937, 78.9629], 5);
        const streetMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 18
        }}).addTo(map);
        const satelliteMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 18
        }});
        L.control.layers({{ "Street Map": streetMap, "Satellite": satelliteMap }}).addTo(map);

        const stations = {stations_json};

        stations.forEach(s => {{
            if (!s.lat || !s.lon) return;
            const marker = L.circleMarker([s.lat, s.lon], {{
                radius: s.outlier ? 9 : 6,
                fillColor: s.color,
                color: s.outlier ? '#000' : '#fff',
                weight: s.outlier ? 2.5 : 1,
                opacity: 1,
                fillOpacity: 0.85
            }}).addTo(map);

            const outlierBadge = s.outlier ? '<span class="badge badge-outlier">STATISTICAL OUTLIER</span>' : '';
            marker.bindPopup(`
                <div style="font-family:sans-serif; min-width:210px; font-size:12px;">
                    <b>${{s.name}}</b><br>
                    <span style="color:#666;">${{s.city}}, ${{s.state}}</span><br>
                    <b>Rank:</b> #${{s.rank}} in India ${{outlierBadge}}<br>
                    <b>Profile:</b> ${{s.profile}}<br>
                    <hr style="margin:5px 0;">
                    PM2.5: <b>${{s.pm25}}</b> &bull; PM10: <b>${{s.pm10}}</b> &bull; NO2: <b>${{s.no2}}</b><br>
                    SO2: <b>${{s.so2}}</b> &bull; CO: <b>${{s.co}}</b> &bull; O3: <b>${{s.o3}}</b>
                    <div style="margin-top:6px; font-size:11px; color:#2980b9;"><b>Action:</b> ${{s.action}}</div>
                </div>
            `);
        }});

        // Render Correlation Table
        const corrData = {corr_matrix_json};
        const pols = {pollutants_json};
        let tableHtml = '<tr><th></th>' + pols.map(p => '<th>' + p + '</th>').join('') + '</tr>';
        corrData.forEach((row, i) => {{
            tableHtml += '<tr><th>' + pols[i] + '</th>';
            row.forEach(val => {{
                let bg = '#ffffff';
                if (val >= 0.7) bg = '#ff7675';
                else if (val >= 0.4) bg = '#fab1a0';
                else if (val >= 0.2) bg = '#ffeaa7';
                else if (val < 0) bg = '#74b9ff';
                tableHtml += `<td style="background:${{bg}}; font-weight:bold;">${{val.toFixed(2)}}</td>`;
            }});
            tableHtml += '</tr>';
        }});
        document.getElementById('corrTable').innerHTML = tableHtml;

        // Cluster Profiles Bar Chart
        const clusterProfiles = {cluster_profiles_json};
        new Chart(document.getElementById('clusterBarChart'), {{
            type: 'bar',
            data: {{
                labels: pols,
                datasets: [
                    {{ label: 'Cluster 0: Multi-Pollutant Severe', data: pols.map(p => clusterProfiles[0][p]), backgroundColor: '#e74c3c' }},
                    {{ label: 'Cluster 1: Baseline Ambient Clean', data: pols.map(p => clusterProfiles[1][p]), backgroundColor: '#2ecc71' }},
                    {{ label: 'Cluster 2: Particulate Dominated', data: pols.map(p => clusterProfiles[2][p]), backgroundColor: '#f39c12' }},
                    {{ label: 'Cluster 3: Industrial Gas Dominated', data: pols.map(p => clusterProfiles[3][p]), backgroundColor: '#9b59b6' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Median Concentration (ug/m3 or mg/m3)' }} }} }}
            }}
        }});

        // PCA Loadings Chart
        new Chart(document.getElementById('pcaLoadingsChart'), {{
            type: 'bar',
            data: {{
                labels: {pca_labels_json},
                datasets: [
                    {{ label: 'PC1 (Particulate Severity)', data: {pc1_json}, backgroundColor: '#3498db' }},
                    {{ label: 'PC2 (Industrial Sulfur)', data: {pc2_json}, backgroundColor: '#9b59b6' }},
                    {{ label: 'PC3 (Ozone Smog)', data: {pc3_json}, backgroundColor: '#e67e22' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ min: -0.6, max: 0.8, title: {{ display: true, text: 'Factor Loading' }} }} }}
            }}
        }});

        // Clustering Metrics Evaluation Chart
        new Chart(document.getElementById('evalMetricsChart'), {{
            type: 'line',
            data: {{
                labels: {k_labels_json},
                datasets: [
                    {{ label: 'Silhouette Score (Higher is better)', data: {sil_json}, borderColor: '#27ae60', tension: 0.2, yAxisID: 'y' }},
                    {{ label: 'Davies-Bouldin Index (Lower is better)', data: {db_json}, borderColor: '#e74c3c', tension: 0.2, yAxisID: 'y1' }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ type: 'linear', position: 'left', title: {{ display: true, text: 'Silhouette' }} }},
                    y1: {{ type: 'linear', position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'Davies-Bouldin' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    output_path = PROJECT_ROOT / "outputs" / "dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("=" * 65)
    print(" EXECUTIVE AIR POLLUTION ANALYTICS DASHBOARD GENERATED")
    print(f" Saved to: {output_path}")
    print("=" * 65)
    return output_path

if __name__ == "__main__":
    generate_full_dashboard()
