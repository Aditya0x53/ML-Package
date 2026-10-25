import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

def generate_interactive_hotspot_map():
    clustered_file = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    if not clustered_file.exists():
        raise FileNotFoundError(f"{clustered_file} not found. Run src/model_clustering.py first.")

    df = pd.read_csv(clustered_file)
    output_html_path = PROJECT_ROOT / "outputs" / "hotspot_map.html"
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Color mapping for the 4 clusters
    cluster_colors = {
        0: "#d9534f",  # Red: Multi-pollutant severe hotspot
        1: "#5cb85c",  # Green: Baseline / Clean ambient
        2: "#f0ad4e",  # Orange: Particulate dominated
        3: "#9b59b6"   # Purple: Industrial / Gas dominated
    }

    # Prepare GeoJSON / Marker data
    stations_data = []
    for _, row in df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
        stations_data.append({
            "name": row['station'],
            "city": row['city'],
            "state": row['state'],
            "lat": float(row['latitude']),
            "lon": float(row['longitude']),
            "cluster": int(row['cluster']),
            "cluster_profile": row['cluster_profile'],
            "color": cluster_colors.get(int(row['cluster']), "#337ab7"),
            "priority_rank": int(row['priority_rank']),
            "is_outlier": bool(row['is_outlier']),
            "pm25": float(row['PM2.5']),
            "pm10": float(row['PM10']),
            "no2": float(row['NO2']),
            "so2": float(row['SO2']),
            "co": float(row['CO']),
            "o3": float(row['OZONE']),
            "nh3": float(row['NH3'])
        })

    stations_json = json.dumps(stations_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPCB Air Pollution Hotspot Map - Indian Cities</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        .legend {{
            position: absolute; bottom: 25px; right: 25px; z-index: 1000;
            background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.25);
            max-width: 320px; font-size: 13px; line-height: 1.5;
        }}
        .legend h4 {{ margin: 0 0 10px; font-size: 14px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block; margin-right: 8px; flex-shrink: 0; }}
        .info-box {{
            position: absolute; top: 15px; left: 60px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 12px 18px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 13px;
        }}
        .info-box h3 {{ margin: 0 0 4px; font-size: 16px; color: #2c3e50; }}
        .badge-outlier {{ background: #ff4757; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="info-box">
        <h3>Air Pollution Hotspot Profiling (CPCB Stations)</h3>
        <div>Monitoring <b>{len(stations_data)}</b> stations across India &bull; Colored by ML Pollution Profile</div>
    </div>

    <div class="legend">
        <h4>Pollution Profiles</h4>
        <div class="legend-item"><span class="legend-color" style="background:#d9534f;"></span><b>Cluster 0:</b> Severe Multi-Pollutant</div>
        <div class="legend-item"><span class="legend-color" style="background:#5cb85c;"></span><b>Cluster 1:</b> Baseline / Clean Ambient</div>
        <div class="legend-item"><span class="legend-color" style="background:#f0ad4e;"></span><b>Cluster 2:</b> Particulate Dominated</div>
        <div class="legend-item"><span class="legend-color" style="background:#9b59b6;"></span><b>Cluster 3:</b> Industrial / Gas Dominated</div>
        <div style="margin-top: 8px; font-size: 11px; color: #666; border-top: 1px solid #eee; pt: 4px;">
            Large pulsing / thick border markers indicate <b>Statistical Outliers</b>.
        </div>
    </div>

    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([22.5937, 78.9629], 5);
        
        // Esri World Street Map (100% free, high-speed, never blocks local file:// URLs)
        const streetMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, USGS, TomTom',
            maxZoom: 18
        }}).addTo(map);

        const satelliteMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Earthstar Geographics',
            maxZoom: 18
        }});

        L.control.layers({{ "Street Map": streetMap, "Satellite": satelliteMap }}).addTo(map);

        const stations = {stations_json};

        stations.forEach(s => {{
            const radius = s.is_outlier ? 9 : 6;
            const marker = L.circleMarker([s.lat, s.lon], {{
                radius: radius,
                fillColor: s.color,
                color: s.is_outlier ? '#000' : '#fff',
                weight: s.is_outlier ? 2.5 : 1,
                opacity: 1,
                fillOpacity: 0.85
            }}).addTo(map);

            const outlierBadge = s.is_outlier ? '<span class="badge-outlier">EXTREME OUTLIER</span>' : '';

            const popupContent = `
                <div style="font-family: sans-serif; font-size: 12px; min-width: 210px;">
                    <div style="font-weight:bold; font-size: 13px; margin-bottom: 3px;">${{s.name}}</div>
                    <div style="color: #666; margin-bottom: 6px;">${{s.city}}, ${{s.state}}</div>
                    <div style="margin-bottom: 6px;"><b>Profile:</b> ${{s.cluster_profile}} ${{outlierBadge}}</div>
                    <div style="margin-bottom: 6px;"><b>Priority Rank:</b> #${{s.priority_rank}} in India</div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 4px;">
                        <tr style="background: #f8f9fa;"><td><b>PM2.5:</b> ${{s.pm25}} &mu;g/m&sup3;</td><td><b>PM10:</b> ${{s.pm10}} &mu;g/m&sup3;</td></tr>
                        <tr><td><b>NO2:</b> ${{s.no2}} &mu;g/m&sup3;</td><td><b>SO2:</b> ${{s.so2}} &mu;g/m&sup3;</td></tr>
                        <tr style="background: #f8f9fa;"><td><b>CO:</b> ${{s.co}} mg/m&sup3;</td><td><b>O3:</b> ${{s.o3}} &mu;g/m&sup3;</td></tr>
                        <tr><td colspan="2"><b>NH3:</b> ${{s.nh3}} &mu;g/m&sup3;</td></tr>
                    </table>
                </div>
            `;
            marker.bindPopup(popupContent);
        }});
    </script>
</body>
</html>
"""
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("=" * 65)
    print(" INTERACTIVE HOTSPOT MAP GENERATED")
    print(f" Saved to: {output_html_path}")
    print(f" Total Stations Plotted: {len(stations_data)}")
    print(" You can open this file in any web browser to view the interactive map!")
    print("=" * 65)

    return output_html_path

if __name__ == "__main__":
    generate_interactive_hotspot_map()
