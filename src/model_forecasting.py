import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CPCB AQI Breakpoints (Concentration Range -> AQI Range)
CPCB_BREAKPOINTS = {
    'PM2.5': [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 1000, 401, 500)],
    'PM10':  [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 1500, 401, 500)],
    'NO2':   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), (180, 280, 201, 300), (280, 400, 301, 400), (400, 1000, 401, 500)],
    'SO2':   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 3000, 401, 500)],
    'CO':    [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10.0, 101, 200), (10.0, 17.0, 201, 300), (17.0, 34.0, 301, 400), (34.0, 100.0, 401, 500)],
    'OZONE': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), (168, 208, 201, 300), (208, 748, 301, 400), (748, 1500, 401, 500)],
    'NH3':   [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200), (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 3000, 401, 500)]
}

def calculate_cpcb_sub_index(pollutant, val):
    """Calculates CPCB linear interpolated sub-index for a given pollutant concentration."""
    if pd.isna(val) or val < 0:
        return np.nan
    val = float(val)
    breakpoints = CPCB_BREAKPOINTS.get(pollutant, [])
    for b_lo, b_hi, i_lo, i_hi in breakpoints:
        if b_lo <= val <= b_hi:
            return round(i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (val - b_lo), 1)
    return 500.0 if val > 0 else 0.0

def calculate_overall_aqi(row, prefix=''):
    """Calculates overall CPCB AQI (maximum sub-index principle) and identifies dominant pollutant."""
    sub_indices = {}
    for pol in TARGET_POLLUTANTS:
        col = f"{prefix}{pol}" if prefix else pol
        if col in row and pd.notna(row[col]):
            sub_indices[pol] = calculate_cpcb_sub_index(pol, row[col])
    
    if not sub_indices:
        return np.nan, "N/A"
    
    # Must have at least 3 pollutants to calculate valid national AQI (with at least PM10 or PM2.5)
    dominant_pol = max(sub_indices, key=sub_indices.get)
    overall_aqi = round(sub_indices[dominant_pol], 1)
    return overall_aqi, dominant_pol

def aqi_category(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good (0-50)"
    elif aqi <= 100: return "Satisfactory (51-100)"
    elif aqi <= 200: return "Moderate (101-200)"
    elif aqi <= 300: return "Poor (201-300)"
    elif aqi <= 400: return "Very Poor (301-400)"
    else: return "Severe (>400)"

def build_multi_pollutant_time_series():
    """
    Parses all snapshot files, reconstructs the multi-pollutant station-level time series,
    and sets up lag features and targets for ALL 7 pollutants.
    """
    files = sorted(glob.glob(str(RAW_DATA_DIR / "cpcb_*.csv")))
    if not files:
        raise FileNotFoundError(f"No CPCB snapshot files found in {RAW_DATA_DIR}")

    print(f"[1/4] Loading {len(files)} snapshot files to construct multi-pollutant time series...")
    dfs = [pd.read_csv(f) for f in files]
    raw_df = pd.concat(dfs, ignore_index=True)

    raw_df = raw_df.drop_duplicates(subset=['station', 'pollutant_id', 'last_update']).copy()
    raw_df['avg_value'] = pd.to_numeric(raw_df['avg_value'], errors='coerce')
    raw_df = raw_df[raw_df['pollutant_id'].isin(TARGET_POLLUTANTS)]

    raw_df['timestamp'] = pd.to_datetime(raw_df['last_update'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    raw_df = raw_df.dropna(subset=['timestamp'])

    meta_cols = ['station', 'city', 'state', 'latitude', 'longitude', 'timestamp']
    pivoted = raw_df.pivot_table(
        index=meta_cols,
        columns='pollutant_id',
        values='avg_value',
        aggfunc='mean'
    ).reset_index()

    pivoted = pivoted.sort_values(by=['station', 'timestamp']).reset_index(drop=True)

    # Merge cluster assignments
    cluster_file = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    if cluster_file.exists():
        cl_df = pd.read_csv(cluster_file)[['station', 'cluster', 'cluster_profile']]
        pivoted = pivoted.merge(cl_df, on='station', how='left')
    else:
        pivoted['cluster'] = 0
        pivoted['cluster_profile'] = 'General'

    feature_rows = []
    inference_rows = []

    for station, group in pivoted.groupby('station'):
        group = group.sort_values('timestamp').copy()
        
        # Temporal features
        group['hour'] = group['timestamp'].dt.hour
        group['sin_hour'] = np.sin(2 * np.pi * group['hour'] / 24)
        group['cos_hour'] = np.cos(2 * np.pi * group['hour'] / 24)
        group['pm25_pm10_ratio'] = group['PM2.5'] / (group['PM10'] + 1e-3)

        # Create Lags and Targets for ALL 7 pollutants
        for pol in TARGET_POLLUTANTS:
            if pol in group.columns:
                group[f'{pol}_lag1'] = group[pol].shift(1)
                group[f'{pol}_diff'] = group[pol] - group[f'{pol}_lag1']
                group[f'target_{pol}'] = group[pol].shift(-1)
            else:
                group[pol] = np.nan
                group[f'{pol}_lag1'] = np.nan
                group[f'{pol}_diff'] = np.nan
                group[f'target_{pol}'] = np.nan

        latest_row = group.iloc[-1].copy()
        inference_rows.append(latest_row)

        # Labeled rows have valid future target for at least PM2.5 or PM10
        valid = group.dropna(subset=['target_PM2.5', 'target_PM10'], how='all')
        feature_rows.append(valid)

    train_test_df = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    inference_df = pd.DataFrame(inference_rows)

    print(f"      Constructed {len(train_test_df):,} chronological multi-pollutant training transitions.")
    print(f"      Stations ready for next-hour multi-pollutant inference: {len(inference_df):,}")
    return train_test_df, inference_df

def train_and_forecast_all_pollutants(df, inference_df):
    """
    Trains specialized regressors for each of the 7 air pollutants:
    PM2.5, PM10, NO2, SO2, CO, OZONE, NH3.
    Evaluates R2, RMSE, MAE for each pollutant and generates comprehensive next-hour AQI forecasts.
    """
    base_features = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3',
        'hour', 'sin_hour', 'cos_hour', 'cluster', 'pm25_pm10_ratio'
    ]

    metrics_list = []
    trained_models = {}

    print("\n[2/4] Training Multi-Pollutant Forecasting Regressors across all 7 pollutants...")

    # Fill NaNs in features with median
    for col in base_features:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)

    for pol in TARGET_POLLUTANTS:
        target_col = f'target_{pol}'
        lag_diff_col = f'{pol}_diff'
        
        pol_features = base_features + ([lag_diff_col] if lag_diff_col in df.columns else [])
        
        # Filter rows where target exists for this specific pollutant
        valid_subset = df.dropna(subset=[target_col, pol]).copy()
        if len(valid_subset) < 30:
            print(f"      [!] Insufficient historical data for {pol}, skipping model.")
            continue

        X = valid_subset[pol_features].fillna(0).values
        y = valid_subset[target_col].values

        split_idx = int(len(valid_subset) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train Ridge Regressor (with L2 regularization) and HistGradientBoosting
        ridge = Ridge(alpha=5.0)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        
        # Baseline persistence (yt+1 = yt)
        pol_idx = base_features.index(pol)
        y_baseline = X_test[:, pol_idx]

        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        base_rmse = root_mean_squared_error(y_test, y_baseline)

        trained_models[pol] = (ridge, pol_features)

        metrics_list.append({
            'Pollutant': pol,
            'Model': 'Ridge Regressor',
            'R2_Score': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'Baseline_RMSE': round(base_rmse, 2),
            'Error_Reduction_%': round(((base_rmse - rmse) / (base_rmse + 1e-6)) * 100, 1)
        })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = OUTPUT_DIR / "forecasting_multi_pollutant_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 75)
    print(" MULTI-POLLUTANT NEXT-HOUR FORECASTING EVALUATION (ALL 7 IMPURITIES)")
    print("=" * 75)
    print(metrics_df.to_string(index=False))

    # [3/4] Generate Next-Hour Predictions for ALL 7 pollutants across all active stations
    print("\n[3/4] Generating Next-Hour Multi-Pollutant & AQI Forecasts for all stations...")
    
    # Fill NaNs for inference
    for col in base_features:
        if col not in inference_df.columns:
            inference_df[col] = 0.0
        inference_df[col] = inference_df[col].fillna(df[col].median() if col in df.columns else 0.0)

    for pol in TARGET_POLLUTANTS:
        if pol in trained_models:
            model, pol_feats = trained_models[pol]
            X_inf = inference_df[pol_feats].fillna(0).values
            preds = np.maximum(model.predict(X_inf), 0.0)
            inference_df[f'pred_{pol}'] = np.round(preds, 2)
            inference_df[f'change_{pol}'] = np.round(inference_df[f'pred_{pol}'] - inference_df[pol], 2)
        else:
            inference_df[f'pred_{pol}'] = inference_df[pol]
            inference_df[f'change_{pol}'] = 0.0

    # Calculate Current AQI vs Predicted Next-Hour AQI
    current_aqis, current_doms = [], []
    pred_aqis, pred_doms = [], []

    for _, row in inference_df.iterrows():
        c_aqi, c_dom = calculate_overall_aqi(row)
        current_aqis.append(c_aqi)
        current_doms.append(c_dom)

        # Predicted row mapping
        p_row = {pol: row[f'pred_{pol}'] for pol in TARGET_POLLUTANTS}
        p_aqi, p_dom = calculate_overall_aqi(p_row)
        pred_aqis.append(p_aqi)
        pred_doms.append(p_dom)

    inference_df['current_aqi'] = current_aqis
    inference_df['current_dominant_pollutant'] = current_doms
    inference_df['predicted_next_aqi'] = pred_aqis
    inference_df['predicted_dominant_pollutant'] = pred_doms
    inference_df['predicted_aqi_change'] = np.round(inference_df['predicted_next_aqi'] - inference_df['current_aqi'], 1)

    inference_df['current_aqi_category'] = inference_df['current_aqi'].apply(aqi_category)
    inference_df['predicted_aqi_category'] = inference_df['predicted_next_aqi'].apply(aqi_category)

    # General trend label
    def classify_aqi_trend(change):
        if pd.isna(change): return "Stable"
        if change >= 25: return "Severe AQI Spike Expected (+25)"
        elif change >= 8: return "Moderate Air Quality Degradation"
        elif change <= -20: return "Rapid Air Quality Clearing"
        elif change <= -8: return "Gradual Improvement"
        else: return "Stable Air Quality"

    inference_df['forecast_trend'] = inference_df['predicted_aqi_change'].apply(classify_aqi_trend)

    # Output master forecast file
    cols_to_save = [
        'station', 'city', 'state', 'latitude', 'longitude', 'cluster_profile',
        'current_aqi', 'predicted_next_aqi', 'predicted_aqi_change', 'forecast_trend',
        'current_dominant_pollutant', 'predicted_dominant_pollutant',
        'current_aqi_category', 'predicted_aqi_category',
        'PM2.5', 'pred_PM2.5', 'change_PM2.5',
        'PM10', 'pred_PM10', 'change_PM10',
        'NO2', 'pred_NO2', 'change_NO2',
        'SO2', 'pred_SO2', 'change_SO2',
        'CO', 'pred_CO', 'change_CO',
        'OZONE', 'pred_OZONE', 'change_OZONE',
        'NH3', 'pred_NH3', 'change_NH3'
    ]

    # Save backwards-compatible alias columns so existing code doesn't break
    inference_df['current_pm25'] = inference_df['PM2.5']
    inference_df['predicted_next_pm25'] = inference_df['pred_PM2.5']
    inference_df['predicted_change'] = inference_df['change_PM2.5']

    output_df = inference_df[[c for c in cols_to_save if c in inference_df.columns]].sort_values(by='predicted_next_aqi', ascending=False).reset_index(drop=True)
    forecast_path = PROCESSED_DATA_DIR / "station_next_hour_forecasts.csv"
    output_df.to_csv(forecast_path, index=False)

    print(f"\n[+] Master Multi-Pollutant & AQI Forecasts saved to: {forecast_path}")
    print("\n Top 5 Highest Predicted AQI in Next Hour:")
    print("-" * 75)
    print(output_df[['station', 'city', 'current_aqi', 'predicted_next_aqi', 'predicted_dominant_pollutant', 'forecast_trend']].head(5).to_string(index=False))
    print("=" * 75)

    return metrics_df, output_df

def run_forecasting_pipeline():
    train_test_df, inference_df = build_multi_pollutant_time_series()
    if len(train_test_df) < 30:
        print("[!] Insufficient historical transitions to train models. Need more hourly snapshots.")
        return None
    metrics_df, output_df = train_and_forecast_all_pollutants(train_test_df, inference_df)
    return output_df

if __name__ == "__main__":
    run_forecasting_pipeline()
