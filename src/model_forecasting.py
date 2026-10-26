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

def build_time_series_dataset():
    """
    Parses all raw snapshot CSV files, reconstructs the station-level time series,
    and constructs autoregressive lag features, pollutant interaction ratios, and temporal encodings.
    """
    files = sorted(glob.glob(str(RAW_DATA_DIR / "cpcb_*.csv")))
    if not files:
        raise FileNotFoundError(f"No CPCB snapshot files found in {RAW_DATA_DIR}")

    print(f"[1/4] Loading {len(files)} snapshot files to construct time series...")
    dfs = [pd.read_csv(f) for f in files]
    raw_df = pd.concat(dfs, ignore_index=True)

    # Clean duplicates & parse timestamps
    raw_df = raw_df.drop_duplicates(subset=['station', 'pollutant_id', 'last_update']).copy()
    raw_df['avg_value'] = pd.to_numeric(raw_df['avg_value'], errors='coerce')
    raw_df = raw_df[raw_df['pollutant_id'].isin(TARGET_POLLUTANTS)]

    # Parse datetime (format: DD-MM-YYYY HH:MM:SS)
    raw_df['timestamp'] = pd.to_datetime(raw_df['last_update'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    raw_df = raw_df.dropna(subset=['timestamp'])

    # Pivot: 1 row per (station, timestamp)
    meta_cols = ['station', 'city', 'state', 'latitude', 'longitude', 'timestamp']
    pivoted = raw_df.pivot_table(
        index=meta_cols,
        columns='pollutant_id',
        values='avg_value',
        aggfunc='mean'
    ).reset_index()

    pivoted = pivoted.sort_values(by=['station', 'timestamp']).reset_index(drop=True)

    # Load cluster labels if available
    cluster_file = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    if cluster_file.exists():
        cl_df = pd.read_csv(cluster_file)[['station', 'cluster', 'cluster_profile']]
        pivoted = pivoted.merge(cl_df, on='station', how='left')
    else:
        pivoted['cluster'] = 0
        pivoted['cluster_profile'] = 'General'

    print(f"      Reconstructed {len(pivoted):,} station-hour observations across {pivoted['station'].nunique()} stations.")

    # Engineer Features per station
    feature_rows = []
    inference_rows = []

    for station, group in pivoted.groupby('station'):
        group = group.sort_values('timestamp').copy()
        
        # Lag features
        group['pm25_lag1'] = group['PM2.5'].shift(1)
        group['pm25_diff'] = group['PM2.5'] - group['pm25_lag1']
        group['pm10_lag1'] = group['PM10'].shift(1)
        
        # Target: Next hour PM2.5 (shift -1)
        group['target_next_pm25'] = group['PM2.5'].shift(-1)
        
        # Temporal features
        group['hour'] = group['timestamp'].dt.hour
        group['sin_hour'] = np.sin(2 * np.pi * group['hour'] / 24)
        group['cos_hour'] = np.cos(2 * np.pi * group['hour'] / 24)
        
        # Particulate ratio
        group['pm25_pm10_ratio'] = group['PM2.5'] / (group['PM10'] + 1e-3)

        # The last row for each station has target NaN -> this is our live Inference row!
        latest_row = group.iloc[-1].copy()
        if pd.notna(latest_row['PM2.5']):
            inference_rows.append(latest_row)

        # Rows with valid target become our Training/Testing set
        valid_rows = group.dropna(subset=['PM2.5', 'target_next_pm25'])
        feature_rows.append(valid_rows)

    train_test_df = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    inference_df = pd.DataFrame(inference_rows)

    print(f"      Labeled observation transitions for modeling: {len(train_test_df):,}")
    print(f"      Stations ready for next-hour live inference: {len(inference_df):,}")
    return train_test_df, inference_df

def train_and_evaluate_forecasting_models(df, inference_df):
    """
    Trains and compares multiple regression models for Next-Hour PM2.5 forecasting:
    - Persistence Baseline (t+1 = t)
    - Ridge Regression
    - Random Forest Regressor
    - HistGradientBoostingRegressor
    """
    feature_cols = [
        'PM2.5', 'pm25_lag1', 'pm25_diff', 'PM10', 'pm25_pm10_ratio',
        'NO2', 'SO2', 'CO', 'OZONE', 'NH3',
        'hour', 'sin_hour', 'cos_hour', 'cluster'
    ]

    # Ensure all feature columns exist and impute NaNs with median
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    X = df[feature_cols].values
    y = df['target_next_pm25'].values

    # Temporal split: 80% train, 20% test (preserving chronological order)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n[2/4] Training models on {len(X_train):,} transitions, testing on {len(X_test):,} transitions...")

    # Define candidate models
    models = {
        "Persistence Baseline (y_t+1 = y_t)": None,
        "Ridge Regressor (L2 Linear)": Ridge(alpha=10.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=150, max_depth=8, random_state=42)
    }

    results = []
    trained_estimators = {}

    for name, model in models.items():
        if model is None:
            # Baseline: predict current PM2.5 (index 0 of feature_cols)
            y_pred = X_test[:, 0]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_estimators[name] = model

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Mean Absolute Percentage Error (clipping y to avoid div by zero)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1.0))) * 100

        results.append({
            "Model": name,
            "RMSE (ug/m3)": round(rmse, 2),
            "MAE (ug/m3)": round(mae, 2),
            "R2_Score": round(r2, 4),
            "MAPE (%)": round(mape, 2)
        })

    eval_df = pd.DataFrame(results).sort_values(by="R2_Score", ascending=False)
    metrics_path = OUTPUT_DIR / "forecasting_model_metrics.csv"
    eval_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 68)
    print(" TIME-SERIES NEXT-HOUR PM2.5 FORECASTING EVALUATION")
    print("=" * 68)
    print(eval_df.to_string(index=False))

    # Feature Importance from Random Forest
    rf = trained_estimators["Random Forest Regressor"]
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": np.round(rf.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False)

    importance_path = OUTPUT_DIR / "forecasting_feature_importance.csv"
    importances.to_csv(importance_path, index=False)
    print(f"\n[+] Feature Importances saved to: {importance_path}")
    print(importances.head(6).to_string(index=False))

    # [3/4] Run Live Inference for all stations
    print("\n[3/4] Generating Next-Hour Forecasts for all active stations...")
    best_model = trained_estimators["HistGradientBoosting Regressor"]

    for col in feature_cols:
        if col not in inference_df.columns:
            inference_df[col] = np.nan
        inference_df[col] = inference_df[col].fillna(df[col].median() if col in df.columns else 0)

    X_inf = inference_df[feature_cols].values
    predicted_pm25 = best_model.predict(X_inf)

    inference_df['current_pm25'] = np.round(inference_df['PM2.5'], 2)
    inference_df['predicted_next_pm25'] = np.round(np.maximum(predicted_pm25, 0), 2)
    inference_df['predicted_change'] = np.round(inference_df['predicted_next_pm25'] - inference_df['current_pm25'], 2)

    # Categorize Risk Trend
    def classify_trend(change):
        if change >= 20:
            return "Severe Spike Expected (+20 ug/m3)"
        elif change >= 5:
            return "Moderate Increase"
        elif change <= -15:
            return "Rapid Air Quality Clearing"
        elif change <= -5:
            return "Gradual Improvement"
        else:
            return "Stable Ambient Level"

    inference_df['forecast_trend'] = inference_df['predicted_change'].apply(classify_trend)

    # Next-hour Air Quality Category according to CPCB National AQI scale for PM2.5
    def cpcb_category(val):
        if val <= 30: return "Good (0-30)"
        elif val <= 60: return "Satisfactory (31-60)"
        elif val <= 90: return "Moderate (61-90)"
        elif val <= 120: return "Poor (91-120)"
        elif val <= 250: return "Very Poor (121-250)"
        else: return "Severe (>250)"

    inference_df['predicted_aqi_category'] = inference_df['predicted_next_pm25'].apply(cpcb_category)

    # Save output
    output_cols = [
        'station', 'city', 'state', 'latitude', 'longitude', 'cluster_profile',
        'current_pm25', 'predicted_next_pm25', 'predicted_change',
        'forecast_trend', 'predicted_aqi_category'
    ]
    forecast_results = inference_df[output_cols].sort_values(by='predicted_next_pm25', ascending=False).reset_index(drop=True)
    forecast_path = PROCESSED_DATA_DIR / "station_next_hour_forecasts.csv"
    forecast_results.to_csv(forecast_path, index=False)

    print(f"\n[+] Master Next-Hour Forecasts saved to: {forecast_path}")
    print("\n Top 5 Highest Predicted PM2.5 Levels in Next Hour:")
    print("-" * 68)
    print(forecast_results[['station', 'city', 'current_pm25', 'predicted_next_pm25', 'predicted_change', 'forecast_trend']].head(5).to_string(index=False))
    print("=" * 68)

    return eval_df, importances, forecast_results

def run_forecasting_pipeline():
    train_test_df, inference_df = build_time_series_dataset()
    if len(train_test_df) < 50:
        print("[!] Insufficient historical transitions to train models. Need more hourly snapshots.")
        return None
    eval_df, importances, forecast_results = train_and_evaluate_forecasting_models(train_test_df, inference_df)
    return forecast_results

if __name__ == "__main__":
    run_forecasting_pipeline()
