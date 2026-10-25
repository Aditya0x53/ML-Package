import glob
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']

# Plausible physical limits for ambient air quality (to discard sensor glitches)
POLLUTANT_BOUNDS = {
    'PM2.5': (0.0, 1000.0),   # ug/m3
    'PM10':  (0.0, 1500.0),   # ug/m3
    'NO2':   (0.0, 1000.0),   # ug/m3
    'SO2':   (0.0, 1000.0),   # ug/m3
    'CO':    (0.0, 100.0),    # mg/m3
    'OZONE': (0.0, 800.0),    # ug/m3
    'NH3':   (0.0, 1000.0)    # ug/m3
}

def load_all_raw_snapshots():
    """Finds and concatenates all hourly CSV snapshots from data/raw/."""
    files = list(RAW_DATA_DIR.glob("cpcb_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CPCB snapshot files found in {RAW_DATA_DIR}")
    
    print(f"[1/5] Loading {len(files)} snapshot file(s) from {RAW_DATA_DIR}...")
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"      Total combined raw records: {len(combined):,}")
    return combined

def clean_and_filter_records(df):
    """Removes duplicates and applies physical plausibility bounds."""
    print("[2/5] Deduplicating and validating physical plausibility bounds...")
    initial_len = len(df)
    
    # Deduplicate exact identical readings (station + pollutant + timestamp)
    dedup_cols = ['station', 'pollutant_id', 'last_update']
    df = df.drop_duplicates(subset=[c for c in dedup_cols if c in df.columns])
    print(f"      Removed {initial_len - len(df):,} duplicate entries.")

    # Convert avg_value to numeric
    df['avg_value'] = pd.to_numeric(df['avg_value'], errors='coerce')

    # Filter to only the 7 standard pollutants
    df = df[df['pollutant_id'].isin(TARGET_POLLUTANTS)].copy()

    # Apply physical bounds: replace impossible readings with NaN
    invalid_mask = pd.Series(False, index=df.index)
    for pollutant, (low, high) in POLLUTANT_BOUNDS.items():
        pol_mask = (df['pollutant_id'] == pollutant)
        out_of_bounds = pol_mask & ((df['avg_value'] < low) | (df['avg_value'] > high))
        df.loc[out_of_bounds, 'avg_value'] = np.nan
        invalid_mask |= out_of_bounds

    print(f"      Flagged and nulled {invalid_mask.sum():,} out-of-bounds readings.")
    return df

def pivot_and_aggregate(df, min_pollutants=4):
    """
    Pivots long-format to wide-format (1 row per station).
    Aggregates snapshot values using the median (as stated in the project abstract).
    Filters out stations with fewer than min_pollutants recorded.
    """
    print(f"[3/5] Pivoting to 1 row per station & computing median across snapshots...")
    
    index_cols = ['station', 'city', 'state', 'latitude', 'longitude']
    # Ensure latitude and longitude are numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Pivot table with median aggregation
    pivoted = df.pivot_table(
        index=index_cols,
        columns='pollutant_id',
        values='avg_value',
        aggfunc='median'
    ).reset_index()

    # Reindex columns to ensure all target pollutants are present
    for pol in TARGET_POLLUTANTS:
        if pol not in pivoted.columns:
            pivoted[pol] = np.nan

    initial_stations = len(pivoted)
    
    # Count non-null pollutants per station
    valid_pollutant_count = pivoted[TARGET_POLLUTANTS].notna().sum(axis=1)
    
    # Filter stations with too few pollutants
    filtered = pivoted[valid_pollutant_count >= min_pollutants].copy()
    dropped = initial_stations - len(filtered)
    print(f"      Total unique stations: {initial_stations}")
    print(f"      Dropped {dropped} stations with fewer than {min_pollutants} pollutants.")
    print(f"      Retained {len(filtered)} stations for modeling.")
    return filtered

def impute_and_scale(df):
    """
    Imputes missing pollutant values using KNN Imputer,
    and scales features using StandardScaler.
    """
    print("[4/5] Imputing remaining missing values with KNN Imputer (k=5)...")
    feature_data = df[TARGET_POLLUTANTS].copy()

    # KNN Imputer preserves multi-pollutant correlations better than simple mean/median
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputed_features = imputer.fit_transform(feature_data)
    
    # Create clean unscaled dataframe
    cleaned_df = df.copy()
    cleaned_df[TARGET_POLLUTANTS] = np.round(imputed_features, 2)

    # Standard Scaling (mean=0, variance=1) for PCA & K-Means
    print("[5/5] Standardizing features (Z-score scaling)...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cleaned_df[TARGET_POLLUTANTS])
    
    scaled_df = cleaned_df.copy()
    scaled_df[TARGET_POLLUTANTS] = np.round(scaled_features, 4)

    return cleaned_df, scaled_df

def run_preprocessing_pipeline():
    """Runs the complete end-to-end preprocessing pipeline."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_df = load_all_raw_snapshots()
    cleaned_records = clean_and_filter_records(raw_df)
    station_matrix = pivot_and_aggregate(cleaned_records, min_pollutants=4)
    cleaned_df, scaled_df = impute_and_scale(station_matrix)

    # Save outputs
    cleaned_path = PROCESSED_DATA_DIR / "station_pollutants_cleaned.csv"
    scaled_path = PROCESSED_DATA_DIR / "station_pollutants_scaled.csv"
    summary_path = PROCESSED_DATA_DIR / "preprocessing_summary.json"

    cleaned_df.to_csv(cleaned_path, index=False)
    scaled_df.to_csv(scaled_path, index=False)

    summary = {
        "total_stations_processed": int(len(cleaned_df)),
        "pollutants": TARGET_POLLUTANTS,
        "pollutant_averages": cleaned_df[TARGET_POLLUTANTS].mean().round(2).to_dict(),
        "pollutant_medians": cleaned_df[TARGET_POLLUTANTS].median().round(2).to_dict(),
        "states_covered": int(cleaned_df['state'].nunique()),
        "cities_covered": int(cleaned_df['city'].nunique()),
        "cleaned_file": str(cleaned_path),
        "scaled_file": str(scaled_path)
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print(" PREPROCESSING PIPELINE COMPLETED")
    print(f" Stations Retained: {len(cleaned_df)} across {cleaned_df['city'].nunique()} cities and {cleaned_df['state'].nunique()} states")
    print(f" Clean Matrix:  {cleaned_path}")
    print(f" Scaled Matrix: {scaled_path}")
    print("=" * 65)
    return cleaned_df, scaled_df

if __name__ == "__main__":
    run_preprocessing_pipeline()
