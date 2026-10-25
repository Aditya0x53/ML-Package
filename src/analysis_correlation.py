import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_correlation_analysis():
    cleaned_file = PROCESSED_DATA_DIR / "station_pollutants_cleaned.csv"
    if not cleaned_file.exists():
        raise FileNotFoundError(f"{cleaned_file} not found. Run src/preprocess.py first.")

    df = pd.read_csv(cleaned_file)
    print("=" * 65)
    print(" SPEARMAN RANK CORRELATION ANALYSIS")
    print(f" Loaded {len(df)} stations across India.")
    print("=" * 65)

    # Compute Spearman correlation matrix
    corr_matrix = df[TARGET_POLLUTANTS].corr(method='spearman').round(3)

    # Save CSV
    corr_csv_path = OUTPUT_DIR / "spearman_correlation_matrix.csv"
    corr_matrix.to_csv(corr_csv_path)
    print(f"\n[+] Spearman Correlation Matrix saved to: {corr_csv_path}\n")
    print(corr_matrix.to_string())

    # Find highest positive and negative co-occurrences (excluding diagonal)
    pairs = []
    for i in range(len(TARGET_POLLUTANTS)):
        for j in range(i + 1, len(TARGET_POLLUTANTS)):
            p1 = TARGET_POLLUTANTS[i]
            p2 = TARGET_POLLUTANTS[j]
            r = corr_matrix.loc[p1, p2]
            pairs.append((p1, p2, r))

    # Sort by correlation coefficient descending
    pairs_df = pd.DataFrame(pairs, columns=['Pollutant_1', 'Pollutant_2', 'Spearman_r'])
    pairs_df = pairs_df.sort_values(by='Spearman_r', ascending=False).reset_index(drop=True)

    pairs_path = OUTPUT_DIR / "pollutant_co_occurrence_ranked.csv"
    pairs_df.to_csv(pairs_path, index=False)

    print("\n Top Co-occurring Pollutant Pairs (Highest Spearman Correlation):")
    print("-" * 55)
    for idx, row in pairs_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Pollutant_1']:<6} &  {row['Pollutant_2']:<6} : r = {row['Spearman_r']:+.3f}")

    print("\n Weakest / Inverse Pairs (Lowest Correlation):")
    print("-" * 55)
    for idx, row in pairs_df.tail(3).iterrows():
        print(f"  •  {row['Pollutant_1']:<6} &  {row['Pollutant_2']:<6} : r = {row['Spearman_r']:+.3f}")
    print("=" * 65)

    return corr_matrix, pairs_df

if __name__ == "__main__":
    run_correlation_analysis()
