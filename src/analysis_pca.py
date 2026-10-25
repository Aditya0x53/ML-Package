import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_pca_analysis(n_components=3):
    scaled_file = PROCESSED_DATA_DIR / "station_pollutants_scaled.csv"
    if not scaled_file.exists():
        raise FileNotFoundError(f"{scaled_file} not found. Run src/preprocess.py first.")

    df = pd.read_csv(scaled_file)
    X = df[TARGET_POLLUTANTS].values

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Explained variance
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    print("=" * 65)
    print(" PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print(f" Features: {TARGET_POLLUTANTS}")
    print(f" Stations: {len(df)}")
    print("=" * 65)

    var_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Explained_Variance': np.round(evr, 4),
        'Cumulative_Variance': np.round(cum_evr, 4)
    })
    print("\nExplained Variance by Component:")
    print(var_df.to_string(index=False))

    # Feature Loadings (which pollutants drive each component)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}_Loading' for i in range(n_components)],
        index=TARGET_POLLUTANTS
    ).round(3)

    loadings_path = OUTPUT_DIR / "pca_component_loadings.csv"
    loadings_df.to_csv(loadings_path)
    print(f"\n[+] Factor Loadings saved to: {loadings_path}\n")
    print(loadings_df)

    # Add PC1 and PC2 coordinates to dataframe for visualization/clustering
    pca_coords_df = df.copy()
    for i in range(n_components):
        pca_coords_df[f'PC{i+1}'] = np.round(X_pca[:, i], 4)

    pca_coords_path = PROCESSED_DATA_DIR / "station_pca_coordinates.csv"
    pca_coords_df.to_csv(pca_coords_path, index=False)
    print(f"\n[+] Station 2D/3D PCA Coordinates saved to: {pca_coords_path}")
    print("=" * 65)

    return pca, pca_coords_df, loadings_df

if __name__ == "__main__":
    run_pca_analysis()
