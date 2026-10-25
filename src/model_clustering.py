import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE', 'NH3']
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_cluster_numbers(X, k_range=range(2, 9)):
    """Computes Inertia (Elbow), Silhouette Score, and Davies-Bouldin index for k in k_range."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        results.append({
            'k': k,
            'inertia': round(km.inertia_, 2),
            'silhouette_score': round(sil, 4),
            'davies_bouldin_index': round(db, 4)
        })
    return pd.DataFrame(results)

def label_cluster_profiles(unscaled_df, cluster_col='cluster'):
    """Generates an intuitive environmental name for each cluster based on average pollutant levels."""
    medians = unscaled_df.groupby(cluster_col)[TARGET_POLLUTANTS].median()
    overall_medians = unscaled_df[TARGET_POLLUTANTS].median()

    profile_names = {}
    for cl in medians.index:
        cl_med = medians.loc[cl]
        # Compare cluster medians with national medians
        pm_high = (cl_med['PM2.5'] > 1.3 * overall_medians['PM2.5']) or (cl_med['PM10'] > 1.3 * overall_medians['PM10'])
        gas_high = (cl_med['NO2'] > 1.3 * overall_medians['NO2']) or (cl_med['CO'] > 1.3 * overall_medians['CO'])
        so2_high = (cl_med['SO2'] > 1.3 * overall_medians['SO2'])
        pm_low = (cl_med['PM2.5'] < overall_medians['PM2.5']) and (cl_med['PM10'] < overall_medians['PM10'])

        if pm_high and gas_high:
            profile_names[cl] = "Cluster {}: Multi-Pollutant Severe Hotspot".format(cl)
        elif pm_high and not gas_high:
            profile_names[cl] = "Cluster {}: Particulate & Dust Dominated".format(cl)
        elif gas_high or so2_high:
            profile_names[cl] = "Cluster {}: Vehicular & Industrial Gas Dominated".format(cl)
        elif pm_low:
            profile_names[cl] = "Cluster {}: Relatively Low / Baseline Ambient".format(cl)
        else:
            profile_names[cl] = "Cluster {}: Moderate Mixed Background".format(cl)

    return profile_names, medians

def run_clustering_and_outliers(n_clusters=None):
    scaled_file = PROCESSED_DATA_DIR / "station_pollutants_scaled.csv"
    cleaned_file = PROCESSED_DATA_DIR / "station_pollutants_cleaned.csv"
    pca_file = PROCESSED_DATA_DIR / "station_pca_coordinates.csv"

    if not scaled_file.exists():
        raise FileNotFoundError("Processed files missing. Run src/preprocess.py first.")

    scaled_df = pd.read_csv(scaled_file)
    cleaned_df = pd.read_csv(cleaned_file)
    pca_df = pd.read_csv(pca_file) if pca_file.exists() else None

    X = scaled_df[TARGET_POLLUTANTS].values

    # Step 1: Model Evaluation across K=2 to 8
    eval_df = evaluate_cluster_numbers(X)
    eval_path = OUTPUT_DIR / "clustering_metrics_evaluation.csv"
    eval_df.to_csv(eval_path, index=False)

    print("=" * 65)
    print(" K-MEANS CLUSTERING EVALUATION")
    print("=" * 65)
    print(eval_df.to_string(index=False))

    # Choose optimal K based on highest Silhouette score if not provided
    if n_clusters is None:
        best_k = eval_df.sort_values(by='silhouette_score', ascending=False).iloc[0]['k']
        # Typically 3 or 4 clusters is most interpretable for environmental AQI
        optimal_k = int(best_k) if best_k in [3, 4, 5] else 4
    else:
        optimal_k = n_clusters

    print(f"\n[+] Selected Optimal Clusters: K = {optimal_k}")

    # Step 2: Fit Final K-Means Model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X)

    # Step 3: Outlier Analysis (Distance from cluster centroid in 7D space)
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X - centroids[cluster_labels], axis=1)

    # 95th percentile distance threshold for outliers
    outlier_threshold = np.percentile(distances, 95)
    is_outlier = distances > outlier_threshold

    # Build Master Results DataFrame
    result_df = cleaned_df.copy()
    result_df['cluster'] = cluster_labels
    result_df['centroid_distance'] = np.round(distances, 3)
    result_df['is_outlier'] = is_outlier

    if pca_df is not None:
        result_df['PC1'] = pca_df['PC1']
        result_df['PC2'] = pca_df['PC2']

    # Step 4: Cluster Profiling
    profile_names, cluster_medians = label_cluster_profiles(result_df, 'cluster')
    result_df['cluster_profile'] = result_df['cluster'].map(profile_names)

    # Save cluster profile medians table
    cluster_medians_path = OUTPUT_DIR / "cluster_profiles_summary.csv"
    cluster_medians.round(2).to_csv(cluster_medians_path)

    print("\n Cluster Profiles Summary (Median Pollutant Concentrations):")
    print("-" * 65)
    for cl, name in profile_names.items():
        count = (result_df['cluster'] == cl).sum()
        pct = (count / len(result_df)) * 100
        print(f" {name} ({count} stations, {pct:.1f}%):")
        meds = cluster_medians.loc[cl]
        print(f"    PM2.5: {meds['PM2.5']} | PM10: {meds['PM10']} | NO2: {meds['NO2']} | SO2: {meds['SO2']} | CO: {meds['CO']}")

    # Step 5: Ranked Priority List for Interventions
    # Composite Priority Score: combines standardized pollutant severity + outlier deviation
    # Particulates & NO2 get higher weight in accordance with CPCB health risk index
    weights = {'PM2.5': 0.30, 'PM10': 0.25, 'NO2': 0.20, 'SO2': 0.10, 'CO': 0.05, 'OZONE': 0.05, 'NH3': 0.05}
    severity_score = sum(scaled_df[pol] * w for pol, w in weights.items())
    
    # Priority Score = Severity Score + 0.2 * Distance
    result_df['priority_score'] = np.round(severity_score + (0.2 * distances), 3)
    result_df['priority_rank'] = result_df['priority_score'].rank(ascending=False, method='min').astype(int)
    
    # Sort by priority rank
    result_df = result_df.sort_values(by='priority_rank').reset_index(drop=True)

    # Save final clustered dataset
    clustered_path = PROCESSED_DATA_DIR / "station_clusters_and_priority.csv"
    result_df.to_csv(clustered_path, index=False)

    print(f"\n[+] Master Clustered & Ranked Dataset saved to: {clustered_path}")

    # Top 10 Priority Hotspots Needing Urgent Attention
    top10 = result_df[['priority_rank', 'station', 'city', 'state', 'PM2.5', 'PM10', 'NO2', 'cluster_profile', 'is_outlier']].head(10)
    top10_path = OUTPUT_DIR / "top_10_priority_hotspots.csv"
    top10.to_csv(top10_path, index=False)

    print("\n Top 10 Urgent Pollution Hotspots (Ranked Priority List):")
    print("-" * 65)
    print(top10[['priority_rank', 'city', 'state', 'PM2.5', 'PM10', 'is_outlier']].to_string(index=False))
    print("=" * 65)

    return result_df, eval_df

if __name__ == "__main__":
    run_clustering_and_outliers()
