import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fetch_data import fetch_hourly_snapshot
from src.preprocess import run_preprocessing_pipeline
from src.analysis_correlation import run_correlation_analysis
from src.analysis_pca import run_pca_analysis
from src.model_clustering import run_clustering_and_outliers
from src.model_forecasting import run_forecasting_pipeline
from src.generate_dashboard import generate_command_center_dashboard
from src.generate_map import generate_interactive_hotspot_map

def run_entire_pipeline():
    """
    Executes the complete end-to-end Machine Learning pipeline:
    1. Ingestion: Fetches the latest live hourly snapshot from CPCB.
    2. Preprocessing: Cleans, pivots, and standardizes multi-pollutant data.
    3. Correlation: Computes Spearman rank correlation and co-occurrence matrix.
    4. PCA: Performs dimensionality reduction and calculates factor loadings.
    5. Clustering: Fits K-Means, profiles clusters, and ranks priority hotspots.
    6. Visualization: Generates interactive hotspot map and executive dashboard.
    7. Hosting Sync: Copies dashboard to index.html and docs/index.html for GitHub Pages.
    """
    start_time = datetime.now()
    print("\n" + "#" * 70)
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING COMPLETE ML PIPELINE RUN")
    print("#" * 70)

    # Step 1: Fetch latest hourly data
    print("\n>>> STEP 1: Attempting to Fetch Latest Live CPCB Snapshot...")
    try:
        saved_snapshot = fetch_hourly_snapshot()
        if not saved_snapshot:
            print("[!] Live fetch returned no new records. Continuing with all existing snapshots in data/raw/...")
    except Exception as e:
        print(f"[!] Live fetch encountered an issue: {e}. Continuing with all existing snapshots in data/raw/...")

    # Step 2: Preprocess and Pivot
    print("\n>>> STEP 2: Running Preprocessing & Standardization...")
    run_preprocessing_pipeline()

    # Step 3: Correlation Analysis
    print("\n>>> STEP 3: Running Spearman Correlation Analysis...")
    run_correlation_analysis()

    # Step 4: PCA
    print("\n>>> STEP 4: Running PCA Dimensionality Reduction...")
    run_pca_analysis()

    # Step 5: Clustering & Priority Ranking
    print("\n>>> STEP 5: Running K-Means Clustering & Outlier Profiling...")
    run_clustering_and_outliers()

    # Step 6: Next-Hour Time-Series Forecasting
    print("\n>>> STEP 6: Training Next-Hour Forecasting Regressors...")
    run_forecasting_pipeline()

    # Step 7: Generate Dashboard & Map
    print("\n>>> STEP 7: Compiling Executive Dashboard & GIS Map...")
    dashboard_path = generate_command_center_dashboard()
    generate_interactive_hotspot_map()

    # Step 8: Prepare for Free Cloud Hosting (GitHub Pages)
    # GitHub Pages serves from root / or /docs folder
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Copy dashboard.html to docs/index.html and root index.html
    shutil.copy(dashboard_path, docs_dir / "index.html")
    shutil.copy(dashboard_path, PROJECT_ROOT / "index.html")
    print("\n>>> STEP 8: Synced live dashboard to docs/index.html and index.html (ready for GitHub Pages)")

    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print(f" COMPLETE ML PIPELINE FINISHED SUCCESSFULLY IN {duration:.1f}s")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_entire_pipeline()
