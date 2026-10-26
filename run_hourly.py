import time
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_entire_pipeline

INTERVAL_SECONDS = 3600  # 1 hour

def push_updates_to_github():
    """Commits and pushes all new data, models, reports, and dashboard to GitHub."""
    try:
        print("\n[Git Sync] Staging all updated models, datasets, and dashboard...")
        subprocess.run(["git", "add", "."], cwd=PROJECT_ROOT, check=True)
        
        # Check if there are changes to commit
        diff_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=PROJECT_ROOT)
        if diff_check.returncode == 0:
            print("[Git Sync] No new changes detected to push.")
            return

        commit_msg = f"Auto: Update hourly dataset, ML models, and live dashboard [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_ROOT, check=True)
        print("[Git Sync] Successfully pushed updates to GitHub!")
        print("          GitHub Pages will update the live website within ~30-60 seconds.")
    except Exception as e:
        print(f"[Git Sync Warning] Could not push to GitHub: {e}")

def main():
    print("=" * 70)
    print(" CPCB Automated Real-Time Machine Learning & Live Hosting Service")
    print(" Frequency: Runs immediately, then every 60 minutes")
    print(" Press Ctrl + C to stop at any time.")
    print("=" * 70)

    try:
        while True:
            cycle_start = datetime.now()
            print(f"\n[CYCLE START] {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                success = run_entire_pipeline()
                if success:
                    push_updates_to_github()
            except Exception as e:
                print(f"[!] Error during pipeline cycle: {e}")

            next_run = cycle_start + timedelta(seconds=INTERVAL_SECONDS)
            print(f"\n--> Next automated cycle scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"--> Sleeping for {INTERVAL_SECONDS // 60} minutes...")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[STOPPED] Service stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
