import time
import sys
import subprocess
from datetime import datetime, timedelta
from fetch_data import fetch_hourly_snapshot

INTERVAL_SECONDS = 3600  # 1 hour (3600 seconds)

def push_to_github(saved_file):
    """Automatically commits and pushes the new hourly snapshot to GitHub."""
    try:
        print("[Git] Committing and pushing snapshot to GitHub...")
        subprocess.run(["git", "add", str(saved_file)], check=True, capture_output=True)
        commit_msg = f"Auto: Ingest CPCB snapshot {saved_file.name}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
        print("[Git] Successfully pushed latest snapshot to GitHub repository!")
    except subprocess.CalledProcessError as e:
        print(f"[Git Note] Git commit/push skipped or no changes: {e}")
    except Exception as e:
        print(f"[Git Error] Could not push to GitHub: {e}")

def main():
    print("=" * 65)
    print(" CPCB Hourly Ingestion & Auto-GitHub Push Service")
    print(f" Interval: Runs immediately, then every {INTERVAL_SECONDS // 60} minutes")
    print(" Press Ctrl + C to stop at any time.")
    print("=" * 65)

    try:
        while True:
            start_time = datetime.now()
            print(f"\n[TRIGGER] Ingestion started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                saved_file = fetch_hourly_snapshot()
                if saved_file:
                    push_to_github(saved_file)
            except Exception as e:
                print(f"[!] Error during hourly cycle: {e}")

            next_run = start_time + timedelta(seconds=INTERVAL_SECONDS)
            print(f"--> Next scheduled fetch: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"--> Sleeping for 60 minutes...")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[STOPPED] Hourly scheduler stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
