import time
import sys
from datetime import datetime, timedelta
from fetch_data import fetch_hourly_snapshot

INTERVAL_SECONDS = 3600  # 1 hour

def main():
    print("=" * 60)
    print(" CPCB Hourly Data Ingestion Service Started")
    print(f" Interval: Every {INTERVAL_SECONDS // 60} minutes (1 hour)")
    print(" Press Ctrl + C to stop at any time.")
    print("=" * 60)

    try:
        while True:
            start_time = datetime.now()
            print(f"\n[TRIGGER] Starting hourly run at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                fetch_hourly_snapshot()
            except Exception as e:
                print(f"[!] Exception during fetch: {e}")

            next_run = start_time + timedelta(seconds=INTERVAL_SECONDS)
            print(f"--> Next scheduled fetch: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"--> Sleeping for 60 minutes...")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[STOPPED] Scheduler terminated by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
