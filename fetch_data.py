import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from config import API_KEY, DEFAULT_SAMPLE_KEY, BASE_URL, RAW_DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def fetch_hourly_snapshot():
    """
    Fetches one complete hourly snapshot from data.gov.in CPCB resource.
    Handles pagination automatically (data.gov.in provides max 1000 records per call).
    """
    is_sample_key = (API_KEY == DEFAULT_SAMPLE_KEY)
    
    print("=" * 65)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CPCB Data Ingestion...")
    if is_sample_key:
        print("[!] Using Sample Key (limited to 10 records).")
    else:
        print(f"[+] Using Personal API Key ({API_KEY[:8]}...{API_KEY[-4:]})")

    all_records = []
    offset = 0
    limit = 10 if is_sample_key else 1000
    max_retries = 3

    while True:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "offset": offset,
            "limit": limit
        }

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"--> Fetching records (offset={offset}, limit={limit}) [Attempt {attempt}/{max_retries}]...")
                response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=45)
                
                if response.status_code == 200:
                    payload = response.json()
                    records = payload.get("records", [])
                    total_available = payload.get("total", "Unknown")

                    if not records:
                        print("    [Info] End of records reached.")
                        success = True
                        break

                    all_records.extend(records)
                    print(f"    [+] Received {len(records)} records. (Accumulated: {len(all_records)} / Server Total: {total_available})")
                    success = True
                    break
                else:
                    print(f"    [!] HTTP {response.status_code}: {response.text[:120]}")
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"    [!] Network error: {e}")
                time.sleep(3)

        if not success or (success and not records):
            break

        if len(records) < limit or is_sample_key:
            break

        offset += limit
        time.sleep(0.5)  # Respectful pause between pages

    if not all_records:
        print("[!] No records could be collected.")
        return None

    df = pd.DataFrame(all_records)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"cpcb_{timestamp_str}.csv"
    output_path = RAW_DATA_DIR / output_filename

    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Successfully saved {len(df)} records across {df['station'].nunique() if 'station' in df.columns else 'N/A'} stations.")
    print(f"          Saved to: {output_path}")
    print("=" * 65)
    return output_path

if __name__ == "__main__":
    result = fetch_hourly_snapshot()
    if result is None:
        sys.exit(1)
