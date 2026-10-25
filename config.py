import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# API Configuration
# Fallback to the public sample key if user hasn't set their key yet
DEFAULT_SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
_env_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
API_KEY = _env_key if _env_key else DEFAULT_SAMPLE_KEY

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

# Data Directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
