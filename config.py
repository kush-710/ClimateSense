"""Central configuration. All secrets come from environment variables."""
import os
from datetime import timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# --- Cities (extend by adding rows; keep ids stable) ---
CITIES = {
    1: {"name": "Delhi",     "lat": 28.6139, "lon": 77.2090, "aqicn_slug": "delhi"},
    2: {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "aqicn_slug": "bangalore"},
}

# --- API endpoints (all free) ---
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AQ       = "https://air-quality-api.open-meteo.com/v1/air-quality"
AQICN_FEED          = "https://api.waqi.info/feed/{slug}/"
# NOAA CPC Oceanic Nino Index (3-month running mean SST anomaly, Nino 3.4)
NOAA_ONI_URL        = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")          # free at aqicn.org/data-platform/token
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///climatesense.db"                      # zero-cost local default; swap for Supabase/RDS
)

MODEL_DIR = os.getenv("MODEL_DIR", "ml/saved_models")

# Hourly variables pulled from Open-Meteo (weather + AQ)
OM_WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "uv_index", "precipitation", "surface_pressure",
]
OM_AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
