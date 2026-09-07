"""Central configuration. All secrets come from environment variables."""
import os
from datetime import timezone, timedelta

from dotenv import load_dotenv
load_dotenv()  # loads .env into the process environment if present (no-op in prod, where
                # real env vars are set directly — e.g. Render/GitHub Actions secrets)

IST = timezone(timedelta(hours=5, minutes=30))

# --- Cities (extend by adding rows; keep ids stable) ---
CITIES = {
    1: {"name": "Delhi",     "lat": 28.6139, "lon": 77.2090, "aqicn_slug": "delhi"},
    2: {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "aqicn_slug": "bangalore"},
    3: {"name": "Mumbai",    "lat": 19.0760, "lon": 72.8777, "aqicn_slug": "mumbai"},
    4: {"name": "Chennai",   "lat": 13.0827, "lon": 80.2707, "aqicn_slug": "chennai"},
    5: {"name": "Kolkata",   "lat": 22.5726, "lon": 88.3639, "aqicn_slug": "kolkata"},
    6: {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "aqicn_slug": "hyderabad"},
    7: {"name": "Pune",      "lat": 18.5204, "lon": 73.8567, "aqicn_slug": "pune"},
    8: {"name": "Lucknow",   "lat": 26.8467, "lon": 80.9462, "aqicn_slug": "lucknow"},
}

# Indo-Gangetic Plain cities: the winter-inversion/PM2.5-accumulation ENSO
# narrative is a regional effect specific to this belt, not all of India.
IGP_CITIES = {"Delhi", "Lucknow"}

# GRAP (Graded Response Action Plan) is a CAQM policy framework legally
# scoped to Delhi-NCR — it does not apply to other cities' AQI readings.
GRAP_APPLICABLE_CITIES = {"Delhi"}

# --- API endpoints (all free) ---
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AQ       = "https://air-quality-api.open-meteo.com/v1/air-quality"
AQICN_FEED          = "https://api.waqi.info/feed/{slug}/"
# NOAA CPC Oceanic Nino Index (3-month running mean SST anomaly, Nino 3.4)
NOAA_ONI_URL        = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
# NOAA CPC official probabilistic ENSO forecast (El Nino/Neutral/La Nina % per
# season, next ~9 overlapping 3-month seasons). HTML page with a data table —
# no clean CSV/JSON is published, so pipeline/fetch.py regex-parses the table.
NOAA_ENSO_OUTLOOK_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/"

AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")          # free at aqicn.org/data-platform/token
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///climatesense.db"                      # zero-cost local default; swap for Supabase/RDS
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")    # free at aistudio.google.com/apikey
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

MODEL_DIR = os.getenv("MODEL_DIR", "ml/saved_models")

# Hourly variables pulled from Open-Meteo (weather + AQ)
OM_WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "uv_index", "precipitation", "surface_pressure",
]
OM_AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
