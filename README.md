# ClimateSense

AI-powered environmental risk intelligence for India: heat/AQI forecasting with explicit
**El Niño (ENSO) integration**, a WBGT/WHO-anchored risk score, a human-impact translation
layer for athletes and schools, and a **pollution-source policy brief** for government
audiences grounded in published source-apportionment studies.

## 1. Data Source Matrix

| Source | Data Type | Endpoint | Cadence | Cost / Key |
|---|---|---|---|---|
| Open-Meteo Forecast | Hourly weather: temp, RH, wind speed+direction, UV, precip, pressure | `api.open-meteo.com/v1/forecast` | 15–30 min pull | Free, no key |
| Open-Meteo Archive | Historical hourly weather (2000→) for model training | `archive-api.open-meteo.com/v1/archive` | One-time seed | Free, no key |
| Open-Meteo Air Quality | Hourly PM2.5, PM10, NO2, O3, SO2, CO (CAMS model), historical + forecast | `air-quality-api.open-meteo.com/v1/air-quality` | 15–30 min pull + seed | Free, no key |
| AQICN (WAQI) | Live station AQI + pollutant readings (best Indian station coverage) | `api.waqi.info/feed/{city}/` | 15–30 min pull | Free token |
| **NOAA CPC ONI** | **El Niño index: 3-month running Niño-3.4 SST anomaly + ENSO phase** | `cpc.ncep.noaa.gov/data/indices/oni.ascii.txt` | Monthly (pulled each run, idempotent) | Free, no key |
| IIT-Kanpur / TERI-ARAI / SAFAR / CSTEP studies | Static PM2.5 source apportionment shares per city × season | Encoded in `data/reference/reference_tables.py` | Static | Free (published) |
| CAQM GRAP framework | Official AQI-triggered policy stages I–IV | Encoded in reference tables | Static | Free (published) |

## 2. End-to-End Workflow

```
NOAA ONI ─┐
Open-Meteo weather ──► pipeline/fetch.py (async, tenacity retries)
Open-Meteo AQ ─┘              │
AQICN live ───────────────────┤
                              ▼
                pipeline/transform.py
                (heat index, CPCB AQI from breakpoints,
                 unit fixes, ffill limits, dedup)
                              ▼
                pipeline/load.py ──► Postgres/SQLite
                (idempotent ON CONFLICT upserts)
                              ▼
     ┌────────────────────────┴─────────────────────────┐
     ▼                                                   ▼
ml/features.py                              services/source_inference.py
(cyclic time, lags, rolling,                (published apportionment prior ×
 ONI, ONI lag-3m, phase,                     live meteo/temporal modifiers →
 ONI×monsoon, ONI×winter)                    ranked sources → interventions
     ▼                                        + GRAP stage)
ml/train.py                                              │
  Prophet (monsoon seasonality                           │
  + ONI extra regressor) — temp & PM2.5                  │
  XGBoost 4-tier risk (walk-forward CV)                  │
     ▼                                                   ▼
              api/main.py (FastAPI, models loaded at startup)
   /climate/{city}  /forecast/{city}  /policy/{city}  /enso  /health
                              ▼
              services/translation.py → plain-language guidance
```

**Scheduling:** GitHub Actions cron (`.github/workflows/etl.yml`) runs the live ETL every
30 minutes against a hosted Postgres — zero server cost. CI runs pytest on every push.

## 3. El Niño Feature Design (why each feature exists)

| Feature | Rationale |
|---|---|
| `oni` | Current Niño-3.4 SST anomaly; El Niño summers in India are hotter (weakened monsoon, less cloud cover) — direct heat-risk driver. |
| `oni_lag_3m` | ENSO→monsoon teleconnection acts with a lag; a developing El Niño in spring predicts a deficient monsoon. |
| `enso_phase` | Categorical neutral / El Niño / La Niña regime flag. |
| `oni_x_monsoon` | The rainfall-deficit effect only expresses Jun–Sep. |
| `oni_x_winter` | ENSO modulates winter ventilation/inversion strength over the Indo-Gangetic plain — a driver of Delhi PM2.5 accumulation episodes. |
| Prophet `add_regressor("oni")` | Injects the ENSO signal directly into the 7-day temp/PM2.5 forecasters; ONI is monthly, so the latest value is carried across the short horizon (documented assumption). |

## 4. Tech Stack

| Layer | Choice | Cost |
|---|---|---|
| Coding & processing | Python 3.11, aiohttp + tenacity (async ingest), pandas/NumPy (transform), GitHub Actions (orchestration + CI) | $0 |
| Storage | SQLite locally → **Supabase Postgres free tier** (500 MB) in prod; schema in `db/schema.sql` | $0 |
| ML | Prophet (forecast + ONI regressor), XGBoost (risk classifier), scikit-learn (TimeSeriesSplit, metrics), joblib (serialisation) | $0 |
| Serving | FastAPI + Uvicorn on Render free tier | $0 |

### Low-cost AWS adaptation (when you want the AWS badge on the README)
- **Storage:** swap `DATABASE_URL` to RDS Postgres `db.t3.micro` (12-month free tier), or land raw JSON in S3 (free tier 5 GB) and query with Athena for the lakehouse story.
- **Scheduling:** EventBridge rule → **Lambda** running `run_etl.py live` (Lambda free tier: 1M requests/month — this workload is effectively free forever).
- **Serving:** FastAPI on Lambda via Mangum + API Gateway, or App Runner.
- Nothing in the code changes except environment variables — that's the point of the SQLAlchemy + env-var design.

## 5. Runbook

```bash
git clone <repo> && cd climatesense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add AQICN_TOKEN; leave DATABASE_URL for local SQLite

# 1. Backfill 2 years of history (weather + AQ + ONI) — no API keys needed
python pipeline/run_etl.py seed --days 730

# 2. Train models (Prophet temp + PM2.5 with ONI regressor; XGBoost risk)
python ml/train.py --city 1 --models prophet xgboost
python ml/train.py --city 2 --models prophet xgboost

# 3. Serve
uvicorn api.main:app --reload --port 8000
#   http://localhost:8000/docs

# 4. Tests
pytest tests/ -v
```

## 6. Honest scoping notes (put these in your paper/article too)

- The policy layer performs **source-likelihood inference**, not chemical source
  apportionment. Likelihoods combine published apportionment priors (cited per city ×
  season) with transparent meteorological/temporal modifiers. Verify the encoded share
  ranges against the cited studies before publishing.
- Open-Meteo AQ values come from the CAMS model, not ground stations; AQICN provides the
  station cross-check. Documenting model-vs-station deltas is itself a nice results table.
- ONI is a monthly, slow-moving index: it improves seasonal calibration and regime
  awareness; it will not move day-to-day forecasts much. Say exactly that — it reads as
  rigour, not weakness.
