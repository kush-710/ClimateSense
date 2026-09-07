"""FastAPI serving layer.

  uvicorn api.main:app --reload --port 8000

Endpoints
  GET /health
  GET /climate/{city_id}?sport=cricket_fielding   -> live risk + translation
  GET /forecast/{city_id}?target=temp_c&days=7    -> Prophet forecast (ONI regressor)
  GET /policy/{city_id}                           -> government source/intervention brief
  GET /enso                                       -> latest ONI + phase
"""
from __future__ import annotations
import os
from datetime import datetime, timezone, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, MODEL_DIR, IST
from pipeline.load import engine, init_db
from ml.risk_score import compute_heat_risk_score, tier_name
from services.translation import translate
from services.source_inference import government_brief

app = FastAPI(title="ClimateSense API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],  # never "*" in prod
    allow_methods=["GET"], allow_headers=["*"],
)

_MODELS: dict = {}   # loaded once at startup — never per request


@app.on_event("startup")
def _startup() -> None:
    init_db()
    for cid in CITIES:
        pkl = os.path.join(MODEL_DIR, f"xgb_risk_city{cid}.pkl")
        if os.path.exists(pkl):
            _MODELS[cid] = joblib.load(pkl)


def _latest_row(city_id: int) -> dict:
    q = text("""
        SELECT w.ts, w.temp_c, w.humidity_pct, w.wind_kph, w.uv_index,
               a.aqi_cpcb, a.pm25, a.pm10
        FROM weather_data w
        LEFT JOIN air_quality a ON a.city_id = w.city_id AND a.ts = w.ts
        WHERE w.city_id = :c
        ORDER BY w.ts DESC LIMIT 1""")
    row = pd.read_sql(q, engine, params={"c": city_id}, parse_dates=["ts"])
    if row.empty:
        raise HTTPException(404, "No data — run the ETL first")
    return row.iloc[0].to_dict()


def _today_hourly_scores(city_id: int) -> list[dict]:
    q = text("""
        SELECT w.ts, w.temp_c, w.humidity_pct, w.uv_index, a.aqi_cpcb
        FROM weather_data w
        LEFT JOIN air_quality a ON a.city_id = w.city_id AND a.ts = w.ts
        WHERE w.city_id = :c AND w.ts >= :start
        ORDER BY w.ts""")
    start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    df = pd.read_sql(q, engine, params={"c": city_id, "start": start},
                     parse_dates=["ts"])
    out = []
    for _, r in df.iterrows():
        s = compute_heat_risk_score(r["temp_c"], r["humidity_pct"],
                                    r["aqi_cpcb"], r["uv_index"])
        out.append({"hour": int(pd.Timestamp(r["ts"]).hour), "risk_score": s})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(_MODELS.keys())}


@app.get("/climate/{city_id}")
def climate(city_id: int, sport: str = "general"):
    if city_id not in CITIES:
        raise HTTPException(404, "Unknown city")
    latest = _latest_row(city_id)
    score = compute_heat_risk_score(latest["temp_c"], latest["humidity_pct"],
                                    latest["aqi_cpcb"], latest["uv_index"])
    payload = translate(score or 0, latest["temp_c"], latest["aqi_cpcb"],
                        latest["uv_index"], latest["humidity_pct"], sport,
                        CITIES[city_id]["name"], _today_hourly_scores(city_id))
    payload["tier"] = tier_name(score or 0)
    payload["observed"] = {k: (None if pd.isna(v) else v) for k, v in latest.items()
                           if k != "ts"}
    payload["observed_at"] = str(latest["ts"])
    return payload


@app.get("/forecast/{city_id}")
def forecast(city_id: int, target: str = "temp_c", days: int = 7):
    if target not in ("temp_c", "pm25"):
        raise HTTPException(400, "target must be temp_c or pm25")
    from ml.train import forecast_prophet
    try:
        fc = forecast_prophet(city_id, target, days)
    except FileNotFoundError:
        raise HTTPException(404, "Model not trained — run ml/train.py")
    return [{"date": str(r.ds.date()), "value": round(r.yhat, 2),
             "lower": round(r.yhat_lower, 2), "upper": round(r.yhat_upper, 2)}
            for r in fc.itertuples()]


@app.get("/policy/{city_id}")
def policy(city_id: int):
    if city_id not in CITIES:
        raise HTTPException(404, "Unknown city")
    latest = _latest_row(city_id)
    def _clean(v):
        return None if v is None or pd.isna(v) else float(v)
    return government_brief(
        CITIES[city_id]["name"], pd.Timestamp(latest["ts"]).to_pydatetime(),
        _clean(latest["aqi_cpcb"]), _clean(latest["pm25"]), _clean(latest["pm10"]),
        _clean(latest["wind_kph"]), _clean(latest["humidity_pct"]))


@app.get("/enso")
def enso():
    df = pd.read_sql(text("SELECT * FROM enso_index ORDER BY year DESC, month DESC LIMIT 6"),
                     engine)
    if df.empty:
        raise HTTPException(404, "ENSO table empty — run the ETL first")
    latest = df.iloc[0]
    return {"latest": {"season": latest["season"], "year": int(latest["year"]),
                       "oni": float(latest["oni"]), "phase": latest["phase"]},
            "recent": df[["season", "year", "oni", "phase"]].to_dict(orient="records")}
