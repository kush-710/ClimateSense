"""FastAPI serving layer.

  uvicorn api.main:app --reload --port 8000

Endpoints
  GET /health
  GET /cities                                     -> id -> {name, lat, lon} for every city
  GET /climate/{city_id}?sport=cricket_fielding   -> live risk + translation
  GET /forecast/{city_id}?target=temp_c&days=7    -> Prophet forecast (ONI regressor)
  GET /policy/{city_id}                           -> government source/intervention brief
  GET /enso?city_id=1                             -> latest ONI + city-aware narrative + outlook
  POST /assistant/ask                             -> Gemini chat, grounded in the above
"""
from __future__ import annotations
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, MODEL_DIR
from pipeline.load import init_db
from ml.risk_score import compute_heat_risk_score, tier_name
from services.translation import translate, enso_narrative
from services.source_inference import government_brief
from services.data_access import latest_observation, today_hourly_scores, latest_enso, enso_outlook
from services.assistant import answer_question

app = FastAPI(title="ClimateSense API", version="1.0")
_frontend_origins = [o.strip() for o in
                     os.getenv("FRONTEND_URL", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,  # never "*" in prod
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

_MODELS: dict = {}   # loaded once at startup — never per request


@app.on_event("startup")
def _startup() -> None:
    init_db()
    for cid in CITIES:
        pkl = os.path.join(MODEL_DIR, f"xgb_risk_city{cid}.pkl")
        if os.path.exists(pkl):
            _MODELS[cid] = joblib.load(pkl)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(_MODELS.keys())}


@app.get("/cities")
def cities():
    return {cid: {"name": c["name"], "lat": c["lat"], "lon": c["lon"]}
            for cid, c in CITIES.items()}


@app.get("/climate/{city_id}")
def climate(city_id: int, sport: str = "general"):
    if city_id not in CITIES:
        raise HTTPException(404, "Unknown city")
    latest = latest_observation(city_id)
    score = compute_heat_risk_score(latest["temp_c"], latest["humidity_pct"],
                                    latest["aqi_cpcb"], latest["uv_index"])
    payload = translate(score or 0, latest["temp_c"], latest["aqi_cpcb"],
                        latest["uv_index"], latest["humidity_pct"], sport,
                        CITIES[city_id]["name"], today_hourly_scores(city_id))
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
    latest = latest_observation(city_id)
    def _clean(v):
        return None if v is None or pd.isna(v) else float(v)
    return government_brief(
        CITIES[city_id]["name"], pd.Timestamp(latest["ts"]).to_pydatetime(),
        _clean(latest["aqi_cpcb"]), _clean(latest["pm25"]), _clean(latest["pm10"]),
        _clean(latest["wind_kph"]), _clean(latest["humidity_pct"]))


@app.get("/enso")
def enso(city_id: int | None = None):
    if city_id is not None and city_id not in CITIES:
        raise HTTPException(404, "Unknown city")
    df = latest_enso()
    if df.empty:
        raise HTTPException(404, "ENSO table empty — run the ETL first")
    latest = df.iloc[0]
    city_name = CITIES[city_id]["name"] if city_id is not None else None

    outlook_df = enso_outlook()
    outlook = outlook_df[["season", "year", "month", "el_nino_pct", "neutral_pct",
                          "la_nina_pct"]].to_dict(orient="records") if not outlook_df.empty else []
    outlook_headline = None
    if outlook:
        nxt = outlook[0]
        favoured = max(("el_nino", nxt["el_nino_pct"]), ("neutral", nxt["neutral_pct"]),
                       ("la_nina", nxt["la_nina_pct"]), key=lambda x: x[1])
        outlook_headline = (f"CPC forecasts {favoured[0].replace('_', ' ')} most likely "
                           f"({favoured[1]}%) for {nxt['season']} {nxt['year']}")

    return {"latest": {"season": latest["season"], "year": int(latest["year"]),
                       "oni": float(latest["oni"]), "phase": latest["phase"],
                       "narrative": enso_narrative(latest["phase"], int(latest["month"]), city_name)},
            "recent": df[["season", "year", "oni", "phase"]].to_dict(orient="records"),
            "outlook": outlook, "outlook_headline": outlook_headline}


class AssistantQuery(BaseModel):
    question: str
    city_id: int = 1


@app.post("/assistant/ask")
def assistant_ask(payload: AssistantQuery):
    if payload.city_id not in CITIES:
        raise HTTPException(404, "Unknown city")
    try:
        return answer_question(payload.question, payload.city_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
