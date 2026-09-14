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
import logging
import os
import time
from collections import defaultdict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("api")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, MODEL_DIR
from pipeline.load import init_db
from ml.risk_score import compute_heat_risk_score, tier_name
from services.translation import translate, enso_narrative
from services.source_inference import government_brief
from services.data_access import (latest_observation, today_hourly_scores,
                                  latest_enso, enso_outlook, uv_peak_today)
from services.assistant import answer_question

app = FastAPI(title="ClimateSense API", version="1.0")
_frontend_origins = [o.strip() for o in
                     os.getenv("FRONTEND_URL", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,  # never "*" in prod
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

_MODELS: dict = {}   # loaded once at startup - never per request


@app.on_event("startup")
def _startup() -> None:
    init_db()
    for cid in CITIES:
        pkl = os.path.join(MODEL_DIR, f"xgb_risk_city{cid}.pkl")
        if not os.path.exists(pkl):
            continue
        try:
            _MODELS[cid] = joblib.load(pkl)
        except Exception as e:
            # A half-written file (retraining in progress) or a version skew
            # shouldn't take the whole API down - the endpoints degrade fine.
            logger.warning("skipping unreadable model for city %s: %s", cid, e)


def _predict_tomorrow_tier(city_id: int) -> dict | None:
    """Risk tier ~24h out, scored from Open-Meteo's own forecast for that hour.

    Deliberately NOT the XGBoost classifier. Repointing that model at a +24h
    label (rather than the same hour, which was circular) showed it scores
    below a trivial "tomorrow looks like today" persistence baseline on
    matched walk-forward folds - so serving its output would be worse than
    useless here. We already ingest the hourly weather forecast, so applying
    the same transparent risk formula to the forecast row is both more
    accurate and easier to explain.
    """
    try:
        from services.data_access import forecast_observation
        fc = forecast_observation(city_id, hours_ahead=24)
        if fc is None:
            return None
        score = compute_heat_risk_score(fc["temp_c"], fc["humidity_pct"],
                                        fc["aqi_cpcb"], fc["uv_index"],
                                        fc.get("solar_wm2"))
        if score is None:
            return None
        return {"horizon_h": 24, "for_ts": str(fc["ts"]),
                "risk_score": score, "tier_name": tier_name(score),
                "basis": "risk formula applied to the 24h weather forecast"}
    except Exception as e:           # never let this break the page
        logger.warning("tomorrow-tier prediction failed for city %s: %s", city_id, e)
        return None


def tier_name_from_index(idx: int) -> str:
    from ml.risk_score import TIERS
    return TIERS[idx] if 0 <= idx < len(TIERS) else "UNKNOWN"


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(_MODELS.keys())}


@app.get("/models")
def models():
    """Honest model scorecard.

    The risk classifier is reported with the persistence baseline it was
    measured against on the same walk-forward folds. It currently scores
    BELOW that baseline in every city, which is why /climate's 24h outlook
    is computed from the weather forecast instead of from this model.
    """
    out = {}
    for cid, b in _MODELS.items():
        cv, base = b.get("cv_f1_mean"), b.get("baseline_f1_mean")
        out[CITIES[cid]["name"]] = {
            "target": f"risk tier +{b.get('horizon_h', 24)}h",
            "cv_f1_weighted": round(cv, 3) if cv is not None else None,
            "persistence_baseline_f1": round(base, 3) if base is not None else None,
            "lift_vs_baseline": round(cv - base, 3) if cv is not None and base is not None else None,
            "used_in_serving": False,
        }
    return {"risk_classifier": out,
            "note": ("Prophet forecasters drive /forecast. The risk classifier is "
                     "trained and reported but not served: it does not beat a "
                     "persistence baseline at this horizon.")}


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
                                    latest["aqi_cpcb"], latest["uv_index"],
                                    latest.get("solar_wm2"))
    payload = translate(score or 0, latest["temp_c"], latest["aqi_cpcb"],
                        latest["uv_index"], latest["humidity_pct"], sport,
                        CITIES[city_id]["name"], today_hourly_scores(city_id),
                        latest.get("wbgt_c"))
    payload["tier"] = tier_name(score or 0)
    payload["tomorrow"] = _predict_tomorrow_tier(city_id)
    # UV is legitimately 0 after dark, which reads as broken data on its own.
    # Today's peak gives the reader something to anchor against.
    payload["uv_peak_today"] = uv_peak_today(city_id)
    payload["observed"] = {k: (None if pd.isna(v) else v) for k, v in latest.items()
                           if k != "ts"}
    payload["observed_at"] = str(latest["ts"])
    return payload


@app.get("/forecast/{city_id}")
def forecast(city_id: int, target: str = "temp_c", days: int = 7):
    if target not in ("temp_c", "pm25"):
        raise HTTPException(400, "target must be temp_c or pm25")
    from ml.train import forecast_prophet, forecast_metrics
    try:
        fc = forecast_prophet(city_id, target, days)
    except FileNotFoundError:
        raise HTTPException(404, "Model not trained - run ml/train.py")
    return {
        "target": target,
        # Walk-forward MAE/RMSE from training, so the UI can show how much
        # to trust these numbers rather than presenting them bare.
        "accuracy": forecast_metrics(city_id, target),
        "points": [{"date": str(r.ds.date()), "value": round(r.yhat, 2),
                    "lower": round(r.yhat_lower, 2), "upper": round(r.yhat_upper, 2)}
                   for r in fc.itertuples()],
    }


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
        raise HTTPException(404, "ENSO table empty - run the ETL first")
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


# Gemini's free tier is ~250 requests/day and this endpoint is public and
# unauthenticated once the site is deployed, so cap per-IP usage. In-memory
# is fine for a single Render instance; a shared store would be needed if
# this ever scales horizontally.
ASSISTANT_RATE_LIMIT = int(os.getenv("ASSISTANT_RATE_LIMIT", "15"))   # per window
ASSISTANT_RATE_WINDOW_S = int(os.getenv("ASSISTANT_RATE_WINDOW_S", "3600"))
_assistant_hits: dict[str, list[float]] = defaultdict(list)


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    hits = [t for t in _assistant_hits[ip] if now - t < ASSISTANT_RATE_WINDOW_S]
    if len(hits) >= ASSISTANT_RATE_LIMIT:
        retry_in = int(ASSISTANT_RATE_WINDOW_S - (now - hits[0]))
        raise HTTPException(429, f"Rate limit reached ({ASSISTANT_RATE_LIMIT} questions "
                                 f"per hour). Try again in {retry_in // 60 + 1} min.")
    hits.append(now)
    _assistant_hits[ip] = hits


@app.post("/assistant/ask")
def assistant_ask(payload: AssistantQuery, request: Request):
    if payload.city_id not in CITIES:
        raise HTTPException(404, "Unknown city")
    _rate_limit(request)
    try:
        return answer_question(payload.question, payload.city_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
