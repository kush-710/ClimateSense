"""Shared read-only DB queries used by both api/main.py and services/assistant.py.

Pulled out of api/main.py so the assistant service can reuse the exact same
'latest observation' / 'today's hourly scores' / 'latest ENSO' logic without
api/main.py and services/assistant.py importing each other.
"""
from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import text

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IST
from pipeline.load import engine


def _with_latest_aq(wx: pd.DataFrame, city_id: int) -> pd.DataFrame:
    """Attach the most recent air-quality reading at-or-before each weather
    timestamp (pandas merge_asof, direction='backward') instead of requiring
    an exact-timestamp match.

    Open-Meteo's AQ (CAMS model) feed routinely lags the weather feed by up
    to a day or two — its trailing hours are often still None while the
    model run finishes — so an exact `a.ts = w.ts` join returns null AQI for
    literally every 'current' observation, not just an occasional gap.
    """
    # Only rows with an actual computed AQI: Open-Meteo's CAMS feed can have
    # gaps longer than clean_air_quality()'s conservative 4h ffill window
    # (observed: 14+ consecutive null hours), so the single most recent row
    # can itself be null — skip straight to the last real reading instead.
    # Sourced from both 'open-meteo' (CAMS model, complete hourly coverage —
    # what the ML models train on) and 'aqicn' (real ground stations, when a
    # free token is configured — usually far fresher). Neither is preferred
    # by name: whichever has the more recent actual reading wins, since
    # that's the more accurate "current" value regardless of source; ties
    # (same timestamp from both) favour the station reading.
    aq = pd.read_sql(
        text("""SELECT ts, aqi_cpcb, pm25, pm10, source FROM air_quality
                WHERE city_id=:c AND aqi_cpcb IS NOT NULL ORDER BY ts"""),
        engine, params={"c": city_id}, parse_dates=["ts"])
    if aq.empty:
        return wx.assign(aqi_cpcb=None, pm25=None, pm10=None, aq_observed_at=None, aq_source=None)
    aq["_pref"] = (aq["source"] != "aqicn").astype(int)  # 0=aqicn, 1=other — lower wins ties
    aq = aq.sort_values(["ts", "_pref"]).drop_duplicates(subset="ts", keep="first").drop(columns="_pref")
    merged = pd.merge_asof(
        wx.sort_values("ts"),
        aq.sort_values("ts").rename(columns={"ts": "aq_observed_at", "source": "aq_source"}),
        left_on="ts", right_on="aq_observed_at", direction="backward")
    return merged


def latest_observation(city_id: int) -> dict:
    # weather_data holds the FULL 7-day forecast from every live pull, not
    # just the current hour — a bare ORDER BY ts DESC grabs the far edge of
    # next week's forecast, not "now". Bound to ts <= real current time so
    # this can only ever return an actual-or-just-passed hour.
    now = datetime.now(IST).replace(tzinfo=None)
    wx = pd.read_sql(
        text("""SELECT ts, temp_c, humidity_pct, wind_kph, uv_index, heat_index_c
                FROM weather_data WHERE city_id=:c AND ts <= :now
                ORDER BY ts DESC LIMIT 1"""),
        engine, params={"c": city_id, "now": now}, parse_dates=["ts"])
    if wx.empty:
        raise HTTPException(404, "No data — run the ETL first")
    return _with_latest_aq(wx, city_id).iloc[0].to_dict()


def today_hourly_scores(city_id: int) -> list[dict]:
    from ml.risk_score import compute_heat_risk_score

    # Same forecast-table caveat as latest_observation(): must bound the
    # upper end too, or "today" silently pulls in the rest of next week's
    # forecast (repeating hours 0-23 across multiple days breaks the
    # contiguous-run logic in compute_safe_window()).
    start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    end = start + timedelta(days=1)
    wx = pd.read_sql(
        text("""SELECT ts, temp_c, humidity_pct, uv_index FROM weather_data
                WHERE city_id=:c AND ts >= :start AND ts < :end ORDER BY ts"""),
        engine, params={"c": city_id, "start": start, "end": end}, parse_dates=["ts"])
    merged = _with_latest_aq(wx, city_id)
    out = []
    for _, r in merged.iterrows():
        s = compute_heat_risk_score(r["temp_c"], r["humidity_pct"],
                                    r["aqi_cpcb"], r["uv_index"])
        out.append({"hour": int(pd.Timestamp(r["ts"]).hour), "risk_score": s})
    return out


def latest_enso(limit: int = 6) -> pd.DataFrame:
    return pd.read_sql(
        text("SELECT * FROM enso_index ORDER BY year DESC, month DESC LIMIT :n"),
        engine, params={"n": limit})


def enso_outlook() -> pd.DataFrame:
    """Full forward ENSO probability outlook (all upcoming seasons), chronological."""
    return pd.read_sql(text("SELECT * FROM enso_outlook ORDER BY year, month"), engine)
