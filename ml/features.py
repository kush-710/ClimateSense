"""Feature engineering for the risk classifier and forecasters.

El Nino integration
-------------------
ONI (Oceanic Nino Index, NOAA CPC) is a monthly signal; hourly rows are joined to
the ONI of their calendar month. Physical rationale for each ENSO feature:

  oni                : current SST anomaly. El Nino summers in India run hotter
                       (weakened monsoon, reduced cloud cover) — direct heat-risk driver.
  oni_lag_3m         : ENSO teleconnection to Indian monsoon acts with a lag; a
                       developing El Nino in spring predicts a deficient monsoon.
  enso_phase (0/1/2) : neutral / el_nino / la_nina categorical.
  oni_x_monsoon      : ONI * is_monsoon_season interaction — the deficit-rainfall
                       effect only expresses June-September.
  oni_x_winter       : ONI * is_winter interaction — ENSO modulates winter
                       ventilation/inversion strength over the Indo-Gangetic plain,
                       which drives PM2.5 accumulation episodes in Delhi.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.load import engine

PHASE_CODE = {"neutral": 0, "el_nino": 1, "la_nina": 2}

FEATURES = [
    # raw / derived weather
    "temp_c", "humidity_pct", "heat_index_c", "wind_kph", "uv_index", "pressure_hpa",
    # pollution
    "pm25", "pm10", "no2", "o3", "aqi_cpcb",
    # temporal (cyclic)
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend",
    # lags / rolling
    "temp_lag_1h", "pm25_lag_3h", "temp_roll_6h_mean", "pm25_roll_6h_max",
    # ENSO / El Nino block
    "oni", "oni_lag_3m", "enso_phase", "oni_x_monsoon", "oni_x_winter",
]


def load_joined(city_id: int) -> pd.DataFrame:
    """Weather + AQ + ENSO joined on the hour."""
    wx = pd.read_sql(text("SELECT * FROM weather_data WHERE city_id=:c ORDER BY ts"),
                     engine, params={"c": city_id}, parse_dates=["ts"])
    aq = pd.read_sql(text("SELECT * FROM air_quality WHERE city_id=:c ORDER BY ts"),
                     engine, params={"c": city_id}, parse_dates=["ts"])
    enso = pd.read_sql(text("SELECT year, month, oni, phase FROM enso_index"), engine)

    df = wx.merge(aq.drop(columns=["id", "city_id", "source"], errors="ignore"),
                  on="ts", how="inner")
    df["year"], df["month"] = df["ts"].dt.year, df["ts"].dt.month
    df = df.merge(enso, on=["year", "month"], how="left")

    # ONI 3-month lag: join against enso shifted by 3 months
    enso_l = enso.copy()
    shifted = pd.to_datetime(dict(year=enso_l.year, month=enso_l.month, day=1)) \
                + pd.DateOffset(months=3)
    enso_l["year"], enso_l["month"] = shifted.dt.year, shifted.dt.month
    enso_l = enso_l.rename(columns={"oni": "oni_lag_3m"})[["year", "month", "oni_lag_3m"]]
    df = df.merge(enso_l, on=["year", "month"], how="left")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    h, m = df["ts"].dt.hour, df["ts"].dt.month
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)
    df["is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(int)

    df["temp_lag_1h"] = df["temp_c"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["temp_roll_6h_mean"] = df["temp_c"].rolling(6, min_periods=3).mean()
    df["pm25_roll_6h_max"] = df["pm25"].rolling(6, min_periods=3).max()

    # --- ENSO block ---
    df["oni"] = df["oni"].ffill()
    df["oni_lag_3m"] = df["oni_lag_3m"].ffill()
    df["enso_phase"] = df["phase"].map(PHASE_CODE).ffill().fillna(0).astype(int)
    is_monsoon = m.isin([6, 7, 8, 9]).astype(int)
    is_winter = m.isin([11, 12, 1]).astype(int)
    df["oni_x_monsoon"] = df["oni"] * is_monsoon
    df["oni_x_winter"] = df["oni"] * is_winter
    return df


def label_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Supervision labels from the WBGT/WHO-anchored composite score (see ml/risk_score)."""
    from ml.risk_score import compute_heat_risk_score
    df["risk_tier"] = df.apply(
        lambda r: _tier(compute_heat_risk_score(
            r.get("temp_c"), r.get("humidity_pct"), r.get("aqi_cpcb"), r.get("uv_index"))),
        axis=1)
    return df


def _tier(score: int | None) -> int:
    if score is None:
        return 0
    return 3 if score >= 81 else 2 if score >= 61 else 1 if score >= 31 else 0


def training_frame(city_id: int) -> pd.DataFrame:
    df = label_risk_tier(add_features(load_joined(city_id)))
    return df.dropna(subset=[f for f in FEATURES if f in df.columns] + ["risk_tier"])
