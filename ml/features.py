"""Feature engineering for the risk classifier and forecasters.

El Nino integration
-------------------
ONI (Oceanic Nino Index, NOAA CPC) is a monthly signal; hourly rows are joined to
the ONI of their calendar month. Physical rationale for each ENSO feature:

  oni                : current SST anomaly. El Nino summers in India run hotter
                       (weakened monsoon, reduced cloud cover) - direct heat-risk driver.
  oni_lag_3m         : ENSO teleconnection to Indian monsoon acts with a lag; a
                       developing El Nino in spring predicts a deficient monsoon.
  enso_phase (0/1/2) : neutral / el_nino / la_nina categorical.
  oni_x_monsoon      : ONI * is_monsoon_season interaction - the deficit-rainfall
                       effect only expresses June-September.
  oni_x_winter       : ONI * is_winter interaction - ENSO modulates winter
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

# How far ahead the risk classifier predicts. The score itself is a closed
# form of the current hour's readings, so training on the *same* hour just
# teaches the model to re-derive arithmetic we already have exactly. Shifting
# the label forward makes it a real forecasting task the formula can't do.
FORECAST_HORIZON_H = 24

FEATURES = [
    # raw / derived weather
    "temp_c", "humidity_pct", "heat_index_c", "wbgt_c", "wind_kph",
    "solar_wm2", "pressure_hpa",
    # pollution
    "pm25", "pm10", "no2", "o3", "aqi_cpcb",
    # temporal (cyclic)
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend",
    # lags / rolling
    "temp_lag_1h", "pm25_lag_3h", "temp_roll_6h_mean", "pm25_roll_6h_max",
    # ENSO / El Nino block
    "oni", "oni_lag_3m", "enso_phase", "oni_x_monsoon", "oni_x_winter",
    # Today's tier is the persistence baseline ("tomorrow looks like today").
    # Handing it to the model as a feature means it starts from that baseline
    # and learns corrections, rather than having to rediscover it. Uses only
    # current-hour readings, so there's no leakage from the future label.
    "risk_tier_now",
]
# NOTE: uv_index is deliberately NOT a feature. Open-Meteo's archive endpoint
# returns it as all-null, so every historical row would train on a constant
# fill while live rows carry real values - a train/serve skew that makes the
# feature worse than useless. solar_wm2 (shortwave radiation) is the real
# solar signal and IS available historically.


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

    # Solar radiation is genuinely 0 overnight; a null here means a gap, not
    # darkness, but 0 is the safe fill for the handful of missing hours.
    if "solar_wm2" in df:
        df["solar_wm2"] = df["solar_wm2"].fillna(0)

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

    # Current-hour risk tier: the persistence baseline, used both as a model
    # feature and (shifted forward) as the training label. Computed here so
    # serving builds it exactly the same way training does.
    from ml.risk_score import compute_heat_risk_score
    df["risk_tier_now"] = df.apply(
        lambda r: _tier(compute_heat_risk_score(
            r.get("temp_c"), r.get("humidity_pct"), r.get("aqi_cpcb"),
            r.get("uv_index"), r.get("solar_wm2"))),
        axis=1)
    return df


def label_risk_tier(df: pd.DataFrame, horizon_h: int = FORECAST_HORIZON_H) -> pd.DataFrame:
    """Label each row with the risk tier `horizon_h` hours LATER.

    Scoring the current hour would be circular - compute_heat_risk_score() is
    a closed form of columns the model already receives as features, so the
    classifier would just relearn arithmetic (and score a meaninglessly high
    F1 doing it). Shifting the target forward asks a real question instead:
    given conditions now, what will the risk tier be tomorrow?
    """
    # risk_tier_now is built in add_features(); rows are hourly and ordered,
    # so a -horizon shift lines each row up with the tier that actually
    # occurred that many hours later. The final horizon_h rows have no future
    # to look at and drop out in training_frame().
    df["risk_tier"] = df["risk_tier_now"].shift(-horizon_h)
    return df


def _tier(score: int | None) -> int:
    if score is None:
        return 0
    return 3 if score >= 81 else 2 if score >= 61 else 1 if score >= 31 else 0


def training_frame(city_id: int) -> pd.DataFrame:
    df = label_risk_tier(add_features(load_joined(city_id)))
    df = df.dropna(subset=[f for f in FEATURES if f in df.columns] + ["risk_tier"])
    df["risk_tier"] = df["risk_tier"].astype(int)
    return df
