"""Cleaning, unit normalisation, CPCB AQI computation, derived metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------- Heat Index ----------------------------------
def compute_heat_index(temp_c: float, humidity: float) -> float:
    """Rothfusz (1990) regression of Steadman's heat index. Valid above 26.7C."""
    if pd.isna(temp_c) or pd.isna(humidity) or temp_c < 26.7:
        return temp_c
    T, H = temp_c, humidity
    hi = (-8.784695 + 1.61139411 * T + 2.338549 * H - 0.14611605 * T * H
          - 0.01230809 * T**2 - 0.01642482 * H**2 + 0.00221173 * T**2 * H
          + 0.00072546 * T * H**2 - 0.00000358 * T**2 * H**2)
    return round(hi, 2)


# ----------------------------- CPCB AQI ------------------------------------
# CPCB (2014) National AQI breakpoints: (conc_lo, conc_hi, aqi_lo, aqi_hi)
_CPCB_BREAKPOINTS = {
    "pm25": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
             (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
             (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
    "no2":  [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
             (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)],
    "o3":   [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
             (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
    "so2":  [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
             (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2000, 401, 500)],
    "co":   [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10, 101, 200),
             (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],   # mg/m3
}


def _sub_index(pollutant: str, conc: float) -> float | None:
    if conc is None or pd.isna(conc):
        return None
    for lo, hi, alo, ahi in _CPCB_BREAKPOINTS[pollutant]:
        if lo <= conc <= hi:
            return alo + (conc - lo) * (ahi - alo) / (hi - lo)
    return 500.0  # above the top breakpoint


def compute_cpcb_aqi(row: dict | pd.Series) -> float | None:
    """Composite CPCB AQI = max of pollutant sub-indices (requires >=1 of PM2.5/PM10)."""
    subs = {p: _sub_index(p, row.get(p)) for p in _CPCB_BREAKPOINTS}
    if subs.get("pm25") is None and subs.get("pm10") is None:
        return None
    vals = [v for v in subs.values() if v is not None]
    return round(max(vals), 1) if vals else None


# ----------------------------- Frame builders -------------------------------
def openmeteo_hourly_to_df(payload: dict, city_id: int, kind: str) -> pd.DataFrame:
    """kind: 'weather' | 'aq'. Converts an Open-Meteo hourly payload to a tidy frame."""
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["ts"] = pd.to_datetime(df.pop("time"))
    df["city_id"] = city_id

    if kind == "weather":
        df = df.rename(columns={
            "temperature_2m": "temp_c", "relative_humidity_2m": "humidity_pct",
            "wind_speed_10m": "wind_kph", "wind_direction_10m": "wind_dir_deg",
            "precipitation": "rainfall_mm", "surface_pressure": "pressure_hpa",
        })
        df["heat_index_c"] = df.apply(
            lambda r: compute_heat_index(r["temp_c"], r["humidity_pct"]), axis=1)
    else:
        df = df.rename(columns={
            "pm2_5": "pm25", "nitrogen_dioxide": "no2", "ozone": "o3",
            "sulphur_dioxide": "so2", "carbon_monoxide": "co",
        })
        if "co" in df:
            df["co"] = df["co"] / 1000.0  # Open-Meteo ug/m3 -> CPCB mg/m3
        # aqi_cpcb is deliberately NOT computed here — Open-Meteo's AQ (CAMS)
        # feed frequently has None pollutants in its most recent hours (the
        # model run hasn't finished ingesting them yet); clean_air_quality()
        # gap-fills those first, so aqi_cpcb is computed there instead, after
        # filling — otherwise "now" almost always shows a null AQI.
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["temp_c"].between(-10, 60, inclusive="both") | df["temp_c"].isna()]
    df = df[df["humidity_pct"].between(0, 100) | df["humidity_pct"].isna()]
    df = df.sort_values("ts")
    for col in ("temp_c", "humidity_pct", "wind_kph", "uv_index"):
        if col in df:
            df[col] = df[col].ffill(limit=8)          # <= 8h gap fill
    df.loc[df["ts"].dt.hour.isin(range(21, 24)) | df["ts"].dt.hour.isin(range(0, 5)),
           "uv_index"] = df["uv_index"].fillna(0)
    return df.drop_duplicates(subset=["city_id", "ts"]).reset_index(drop=True)


def clean_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts")
    for col in ("pm25", "pm10", "no2", "o3", "so2", "co"):
        if col in df:
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].ffill(limit=4)          # AQ: conservative 4h fill only
    df["aqi_cpcb"] = df.apply(compute_cpcb_aqi, axis=1)   # after fill, not before
    return df.drop_duplicates(subset=["city_id", "ts"]).reset_index(drop=True)


def aqicn_to_df(payload: dict, city_id: int) -> pd.DataFrame:
    """AQICN ground-station reading -> one-row frame matching the air_quality
    schema, source='aqicn'. Real-world station data is typically far fresher
    than Open-Meteo's CAMS model feed (which can lag 1-2 days), so this is
    the preferred 'current AQI' source when a free AQICN token is configured.

    We only take AQICN's raw pollutant concentrations (iaqi.*.v) and compute
    aqi_cpcb ourselves via compute_cpcb_aqi — AQICN's own top-level 'aqi'
    figure isn't guaranteed to be on the CPCB scale, and mixing index
    methodologies under one 'AQI (CPCB)' label would be misleading.
    """
    d = payload["data"]
    iaqi = d.get("iaqi", {})
    row = {
        "city_id": city_id,
        "ts": pd.to_datetime(d["time"]["s"]),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "source": "aqicn",
    }
    row["aqi_cpcb"] = compute_cpcb_aqi(row)
    return pd.DataFrame([row])
