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


# ----------------------------- WBGT ----------------------------------------
def wet_bulb_temp(temp_c: float, humidity: float) -> float | None:
    """Psychrometric wet-bulb temperature, Stull (2011) single-equation fit.

    Valid roughly -20..50 C and 5..99 % RH, which covers every Indian city
    condition we ingest.
    """
    if temp_c is None or humidity is None or pd.isna(temp_c) or pd.isna(humidity):
        return None
    rh = min(max(float(humidity), 1.0), 100.0)
    T = float(temp_c)
    return (T * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
            + np.arctan(T + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def globe_temp(temp_c: float, humidity: float, solar_wm2: float | None) -> float | None:
    """Black-globe temperature estimate, Hunter & Minyard (1999) regression.

        Tg = 0.01498*S + 1.184*Ta - 0.0789*RH - 2.739

    S is shortwave radiation (W/m2); at night S=0 and the fit correctly
    returns Tg below air temperature (radiative cooling to the sky).
    """
    if temp_c is None or humidity is None or pd.isna(temp_c) or pd.isna(humidity):
        return None
    s = 0.0 if solar_wm2 is None or pd.isna(solar_wm2) else float(solar_wm2)
    return 0.01498 * s + 1.184 * float(temp_c) - 0.0789 * float(humidity) - 2.739


def compute_wbgt(temp_c: float, humidity: float, solar_wm2: float | None = None) -> float | None:
    """Outdoor Wet Bulb Globe Temperature (C) - the heat-stress index that
    ACSM/FIFA/military work-rest guidance is actually written against, which
    is why SPORT_LIMITS carries a max_wbgt per sport.

        WBGT = 0.7*Tnwb + 0.2*Tg + 0.1*Ta

    Estimated, not instrument-grade: natural wet-bulb is approximated by the
    psychrometric wet-bulb (Stull) and the globe term by the Hunter-Minyard
    regression, because a true Liljegren solve needs direct/diffuse split and
    a measured globe. Good enough to rank hours and flag unsafe sessions.
    """
    tw = wet_bulb_temp(temp_c, humidity)
    tg = globe_temp(temp_c, humidity, solar_wm2)
    if tw is None or tg is None:
        return None
    return round(0.7 * tw + 0.2 * tg + 0.1 * float(temp_c), 2)


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
            "shortwave_radiation": "solar_wm2",
        })
        if "solar_wm2" not in df:
            df["solar_wm2"] = np.nan
        df["heat_index_c"] = df.apply(
            lambda r: compute_heat_index(r["temp_c"], r["humidity_pct"]), axis=1)
        df["wbgt_c"] = df.apply(
            lambda r: compute_wbgt(r["temp_c"], r["humidity_pct"], r.get("solar_wm2")), axis=1)
    else:
        df = df.rename(columns={
            "pm2_5": "pm25", "nitrogen_dioxide": "no2", "ozone": "o3",
            "sulphur_dioxide": "so2", "carbon_monoxide": "co",
        })
        if "co" in df:
            df["co"] = df["co"] / 1000.0  # Open-Meteo ug/m3 -> CPCB mg/m3
        # aqi_cpcb is deliberately NOT computed here - Open-Meteo's AQ (CAMS)
        # feed frequently has None pollutants in its most recent hours (the
        # model run hasn't finished ingesting them yet); clean_air_quality()
        # gap-fills those first, so aqi_cpcb is computed there instead, after
        # filling - otherwise "now" almost always shows a null AQI.
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["temp_c"].between(-10, 60, inclusive="both") | df["temp_c"].isna()]
    df = df[df["humidity_pct"].between(0, 100) | df["humidity_pct"].isna()]
    df = df.sort_values("ts")
    for col in ("temp_c", "humidity_pct", "wind_kph", "uv_index", "solar_wm2", "wbgt_c"):
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


# US EPA AQI breakpoints, used to invert AQICN's sub-index values back into
# concentrations: (aqi_lo, aqi_hi, conc_lo, conc_hi).
_EPA_BREAKPOINTS = {
    "pm25": [(0, 50, 0.0, 12.0), (51, 100, 12.1, 35.4), (101, 150, 35.5, 55.4),
             (151, 200, 55.5, 150.4), (201, 300, 150.5, 250.4),
             (301, 400, 250.5, 350.4), (401, 500, 350.5, 500.4)],
    "pm10": [(0, 50, 0, 54), (51, 100, 55, 154), (101, 150, 155, 254),
             (151, 200, 255, 354), (201, 300, 355, 424),
             (301, 400, 425, 504), (401, 500, 505, 604)],
}


def epa_subindex_to_concentration(pollutant: str, subindex: float | None) -> float | None:
    """Invert a US EPA sub-index back to a concentration in ug/m3.

    AQICN's `iaqi.<pollutant>.v` fields are EPA *sub-index* values, NOT raw
    concentrations - the giveaway is that its reported pm10 sub-index can sit
    below its pm25 sub-index, which is physically impossible for
    concentrations (PM10 includes PM2.5 by definition). Feeding those numbers
    straight into CPCB's concentration breakpoints overstates AQI ~3-4x.
    """
    if subindex is None or pd.isna(subindex) or pollutant not in _EPA_BREAKPOINTS:
        return None
    for alo, ahi, clo, chi in _EPA_BREAKPOINTS[pollutant]:
        if alo <= subindex <= ahi:
            return round(clo + (subindex - alo) * (chi - clo) / (ahi - alo), 1)
    return None  # above the top of the scale - don't extrapolate


def aqicn_to_df(payload: dict, city_id: int) -> pd.DataFrame:
    """AQICN ground-station reading -> one-row frame matching the air_quality
    schema, source='aqicn'. Station data is real measurement (vs Open-Meteo's
    CAMS model estimate) and usually much fresher, so it's the better source
    for 'current AQI' when a free AQICN token is configured.

    AQICN reports EPA sub-indices, so they're converted back to concentrations
    first; aqi_cpcb is then computed from those with India's CPCB breakpoints,
    keeping every stored AQI on one consistent scale regardless of source.
    """
    d = payload["data"]
    iaqi = d.get("iaqi", {})
    row = {
        "city_id": city_id,
        "ts": pd.to_datetime(d["time"]["s"]),
        "pm25": epa_subindex_to_concentration("pm25", iaqi.get("pm25", {}).get("v")),
        "pm10": epa_subindex_to_concentration("pm10", iaqi.get("pm10", {}).get("v")),
        "source": "aqicn",
    }
    row["aqi_cpcb"] = compute_cpcb_aqi(row)
    return pd.DataFrame([row])
