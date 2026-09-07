"""Async fetchers for every external source.

Sources (all zero-cost):
  - Open-Meteo forecast + archive  : weather, no key, unlimited
  - Open-Meteo air-quality API     : hourly PM2.5/PM10/NO2/O3/SO2/CO (CAMS model), no key
  - AQICN                          : live station AQI + pollutant readings (free token)
  - NOAA CPC ONI                   : El Nino / ENSO index, plain-text file, no key
"""
from __future__ import annotations
import asyncio
import logging
from datetime import date

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OPEN_METEO_FORECAST, OPEN_METEO_ARCHIVE, OPEN_METEO_AQ, AQICN_FEED,
    NOAA_ONI_URL, AQICN_TOKEN, OM_WEATHER_VARS, OM_AQ_VARS,
)

logger = logging.getLogger(__name__)
RETRY = dict(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=60))


@retry(**RETRY)
async def fetch_weather_forecast(session: aiohttp.ClientSession, lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(OM_WEATHER_VARS),
        "forecast_days": 7, "timezone": "Asia/Kolkata",
    }
    async with session.get(OPEN_METEO_FORECAST, params=params,
                           timeout=aiohttp.ClientTimeout(total=20)) as r:
        r.raise_for_status()
        return await r.json()


@retry(**RETRY)
async def fetch_weather_archive(session: aiohttp.ClientSession, lat: float, lon: float,
                                start: date, end: date) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(OM_WEATHER_VARS), "timezone": "Asia/Kolkata",
    }
    async with session.get(OPEN_METEO_ARCHIVE, params=params,
                           timeout=aiohttp.ClientTimeout(total=60)) as r:
        r.raise_for_status()
        return await r.json()


@retry(**RETRY)
async def fetch_air_quality(session: aiohttp.ClientSession, lat: float, lon: float,
                            start: date | None = None, end: date | None = None) -> dict:
    """Hourly pollutant series. With start/end it returns history (2013+)."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(OM_AQ_VARS), "timezone": "Asia/Kolkata",
    }
    if start and end:
        params["start_date"], params["end_date"] = start.isoformat(), end.isoformat()
    async with session.get(OPEN_METEO_AQ, params=params,
                           timeout=aiohttp.ClientTimeout(total=60)) as r:
        r.raise_for_status()
        return await r.json()


@retry(**RETRY)
async def fetch_aqicn_live(session: aiohttp.ClientSession, slug: str) -> dict:
    """Live station reading — used for the dashboard 'now' card and cross-validation
    of the CAMS model values from Open-Meteo."""
    url = AQICN_FEED.format(slug=slug)
    async with session.get(url, params={"token": AQICN_TOKEN},
                           timeout=aiohttp.ClientTimeout(total=15)) as r:
        r.raise_for_status()
        data = await r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"AQICN error for {slug}: {data}")
        return data


@retry(**RETRY)
async def fetch_oni(session: aiohttp.ClientSession) -> list[dict]:
    """Parse NOAA CPC's ONI table.

    File format (whitespace-separated):
        SEAS  YR    TOTAL  ANOM
        DJF   1950  24.72  -1.53
        ...
    ANOM is the 3-month running-mean Nino 3.4 SST anomaly (the ONI itself).
    Phase convention: >= +0.5 El Nino, <= -0.5 La Nina, else neutral.
    """
    season_centre_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    async with session.get(NOAA_ONI_URL, timeout=aiohttp.ClientTimeout(total=30)) as r:
        r.raise_for_status()
        text = await r.text()

    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in season_centre_month:
            continue  # header or malformed line
        season, yr, _total, anom = parts
        oni = float(anom)
        phase = "el_nino" if oni >= 0.5 else "la_nina" if oni <= -0.5 else "neutral"
        rows.append({
            "year": int(yr), "month": season_centre_month[season],
            "season": season, "oni": oni, "phase": phase,
        })
    if not rows:
        raise RuntimeError("ONI parse produced zero rows — file format may have changed")
    return rows


async def fetch_all_live(cities: dict) -> dict:
    """One shot: forecast weather + live AQ for every city, plus ONI. Runs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = {"oni": fetch_oni(session)}
        for cid, c in cities.items():
            tasks[f"wx_{cid}"] = fetch_weather_forecast(session, c["lat"], c["lon"])
            tasks[f"aq_{cid}"] = fetch_air_quality(session, c["lat"], c["lon"])
            if AQICN_TOKEN:
                tasks[f"aqicn_{cid}"] = fetch_aqicn_live(session, c["aqicn_slug"])
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out = {}
        for key, res in zip(tasks.keys(), results):
            if isinstance(res, Exception):
                logger.error("fetch %s failed: %s", key, res)
            else:
                out[key] = res
        return out
