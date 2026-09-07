"""Async fetchers for every external source.

Sources (all zero-cost):
  - Open-Meteo forecast + archive  : weather, no key, unlimited
  - Open-Meteo air-quality API     : hourly PM2.5/PM10/NO2/O3/SO2/CO (CAMS model), no key
  - AQICN                          : live station AQI + pollutant readings (free token)
  - NOAA CPC ONI                   : El Nino / ENSO index (current/historical), plain-text, no key
  - NOAA CPC ENSO probabilities    : forward-looking El Nino/Neutral/La Nina probability
                                     forecast for the next 9 overlapping seasons, no key
"""
from __future__ import annotations
import asyncio
import logging
import re
from datetime import date

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OPEN_METEO_FORECAST, OPEN_METEO_ARCHIVE, OPEN_METEO_AQ, AQICN_FEED,
    NOAA_ONI_URL, NOAA_ENSO_OUTLOOK_URL, AQICN_TOKEN, OM_WEATHER_VARS, OM_AQ_VARS,
)

logger = logging.getLogger(__name__)
RETRY = dict(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=60))

# Shared by fetch_oni() and fetch_enso_outlook(): NOAA's 3-month running-season
# labels, mapped to the season's centre calendar month.
SEASON_CENTRE_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


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
    async with session.get(NOAA_ONI_URL, timeout=aiohttp.ClientTimeout(total=30)) as r:
        r.raise_for_status()
        text = await r.text()

    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in SEASON_CENTRE_MONTH:
            continue  # header or malformed line
        season, yr, _total, anom = parts
        oni = float(anom)
        phase = "el_nino" if oni >= 0.5 else "la_nina" if oni <= -0.5 else "neutral"
        rows.append({
            "year": int(yr), "month": SEASON_CENTRE_MONTH[season],
            "season": season, "oni": oni, "phase": phase,
        })
    if not rows:
        raise RuntimeError("ONI parse produced zero rows — file format may have changed")
    return rows


_ENSO_OUTLOOK_ISSUED_RE = re.compile(r"<h2>Issued (\w+) (\d{4})</h2>")
_ENSO_OUTLOOK_ROW_RE = re.compile(
    r'<abbr>([A-Z]{3}) <span class="tooltip[^"]*"[^>]*>[^<]*</span></abbr></th>'
    r"<td>(\d+)</td><td>(\d+)</td><td>(\d+)</td>"
)
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def parse_enso_outlook_html(html: str) -> list[dict]:
    """Pure parser for NOAA CPC's official probabilistic ENSO forecast page.

    No CSV/JSON is published for this product — the page renders an HTML
    table (id="probabilities-table") of 9 overlapping 3-month seasons, each
    with El Nino / Neutral / La Nina percentage chances. Separated from the
    network call so it's unit-testable against a fixture string.
    """
    issued = _ENSO_OUTLOOK_ISSUED_RE.search(html)
    if not issued:
        raise RuntimeError("CPC ENSO outlook parse failed — issuance header not found "
                           "(page format may have changed)")
    issued_month = _MONTH_NAMES.index(issued.group(1)) + 1
    issued_year = int(issued.group(2))

    rows = _ENSO_OUTLOOK_ROW_RE.findall(html)
    if not rows:
        raise RuntimeError("CPC ENSO outlook parse produced zero rows — "
                           "page format may have changed")

    out = []
    for season, el_nino_pct, neutral_pct, la_nina_pct in rows:
        centre_month = SEASON_CENTRE_MONTH[season]
        year = issued_year + (1 if centre_month < issued_month else 0)
        out.append({
            "year": year, "month": centre_month, "season": season,
            "el_nino_pct": int(el_nino_pct), "neutral_pct": int(neutral_pct),
            "la_nina_pct": int(la_nina_pct),
            "issued_year": issued_year, "issued_month": issued_month,
        })
    return out


@retry(**RETRY)
async def fetch_enso_outlook(session: aiohttp.ClientSession) -> list[dict]:
    """Official NOAA CPC/IRI consensus probabilistic ENSO forecast — a genuine
    forward-looking outlook (unlike the ONI file, which is current/historical
    only), updated roughly monthly."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ClimateSense/1.0)"}
    async with session.get(NOAA_ENSO_OUTLOOK_URL, headers=headers,
                           timeout=aiohttp.ClientTimeout(total=30)) as r:
        r.raise_for_status()
        html = await r.text()
    return parse_enso_outlook_html(html)


async def fetch_all_live(cities: dict) -> dict:
    """One shot: forecast weather + live AQ for every city, plus ONI. Runs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = {"oni": fetch_oni(session), "enso_outlook": fetch_enso_outlook(session)}
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
