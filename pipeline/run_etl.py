"""Entry points.

  python pipeline/run_etl.py live                 # 15-min cadence job (GitHub Actions cron)
  python pipeline/run_etl.py seed --days 730      # historical backfill for model training
"""
from __future__ import annotations
import argparse
import asyncio
import logging
from datetime import date, timedelta

import aiohttp
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES
from pipeline.fetch import (fetch_all_live, fetch_weather_archive,
                            fetch_air_quality, fetch_oni)
from pipeline.transform import openmeteo_hourly_to_df, clean_weather, clean_air_quality, aqicn_to_df
from pipeline.load import (init_db, upsert_df, replace_all,
                           WeatherData, AirQuality, EnsoIndex, EnsoOutlook)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("etl")


def run_live() -> None:
    init_db()
    raw = asyncio.run(fetch_all_live(CITIES))

    if "oni" in raw:
        upsert_df(pd.DataFrame(raw["oni"]), EnsoIndex, ["year", "month"])
    if "enso_outlook" in raw:
        replace_all(pd.DataFrame(raw["enso_outlook"]), EnsoOutlook)

    for cid in CITIES:
        if f"wx_{cid}" in raw:
            wx = clean_weather(openmeteo_hourly_to_df(raw[f"wx_{cid}"], cid, "weather"))
            upsert_df(wx, WeatherData, ["city_id", "ts"])
        if f"aq_{cid}" in raw:
            aq = clean_air_quality(openmeteo_hourly_to_df(raw[f"aq_{cid}"], cid, "aq"))
            upsert_df(aq, AirQuality, ["city_id", "ts", "source"])
        if f"aqicn_{cid}" in raw:
            upsert_df(aqicn_to_df(raw[f"aqicn_{cid}"], cid), AirQuality, ["city_id", "ts", "source"])
    logger.info("live ETL complete")


async def _seed(days: int) -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)
    async with aiohttp.ClientSession() as session:
        oni = await fetch_oni(session)
        upsert_df(pd.DataFrame(oni), EnsoIndex, ["year", "month"])
        for cid, c in CITIES.items():
            logger.info("seeding %s %s -> %s", c["name"], start, end)
            wx_raw = await fetch_weather_archive(session, c["lat"], c["lon"], start, end)
            wx = clean_weather(openmeteo_hourly_to_df(wx_raw, cid, "weather"))
            upsert_df(wx, WeatherData, ["city_id", "ts"])
            aq_raw = await fetch_air_quality(session, c["lat"], c["lon"], start, end)
            aq = clean_air_quality(openmeteo_hourly_to_df(aq_raw, cid, "aq"))
            upsert_df(aq, AirQuality, ["city_id", "ts", "source"])


def run_seed(days: int) -> None:
    init_db()
    asyncio.run(_seed(days))
    logger.info("historical seed complete (%d days)", days)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["live", "seed"])
    p.add_argument("--days", type=int, default=730)
    args = p.parse_args()
    run_live() if args.mode == "live" else run_seed(args.days)
