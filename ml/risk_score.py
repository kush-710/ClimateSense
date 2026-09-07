"""Composite Heat Risk Score (0-100), anchored to WBGT science and WHO AQG 2021."""
from __future__ import annotations
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.transform import compute_heat_index

TIERS = ["SAFE", "MODERATE", "HIGH", "CRITICAL"]


def compute_heat_risk_score(temp_c, humidity, aqi, uv) -> int | None:
    if temp_c is None or pd.isna(temp_c):
        return None
    humidity = 50 if humidity is None or pd.isna(humidity) else humidity
    aqi = 0 if aqi is None or pd.isna(aqi) else aqi
    uv = 0 if uv is None or pd.isna(uv) else uv

    hi = compute_heat_index(temp_c, humidity)
    hi_score   = max(0, min(100, (hi - 25) * 2.86))       # 25C -> 0, 60C -> 100
    temp_score = max(0, min(100, (temp_c - 20) * 3.33))   # 20C -> 0, 50C -> 100
    aqi_score  = max(0, min(100, aqi / 5))                # CPCB 0-500 -> 0-100
    uv_score   = max(0, min(100, uv / 11 * 100))
    composite = hi_score * 0.40 + aqi_score * 0.30 + temp_score * 0.20 + uv_score * 0.10
    return round(min(100, composite))


def tier_name(score: int) -> str:
    return TIERS[3 if score >= 81 else 2 if score >= 61 else 1 if score >= 31 else 0]
