"""Composite Heat Risk Score (0-100), anchored to WBGT science and WHO AQG 2021."""
from __future__ import annotations
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.transform import compute_wbgt

TIERS = ["SAFE", "MODERATE", "HIGH", "CRITICAL"]

# CPCB AQI at/above which air quality alone caps how "safe" we may report,
# regardless of how mild the heat is. CPCB 201+ is the "Poor" category.
AQI_POOR = 201
AQI_VERY_POOR = 301


def compute_heat_risk_score(temp_c, humidity, aqi, uv, solar_wm2=None) -> int | None:
    """Composite 0-100 risk score.

    Heat term is real WBGT (see pipeline.transform.compute_wbgt) rather than
    the Rothfusz heat index, so the score is anchored to the same index the
    per-sport thresholds in SPORT_LIMITS are written against. WBGT 18 C is
    the bottom of the sports-medicine "low risk" band and 33 C is past every
    published stop-play threshold, so that range maps onto 0-100.
    """
    if temp_c is None or pd.isna(temp_c):
        return None
    humidity = 50 if humidity is None or pd.isna(humidity) else humidity
    aqi = 0 if aqi is None or pd.isna(aqi) else aqi
    uv = 0 if uv is None or pd.isna(uv) else uv

    wbgt = compute_wbgt(temp_c, humidity, solar_wm2)
    wbgt_score = 0.0 if wbgt is None else max(0, min(100, (wbgt - 18) * (100 / 15)))
    temp_score = max(0, min(100, (temp_c - 20) * 3.33))   # 20C -> 0, 50C -> 100
    aqi_score  = max(0, min(100, aqi / 5))                # CPCB 0-500 -> 0-100
    uv_score   = max(0, min(100, uv / 11 * 100))
    composite = wbgt_score * 0.40 + aqi_score * 0.30 + temp_score * 0.20 + uv_score * 0.10

    # Air-quality floor: a cool night under "Poor"/"Very Poor" air is not
    # SAFE to exercise in, but a purely weighted blend would say it is
    # (AQI is only 30% of the score). Raise the floor into the matching tier.
    if aqi >= AQI_VERY_POOR:
        composite = max(composite, 81)   # CRITICAL band
    elif aqi >= AQI_POOR:
        composite = max(composite, 61)   # HIGH band
    return round(min(100, composite))


def tier_name(score: int) -> str:
    return TIERS[3 if score >= 81 else 2 if score >= 61 else 1 if score >= 31 else 0]
