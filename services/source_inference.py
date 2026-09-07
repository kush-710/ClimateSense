"""Pollution source-likelihood inference + government recommendation engine.

Method: start from the published seasonal apportionment prior for the city, then
re-weight with live meteorological/temporal evidence (hour of day, wind speed,
humidity, month) using transparent multiplicative modifiers. Output is a ranked
list of *likely dominant* sources with the interventions mapped to each, plus the
applicable GRAP stage. All heuristics are documented and cite their rationale.
"""
from __future__ import annotations
from datetime import datetime

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reference.reference_tables import (
    SOURCE_APPORTIONMENT, GENERIC_URBAN_PRIOR, INTERVENTIONS, GRAP_STAGES,
    season_of, aqi_category)
from config import GRAP_APPLICABLE_CITIES


def _modifiers(source: str, ts: datetime, wind_kph: float | None,
               humidity: float | None, pm25_to_pm10: float | None) -> float:
    """Multiplicative evidence weights. 1.0 = no adjustment."""
    w = 1.0
    hour, month = ts.hour, ts.month

    if source == "vehicular":
        if hour in (8, 9, 10, 18, 19, 20, 21):
            w *= 1.4                       # peak traffic hours
        if hour in (2, 3, 4):
            w *= 0.6
    if source == "stubble_burning":
        w *= 1.6 if month in (10, 11) else 0.1   # tightly seasonal
    if source in ("road_construction_dust",):
        if pm25_to_pm10 is not None and pm25_to_pm10 < 0.4:
            w *= 1.5                       # coarse-fraction dominance implies dust
        if humidity is not None and humidity > 80:
            w *= 0.6                       # wet surfaces suppress resuspension
    if source == "biomass_residential_burning":
        if hour in (22, 23, 0, 1, 2, 3, 4, 5) and month in (11, 12, 1):
            w *= 1.5                       # cold-night heating fires
    if source == "secondary_aerosols":
        if wind_kph is not None and wind_kph < 6:
            w *= 1.3                       # stagnation favours accumulation/formation
        if humidity is not None and humidity > 70:
            w *= 1.2                       # aqueous-phase formation
    return w


def infer_sources(city: str, ts: datetime, pm25: float | None, pm10: float | None,
                  wind_kph: float | None, humidity: float | None) -> list[dict]:
    season = season_of(ts.month)
    if city in SOURCE_APPORTIONMENT:
        prior = SOURCE_APPORTIONMENT[city].get(season, SOURCE_APPORTIONMENT[city]["winter"])
    else:
        # No published city-specific study encoded — use the generic national
        # prior rather than silently mislabeling another city's real citations.
        prior = GENERIC_URBAN_PRIOR
    refs = prior.get("refs", [])
    ratio = (pm25 / pm10) if pm25 and pm10 and pm10 > 0 else None

    scored = []
    for source, share in prior.items():
        if source == "refs":
            continue
        mid = (share[0] + share[1]) / 2.0
        weight = mid * _modifiers(source, ts, wind_kph, humidity, ratio)
        scored.append((source, weight, share))
    total = sum(w for _, w, _ in scored) or 1.0

    out = []
    for source, weight, share in sorted(scored, key=lambda x: -x[1]):
        iv = INTERVENTIONS.get(source, {})
        out.append({
            "source": source,
            "likelihood_pct": round(100 * weight / total, 1),
            "published_share_pct": f"{share[0]}-{share[1]}",
            "short_term_actions": iv.get("short_term", []),
            "long_term_actions": iv.get("long_term", []),
            "responsible_agencies": iv.get("responsible", []),
            "evidence": iv.get("evidence", ""),
        })
    if out:
        out[0]["apportionment_refs"] = refs
    return out


def grap_stage(aqi: float | None) -> dict | None:
    if aqi is None:
        return None
    for s in GRAP_STAGES:
        lo, hi = s["aqi_range"]
        if lo <= aqi <= hi:
            return s
    if aqi > 500:
        return GRAP_STAGES[-1]
    return None


def government_brief(city: str, ts: datetime, aqi: float | None, pm25: float | None,
                     pm10: float | None, wind_kph: float | None,
                     humidity: float | None) -> dict:
    """The deliverable for the policy audience: current stage + ranked sources +
    concrete interventions with responsible agencies."""
    sources = infer_sources(city, ts, pm25, pm10, wind_kph, humidity)
    grap_applicable = city in GRAP_APPLICABLE_CITIES
    return {
        "city": city,
        "as_of": ts.isoformat(),
        "aqi_cpcb": aqi,
        "aqi_category": aqi_category(aqi),   # national CPCB scale — applies everywhere
        "pm25": pm25,
        "pm10": pm10,
        "grap_applicable": grap_applicable,  # GRAP is Delhi-NCR-specific policy machinery
        "grap_stage": grap_stage(aqi) if grap_applicable else None,
        "grap_reference": GRAP_STAGES,
        "used_generic_source_prior": city not in SOURCE_APPORTIONMENT,
        "likely_dominant_sources": sources[:3],
        "all_sources": sources,
        "methodology_note": (
            "Source likelihoods combine published apportionment studies with live "
            "meteorological/temporal evidence via documented heuristics. This is "
            "decision-support, not real-time chemical apportionment."),
    }
