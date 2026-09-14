"""Human Impact Translation Layer: risk numbers -> plain-language guidance."""
from __future__ import annotations

import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reference.reference_tables import SPORT_LIMITS
from config import IGP_CITIES


def enso_narrative(phase: str, month: int, city: str | None = None) -> str:
    """Plain-language explanation of what the current ENSO phase means for
    India right now, mirroring the oni_x_monsoon / oni_x_winter interaction
    terms the ML models are actually trained on (see ml/features.py). The
    winter clause is IGP-specific (Indo-Gangetic-plain inversion/PM2.5) -
    only stated for cities where that's actually the physical mechanism."""
    is_monsoon = month in (6, 7, 8, 9)
    is_winter = month in (11, 12, 1)
    is_igp = city in IGP_CITIES if city else False

    if phase == "el_nino":
        if is_monsoon:
            return ("El Nino is active during the monsoon - historically linked to a "
                    "weaker, rainfall-deficient monsoon and hotter summers across India.")
        if is_winter:
            if is_igp:
                return ("El Nino is active in winter - historically linked to weaker "
                        "ventilation over the Indo-Gangetic plain, worsening PM2.5 "
                        "accumulation episodes here.")
            return ("El Nino is active in winter. Its clearest winter effect in India is "
                    "worsened PM2.5 accumulation over the Indo-Gangetic plain (Delhi, "
                    "Lucknow) from weaker ventilation - a less direct driver here.")
        return ("El Nino is active. Its clearest effects in India show up during the monsoon "
                "(rainfall deficit) and winter (pollution accumulation in the north).")
    if phase == "la_nina":
        if is_monsoon:
            return ("La Nina is active during the monsoon - historically linked to a "
                    "stronger-than-average monsoon and above-normal rainfall.")
        if is_winter:
            return ("La Nina is active in winter - winter ventilation patterns tend to be "
                    "less disrupted than in an El Nino winter.")
        return "La Nina is active, generally associated with a wetter monsoon outlook."
    return "ENSO is neutral - neither El Nino nor La Nina is meaningfully influencing conditions right now."


DAYLIGHT_START, DAYLIGHT_END = 5, 19   # 05:00-19:00 IST, usable outdoor hours


def compute_safe_window(hourly: list[dict], threshold: int = 40,
                        daylight_only: bool = True) -> dict:
    """hourly: [{'hour': 6, 'risk_score': 22}, ...] for today (0-23).

    Restricted to daylight by default: risk scores are lowest at 2-4am, so an
    unbounded search kept returning windows nobody can train or hold a PE
    class in. India spans one timezone, so a fixed 05:00-19:00 band is a
    reasonable stand-in for sunrise/sunset across all eight cities.
    """
    candidates = [h for h in hourly
                  if not daylight_only or DAYLIGHT_START <= h["hour"] <= DAYLIGHT_END]
    safe = sorted(h["hour"] for h in candidates if h.get("risk_score") is not None
                  and h["risk_score"] <= threshold)
    if not safe:
        return {"has_window": False,
                "message": "No safe outdoor window during daylight today"
                           if daylight_only else "No safe outdoor window today"}
    # longest contiguous run
    best_start = best_len = cur_start = cur_len = None
    for h in safe:
        if cur_start is None or h != prev + 1:  # noqa: F821 (prev set below)
            cur_start, cur_len = h, 1
        else:
            cur_len += 1
        if best_len is None or cur_len > best_len:
            best_start, best_len = cur_start, cur_len
        prev = h
    end = best_start + best_len
    return {"has_window": True, "start": f"{best_start:02d}:00", "end": f"{end:02d}:00",
            "message": f"Safe window: {best_start:02d}:00-{end:02d}:00"}


def sport_verdict(sport: str, temp_c, aqi, uv, wbgt=None) -> dict:
    """WBGT is the threshold sports-medicine guidance is actually written
    against, so it's checked first and reported alongside the others."""
    max_t, max_wbgt, max_aqi, max_uv = SPORT_LIMITS.get(sport, SPORT_LIMITS["general"])
    breaches = []
    if wbgt is not None and not pd.isna(wbgt) and wbgt > max_wbgt:
        breaches.append(f"WBGT {wbgt:.1f}C exceeds {max_wbgt}C limit")
    if temp_c is not None and temp_c > max_t:
        breaches.append(f"temperature {temp_c:.0f}C exceeds {max_t}C limit")
    if aqi is not None and aqi > max_aqi:
        breaches.append(f"AQI {aqi:.0f} exceeds {max_aqi} limit")
    if uv is not None and uv > max_uv:
        breaches.append(f"UV {uv:.0f} exceeds {max_uv} limit")
    return {
        "sport": sport, "playable": not breaches, "breaches": breaches,
        "limits": {"max_temp_c": max_t, "max_wbgt_c": max_wbgt,
                   "max_aqi": max_aqi, "max_uv": max_uv},
    }


def translate(risk_score: int, temp_c, aqi, uv, humidity, sport: str,
              city: str, hourly: list[dict], wbgt=None) -> dict:
    win = compute_safe_window(hourly)
    air_driven = (aqi or 0) >= 201   # score was floored by air quality, not heat
    if risk_score >= 81:
        sev = "CRITICAL"
        action = ("Air quality is hazardous - suspend all outdoor activity."
                  if air_driven else "Suspend all outdoor sporting activity.")
    elif risk_score >= 61:
        sev = "HIGH"
        action = (f"Air quality is poor - move {sport} indoors or postpone."
                  if air_driven else f"Avoid {sport} during peak hours; morning window only.")
    elif risk_score >= 31:
        sev, action = "MODERATE", "Reduce session intensity ~30% and monitor conditions."
    else:
        sev, action = "SAFE", f"Conditions clear. Full {sport} activity permitted."

    if temp_c is not None and (temp_c >= 38 or (humidity or 0) >= 80):
        hyd = "250 ml every 15 min; electrolytes mandatory beyond 60 min."
    elif temp_c is not None and temp_c >= 33:
        hyd = "200 ml every 20 min; monitor urine colour."
    else:
        hyd = "Normal hydration: 200 ml every 30 min."

    mask = ("N95 mandatory" if (aqi or 0) > 200
            else "Surgical mask recommended" if (aqi or 0) > 150 else "None required")

    return {
        "city": city, "severity": sev, "risk_score": risk_score,
        "primary_action": action, "safe_window": win,
        "sport_verdict": sport_verdict(sport, temp_c, aqi, uv, wbgt),
        "hydration": hyd, "mask": mask,
        "uv_advice": ("SPF 50+ and eye protection required" if (uv or 0) >= 8
                      else "SPF 30 sufficient"),
        "summary": f"{city}: {sev} ({risk_score}/100). {action} {win['message']}.",
    }
