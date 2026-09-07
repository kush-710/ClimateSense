"""Human Impact Translation Layer: risk numbers -> plain-language guidance."""
from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reference.reference_tables import SPORT_LIMITS


def compute_safe_window(hourly: list[dict], threshold: int = 40) -> dict:
    """hourly: [{'hour': 6, 'risk_score': 22}, ...] for today (0-23)."""
    safe = sorted(h["hour"] for h in hourly if h.get("risk_score") is not None
                  and h["risk_score"] <= threshold)
    if not safe:
        return {"has_window": False, "message": "No safe outdoor window today"}
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


def sport_verdict(sport: str, temp_c, aqi, uv) -> dict:
    max_t, _max_wbgt, max_aqi, max_uv = SPORT_LIMITS.get(sport, SPORT_LIMITS["general"])
    breaches = []
    if temp_c is not None and temp_c > max_t:
        breaches.append(f"temperature {temp_c:.0f}C exceeds {max_t}C limit")
    if aqi is not None and aqi > max_aqi:
        breaches.append(f"AQI {aqi:.0f} exceeds {max_aqi} limit")
    if uv is not None and uv > max_uv:
        breaches.append(f"UV {uv:.0f} exceeds {max_uv} limit")
    return {"sport": sport, "playable": not breaches, "breaches": breaches}


def translate(risk_score: int, temp_c, aqi, uv, humidity, sport: str,
              city: str, hourly: list[dict]) -> dict:
    win = compute_safe_window(hourly)
    if risk_score >= 81:
        sev, action = "CRITICAL", "Suspend all outdoor sporting activity."
    elif risk_score >= 61:
        sev, action = "HIGH", f"Avoid {sport} during peak hours; morning window only."
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
        "sport_verdict": sport_verdict(sport, temp_c, aqi, uv),
        "hydration": hyd, "mask": mask,
        "uv_advice": ("SPF 50+ and eye protection required" if (uv or 0) >= 8
                      else "SPF 30 sufficient"),
        "summary": f"{city}: {sev} ({risk_score}/100). {action} {win['message']}.",
    }
