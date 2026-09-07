"""LLM-grounded conversational assistant.

The model (Gemini, free tier) is never allowed to invent numbers: every
answer is generated from a JSON context block built entirely out of this
app's own live data (current risk, forecast, policy brief, ENSO state). If
the context doesn't cover the question, the prompt instructs the model to
say so rather than guess.
"""
from __future__ import annotations

import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, GEMINI_API_KEY, GEMINI_MODEL
from services.data_access import latest_observation, today_hourly_scores, latest_enso, enso_outlook
from services.translation import translate, enso_narrative
from services.source_inference import government_brief
from ml.risk_score import compute_heat_risk_score

# City-name aliases for detecting which city a free-text question is about.
_CITY_ALIASES = {
    1: ["delhi", "new delhi", "ncr"],
    2: ["bengaluru", "bangalore"],
    3: ["mumbai", "bombay"],
    4: ["chennai", "madras"],
    5: ["kolkata", "calcutta"],
    6: ["hyderabad"],
    7: ["pune"],
    8: ["lucknow"],
}


def detect_city(question: str) -> int | None:
    """Lowercase substring match against known city aliases. If more than
    one city is named, the one appearing earliest in the text wins."""
    q = question.lower()
    hits = [(q.index(alias), cid) for cid, aliases in _CITY_ALIASES.items()
            for alias in aliases if alias in q]
    return min(hits)[1] if hits else None


def _clean(v):
    return None if v is None or pd.isna(v) else float(v)


def _build_context(city_id: int) -> dict:
    city_name = CITIES[city_id]["name"]
    latest = latest_observation(city_id)
    score = compute_heat_risk_score(latest["temp_c"], latest["humidity_pct"],
                                    latest["aqi_cpcb"], latest["uv_index"])
    guidance = translate(score or 0, latest["temp_c"], latest["aqi_cpcb"],
                         latest["uv_index"], latest["humidity_pct"], "general",
                         city_name, today_hourly_scores(city_id))
    brief = government_brief(
        city_name, pd.Timestamp(latest["ts"]).to_pydatetime(),
        _clean(latest["aqi_cpcb"]), _clean(latest["pm25"]), _clean(latest["pm10"]),
        _clean(latest["wind_kph"]), _clean(latest["humidity_pct"]))

    context = {
        "city": city_name,
        "observed_at": str(latest["ts"]),
        "current_conditions": {
            "temp_c": _clean(latest["temp_c"]),
            "humidity_pct": _clean(latest["humidity_pct"]),
            "wind_kph": _clean(latest["wind_kph"]),
            "uv_index": _clean(latest["uv_index"]),
            "aqi_cpcb": _clean(latest["aqi_cpcb"]),
            "pm25": _clean(latest["pm25"]),
            "pm10": _clean(latest["pm10"]),
        },
        "risk": {"score": score, "severity": guidance["severity"],
                 "summary": guidance["summary"], "safe_window": guidance["safe_window"],
                 "hydration": guidance["hydration"], "mask": guidance["mask"]},
        "policy_brief": {
            "aqi_category": brief["aqi_category"],
            "grap_applicable": brief["grap_applicable"],
            "grap_stage": brief["grap_stage"],
            "likely_dominant_sources": brief["likely_dominant_sources"],
            "used_generic_source_prior": brief["used_generic_source_prior"],
        },
    }

    try:
        from ml.train import forecast_prophet
        fc = forecast_prophet(city_id, "temp_c", 5)
        context["temp_forecast_5day"] = [
            {"date": str(r.ds.date()), "value": round(r.yhat, 2)} for r in fc.itertuples()]
    except FileNotFoundError:
        pass  # model not trained yet for this city — assistant still works without it

    enso_df = latest_enso(limit=1)
    if not enso_df.empty:
        e = enso_df.iloc[0]
        context["enso"] = {
            "season": e["season"], "year": int(e["year"]),
            "oni": float(e["oni"]), "phase": e["phase"],
            "effect_right_now": enso_narrative(e["phase"], int(e["month"]), city_name),
            "how_the_forecast_uses_it": (
                "The current ONI value is fed as a live regressor into the Prophet "
                "temperature and PM2.5 forecast models above, and as engineered features "
                "(oni, oni_lag_3m, enso_phase, and interaction terms with monsoon/winter "
                "season) into the XGBoost risk-tier classifier behind the current risk score."
            ),
        }

    outlook_df = enso_outlook()
    if not outlook_df.empty:
        # Real NOAA CPC/IRI probabilistic forecast — genuinely forward-looking,
        # unlike the ONI reading above which is current/historical only.
        context["enso_forecast_outlook"] = outlook_df[
            ["season", "year", "el_nino_pct", "neutral_pct", "la_nina_pct"]
        ].to_dict(orient="records")

    return context


def _build_prompt(question: str, city_name: str, context: dict) -> str:
    import json
    return (
        "You are ClimateSense's assistant, answering questions about heat and air-quality "
        f"risk in {city_name}, India. Use ONLY the JSON data below to answer — do not invent "
        "numbers or facts that aren't in it. If the data doesn't cover what's being asked, "
        "say so plainly instead of guessing. For 'how do we fix/reduce this' questions, base "
        "your answer on policy_brief.likely_dominant_sources' short_term_actions / "
        "long_term_actions / responsible_agencies (if policy_brief.used_generic_source_prior "
        "is true, mention these are generic indicative shares, not a city-specific study). "
        "For the current El Niño/ENSO state, use enso.effect_right_now. For questions about "
        "the FUTURE ENSO trend (e.g. 'will El Nino continue', 'what's expected next season'), "
        "use enso_forecast_outlook — this is a real NOAA CPC/IRI probabilistic forecast, not a "
        "guess; cite the actual percentages and season. If a policy_brief.grap_stage is null "
        "and grap_applicable is false, say GRAP is a Delhi-NCR-specific framework that doesn't "
        "apply here, and use aqi_category instead. Answer in plain language, no markdown — "
        "2-4 sentences normally, up to 6 if listing concrete policy actions or probabilities.\n\n"
        f"DATA:\n{json.dumps(context, default=str)}\n\n"
        f"QUESTION: {question}"
    )


def answer_question(question: str, default_city_id: int = 1) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    detected = detect_city(question)
    city_id = detected or default_city_id
    context = _build_context(city_id)
    prompt = _build_prompt(question, CITIES[city_id]["name"], context)

    from google import genai
    from google.genai import errors as genai_errors
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except genai_errors.APIError as e:
        raise RuntimeError(f"Gemini API error: {e}") from e

    return {
        "answer": resp.text,
        "city": CITIES[city_id]["name"],
        "city_detected": detected is not None,
        "grounded_on": context,
    }
