"""Unit tests for the pure-logic layers (no network, no DB)."""
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.transform import compute_heat_index, compute_cpcb_aqi
from ml.risk_score import compute_heat_risk_score, tier_name
from services.translation import compute_safe_window, translate, enso_narrative
from services.source_inference import infer_sources, grap_stage, government_brief
from pipeline.fetch import parse_enso_outlook_html


def test_heat_index_below_threshold_passthrough():
    assert compute_heat_index(25.0, 90) == 25.0


def test_heat_index_hot_humid_amplifies():
    assert compute_heat_index(40.0, 70) > 40.0


def test_cpcb_aqi_pm25_severe():
    aqi = compute_cpcb_aqi({"pm25": 250, "pm10": None, "no2": None,
                            "o3": None, "so2": None, "co": None})
    assert 395 <= aqi <= 405   # 250 ug/m3 sits at the 400/401 breakpoint


def test_cpcb_aqi_requires_pm():
    assert compute_cpcb_aqi({"pm25": None, "pm10": None, "no2": 100,
                             "o3": None, "so2": None, "co": None}) is None


def test_risk_score_extremes():
    assert compute_heat_risk_score(45, 80, 350, 10) >= 81
    assert compute_heat_risk_score(22, 40, 40, 3) <= 30
    assert tier_name(90) == "CRITICAL" and tier_name(10) == "SAFE"


def test_safe_window_contiguous():
    hourly = [{"hour": h, "risk_score": 20 if 6 <= h <= 9 else 70} for h in range(24)]
    win = compute_safe_window(hourly)
    assert win["has_window"] and win["start"] == "06:00"


def test_safe_window_none():
    hourly = [{"hour": h, "risk_score": 90} for h in range(24)]
    assert compute_safe_window(hourly)["has_window"] is False


def test_translate_critical_shape():
    hourly = [{"hour": h, "risk_score": 90} for h in range(24)]
    out = translate(89, 43, 218, 9, 30, "cricket_fielding", "Delhi", hourly)
    assert out["severity"] == "CRITICAL"
    assert out["mask"] == "N95 mandatory"
    assert not out["sport_verdict"]["playable"]


def test_source_inference_stubble_only_in_oct_nov():
    nov = infer_sources("Delhi", datetime(2025, 11, 5, 20), 300, 400, 4, 60)
    jun = infer_sources("Delhi", datetime(2025, 6, 5, 20), 150, 400, 10, 40)
    nov_sources = [s["source"] for s in nov[:3]]
    assert "stubble_burning" in nov_sources
    jun_top = [s["source"] for s in jun[:2]]
    assert "stubble_burning" not in jun_top


def test_likelihoods_sum_to_100():
    out = infer_sources("Bengaluru", datetime(2025, 1, 10, 9), 80, 140, 8, 55)
    assert abs(sum(s["likelihood_pct"] for s in out) - 100) < 1.0


def test_grap_stages():
    assert grap_stage(150) is None
    assert grap_stage(320)["stage"] == "II"
    assert grap_stage(460)["stage"] == "IV"


def test_enso_narrative_el_nino_monsoon_mentions_rainfall():
    text = enso_narrative("el_nino", 7)  # July = monsoon
    assert "monsoon" in text.lower() and "rainfall" in text.lower()


def test_enso_narrative_el_nino_winter_mentions_pm25():
    text = enso_narrative("el_nino", 12)  # December = winter
    assert "pm2.5" in text.lower()


def test_enso_narrative_neutral():
    assert "neutral" in enso_narrative("neutral", 3).lower()


_ENSO_OUTLOOK_FIXTURE = """
<h1>Official NOAA CPC ENSO Probabilities</h1>
<h2>Issued August 2026</h2>
<table id="probabilities-table"><tbody>
<tr><th scope="row"><abbr>JAS <span class="tooltip tooltip-right" role="tooltip">Jul Aug Sep</span></abbr></th><td>0</td><td>0</td><td>100</td></tr><tr><th scope="row"><abbr>DJF <span class="tooltip tooltip-right" role="tooltip">Dec Jan Feb</span></abbr></th><td>0</td><td>0</td><td>100</td></tr><tr><th scope="row"><abbr>FMA <span class="tooltip tooltip-right" role="tooltip">Feb Mar Apr</span></abbr></th><td>0</td><td>3</td><td>97</td></tr>
</tbody></table>
"""


def test_enso_outlook_parses_issuance_and_rows():
    rows = parse_enso_outlook_html(_ENSO_OUTLOOK_FIXTURE)
    assert len(rows) == 3
    assert rows[0] == {"year": 2026, "month": 8, "season": "JAS",
                       "el_nino_pct": 0, "neutral_pct": 0, "la_nina_pct": 100,
                       "issued_year": 2026, "issued_month": 8}


def test_enso_outlook_rolls_year_forward_past_december():
    rows = parse_enso_outlook_html(_ENSO_OUTLOOK_FIXTURE)
    djf = next(r for r in rows if r["season"] == "DJF")
    assert djf["year"] == 2027  # issued Aug 2026; DJF's centre month (Jan) is next year


def test_enso_outlook_missing_header_raises():
    import pytest
    with pytest.raises(RuntimeError):
        parse_enso_outlook_html("<html>no data here</html>")


def test_government_brief_shape():
    b = government_brief("Delhi", datetime(2025, 11, 10, 21), 420, 260, 380, 3, 65)
    assert b["grap_stage"]["stage"] == "III"
    assert len(b["likely_dominant_sources"]) == 3
    assert all("short_term_actions" in s for s in b["all_sources"])


def test_grap_not_applicable_outside_delhi_ncr():
    # Mumbai isn't in GRAP_APPLICABLE_CITIES: even at a severe AQI, grap_stage
    # must stay null rather than implying CAQM's Delhi-NCR framework applies.
    b = government_brief("Mumbai", datetime(2025, 11, 10, 21), 420, 260, 380, 3, 65)
    assert b["grap_applicable"] is False
    assert b["grap_stage"] is None
    assert b["aqi_category"] == "Severe"


def test_unknown_city_uses_generic_prior_not_delhi_citations():
    b = government_brief("Mumbai", datetime(2025, 11, 10, 21), 150, 80, 140, 5, 55)
    assert b["used_generic_source_prior"] is True
    assert "stubble_burning" not in [s["source"] for s in b["all_sources"]]
    refs = b["likely_dominant_sources"][0]["apportionment_refs"]
    assert any("no published city-specific" in r.lower() for r in refs)
