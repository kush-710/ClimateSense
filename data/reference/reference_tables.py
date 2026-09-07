"""Static reference tables for the pollution-source guidance layer.

IMPORTANT DESIGN NOTE
---------------------
Real-time chemical source apportionment is impossible from concentration data alone;
it requires speciated measurements + receptor models (PMF/CMB). These tables therefore
encode *published* apportionment results (IIT-Kanpur 2016 Delhi study, TERI 2018,
SAFAR, CPCB) and the platform infers which known source is *likely dominant right now*
from season, hour, and meteorology. Every recommendation cites its evidence base.

All shares are indicative ranges from the literature — verify the exact figures against
the cited reports before publishing, and cite them in the README/paper.
"""

# ---------------------------------------------------------------------------
# 1. Seasonal PM2.5 source apportionment (percent share ranges, city x season)
#    season keys: winter (Nov-Jan), summer (Mar-Jun), monsoon (Jul-Sep),
#                 post_monsoon (Oct), transition (Feb)
# ---------------------------------------------------------------------------
SOURCE_APPORTIONMENT = {
    "Delhi": {
        "winter": {
            "vehicular": (20, 30), "biomass_residential_burning": (15, 25),
            "industry_power": (15, 25), "road_construction_dust": (10, 15),
            "secondary_aerosols": (15, 25), "waste_burning": (5, 10),
            "refs": ["IIT-Kanpur (2016) Delhi Source Apportionment",
                     "TERI-ARAI (2018) Delhi-NCR Apportionment"],
        },
        "summer": {
            "road_construction_dust": (30, 40), "vehicular": (15, 25),
            "industry_power": (15, 20), "secondary_aerosols": (10, 20),
            "waste_burning": (5, 10),
            "refs": ["IIT-Kanpur (2016)"],
        },
        "post_monsoon": {
            "stubble_burning": (20, 40), "vehicular": (15, 25),
            "road_construction_dust": (10, 20), "industry_power": (10, 15),
            "secondary_aerosols": (10, 20),
            "refs": ["SAFAR stubble contribution advisories (Oct-Nov)",
                     "CEEW (2021) NCR emissions inventory"],
        },
        "monsoon": {
            "vehicular": (25, 35), "industry_power": (20, 30),
            "road_construction_dust": (10, 15), "secondary_aerosols": (10, 20),
            "refs": ["TERI-ARAI (2018)"],
        },
        "transition": {
            "vehicular": (20, 30), "industry_power": (15, 25),
            "road_construction_dust": (15, 25), "secondary_aerosols": (15, 25),
            "refs": ["TERI-ARAI (2018)"],
        },
    },
    "Bengaluru": {
        "winter": {
            "vehicular": (40, 50), "road_construction_dust": (15, 25),
            "waste_burning": (10, 15), "industry_dg_sets": (10, 15),
            "secondary_aerosols": (10, 15),
            "refs": ["CSTEP (2022) Bengaluru Source Apportionment"],
        },
        "summer": {
            "vehicular": (35, 45), "road_construction_dust": (20, 30),
            "waste_burning": (10, 15), "industry_dg_sets": (10, 15),
            "refs": ["CSTEP (2022)"],
        },
        "monsoon": {
            "vehicular": (40, 50), "industry_dg_sets": (15, 20),
            "road_construction_dust": (10, 15),
            "refs": ["CSTEP (2022)"],
        },
        "post_monsoon": {
            "vehicular": (40, 50), "road_construction_dust": (15, 25),
            "waste_burning": (10, 15),
            "refs": ["CSTEP (2022)"],
        },
        "transition": {
            "vehicular": (40, 50), "road_construction_dust": (15, 25),
            "refs": ["CSTEP (2022)"],
        },
    },
}

# Fallback prior for any city without a specific encoded study above (used by
# infer_sources() for Mumbai/Chennai/Kolkata/Hyderabad/Pune/etc.). Deliberately
# omits Delhi-specific categories (stubble_burning, biomass_residential_burning)
# that don't generalise nationally, and its refs say plainly that this is NOT a
# published city-specific study — silently reusing Delhi's IIT-Kanpur/TERI
# citations for an unrelated city would misattribute real research.
GENERIC_URBAN_PRIOR = {
    "vehicular": (30, 45), "road_construction_dust": (15, 25),
    "industry_power": (10, 20), "secondary_aerosols": (10, 20),
    "waste_burning": (5, 15),
    "refs": ["No published city-specific source-apportionment study encoded — "
             "indicative generic urban shares only; verify locally before use."],
}

# ---------------------------------------------------------------------------
# 2. Source -> intervention mapping (policy recommendations, each with evidence)
# ---------------------------------------------------------------------------
INTERVENTIONS = {
    "vehicular": {
        "short_term": ["Enforce GRAP vehicle restrictions (BS-III petrol / BS-IV diesel bans)",
                       "Odd-even rationing during CRITICAL episodes",
                       "Intensify PUC-certificate enforcement at fuel stations"],
        "long_term": ["Accelerate EV adoption for last-mile fleets and buses",
                      "Expand metro/bus network capacity and frequency",
                      "Low-emission zones in high-exposure wards"],
        "evidence": "TERI-ARAI (2018): transport is a leading PM2.5 source in NCR winters",
        "responsible": ["Transport Dept", "Traffic Police", "CAQM"],
    },
    "stubble_burning": {
        "short_term": ["Activate SAFAR fire-count advisories; pre-position Pusa bio-decomposer",
                       "Coordinate with Punjab/Haryana on paddy-residue windows"],
        "long_term": ["Subsidise Happy Seeder / in-situ residue management",
                      "Crop diversification incentives away from late-kharif paddy",
                      "Residue-to-energy procurement (biomass pellets for power plants)"],
        "evidence": "SAFAR: stubble share of Delhi PM2.5 peaks 20-40% in Oct-Nov episodes",
        "responsible": ["CAQM", "State Agriculture Depts", "MoEFCC"],
    },
    "road_construction_dust": {
        "short_term": ["GRAP construction-and-demolition (C&D) bans at Stage III+",
                       "Mechanised road sweeping + water sprinkling on hotspots",
                       "Anti-smog guns at large construction sites"],
        "long_term": ["End-to-end C&D waste processing capacity",
                      "Greening/paving of exposed shoulders and open plots"],
        "evidence": "IIT-Kanpur (2016): dust dominates Delhi summer PM10/PM2.5",
        "responsible": ["Municipal corporations", "DPCC/KSPCB", "PWD"],
    },
    "industry_power": {
        "short_term": ["GRAP fuel restrictions on non-PNG industry",
                       "Ramp down captive coal/pet-coke units during episodes"],
        "long_term": ["PNG conversion of industrial clusters",
                      "FGD retrofits + strict emission monitoring on power plants",
                      "Relocate red-category industry outside airshed"],
        "evidence": "TERI-ARAI (2018): industry + power share of NCR PM2.5",
        "responsible": ["State PCBs", "Industries Dept", "CAQM"],
    },
    "industry_dg_sets": {
        "short_term": ["Restrict diesel generator use per GRAP-style triggers",
                       "Enforce retrofit emission-control devices on DG sets"],
        "long_term": ["Reliable grid supply to eliminate DG dependence",
                      "Rooftop solar + storage mandates for large campuses"],
        "evidence": "CSTEP (2022): DG sets are a notable Bengaluru PM source",
        "responsible": ["BESCOM", "KSPCB", "Municipal corporation"],
    },
    "biomass_residential_burning": {
        "short_term": ["Distribute electric heaters to night-shelter and outdoor workers",
                       "Ban open burning of refuse/leaves; enforce with marshals"],
        "long_term": ["Deepen LPG/PNG penetration (Ujjwala follow-through)",
                      "District heating pilots for informal settlements"],
        "evidence": "IIT-Kanpur (2016): residential biomass burning peaks in Delhi winters",
        "responsible": ["Municipal corporations", "MoPNG schemes"],
    },
    "waste_burning": {
        "short_term": ["Marshal patrols + fines for open waste fires",
                       "Rapid response to landfill fire ignitions"],
        "long_term": ["100% waste segregation and processing; close legacy dumpsites",
                      "Waste-to-compost / bio-CNG capacity expansion"],
        "evidence": "CPCB inventories: open waste burning is a persistent urban PM source",
        "responsible": ["Municipal SWM departments"],
    },
    "secondary_aerosols": {
        "short_term": ["Regional NOx/SO2/NH3 episode controls (precursor gases)"],
        "long_term": ["Airshed-level management: coordinated NCR-wide precursor reduction",
                      "Fertiliser NH3 management in surrounding agriculture"],
        "evidence": "Secondary inorganic aerosols form from precursor gases regionally",
        "responsible": ["CAQM (airshed authority)", "Neighbouring state PCBs"],
    },
}

# ---------------------------------------------------------------------------
# 3. GRAP stage triggers (CAQM, Delhi-NCR) — maps live AQI to the official
#    response framework so recommendations reference real policy machinery.
# ---------------------------------------------------------------------------
GRAP_STAGES = [
    {"stage": "I",   "aqi_range": (201, 300), "label": "Poor",
     "actions": ["Dust control on roads/C&D sites", "Strict PUC enforcement",
                 "No open burning of waste"]},
    {"stage": "II",  "aqi_range": (301, 400), "label": "Very Poor",
     "actions": ["Ban DG sets (except emergency)", "Parking fee hikes",
                 "Augment bus/metro service"]},
    {"stage": "III", "aqi_range": (401, 450), "label": "Severe",
     "actions": ["Ban non-essential construction", "BS-III petrol / BS-IV diesel car ban",
                 "Shift schools to hybrid (younger classes)"]},
    {"stage": "IV",  "aqi_range": (451, 500), "label": "Severe+",
     "actions": ["Ban truck entry (non-essential)", "Close non-PNG industry",
                 "50% WFH for offices; consider odd-even"]},
]

# ---------------------------------------------------------------------------
# 4. Sport-specific environmental thresholds (FIFA / ACSM / WHO anchored)
#    (max_temp_c, max_wbgt_c, max_aqi, max_uv)
# ---------------------------------------------------------------------------
SPORT_LIMITS = {
    "football":       (34, 32, 150, 8),
    "cricket_batting": (36, 32, 120, 7),
    "cricket_fielding": (33, 30, 100, 6),
    "marathon":       (28, 26, 100, 6),
    "running_casual": (32, 29, 120, 7),
    "cycling":        (33, 30, 130, 7),
    "school_pe":      (30, 26, 100, 6),
    "general":        (35, 32, 150, 8),
}


def season_of(month: int) -> str:
    """Seasonal buckets aligned to Indo-Gangetic pollution regimes.
    post_monsoon covers Oct-Nov because stubble-burning episodes span both months."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5, 6):
        return "summer"
    if month in (7, 8, 9):
        return "monsoon"
    return "post_monsoon"  # 10, 11


# National CPCB AQI categories (apply everywhere in India, unlike GRAP which
# is Delhi-NCR-specific policy machinery).
_AQI_CATEGORIES = [
    (0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
    (201, 300, "Poor"), (301, 400, "Very Poor"), (401, 500, "Severe"),
]


def aqi_category(aqi: float | None) -> str | None:
    if aqi is None:
        return None
    for lo, hi, label in _AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Severe" if aqi > 500 else None
