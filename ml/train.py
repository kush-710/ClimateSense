"""Model training.

  python ml/train.py --city 1 --models prophet xgboost

Prophet
  - Daily-mean temp and PM2.5 forecasters per city.
  - Custom monsoon seasonality (period ~= 91 days).
  - ONI added as an *extra regressor* - the explicit El Nino signal. Because ONI is a
    slow monthly index, its future values over a 7-day horizon are safely approximated
    by the latest published value (documented model assumption).

XGBoost
  - 4-class risk-tier classifier over the full feature set in ml/features.FEATURES,
    which includes the ENSO block. Walk-forward TimeSeriesSplit CV (never random k-fold).
"""
from __future__ import annotations
import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, CITIES
from ml.features import (training_frame, add_features, load_joined, FEATURES,
                         FORECAST_HORIZON_H)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------- Prophet ------------------------------------
def train_prophet(city_id: int, target: str = "temp_c"):
    from prophet import Prophet
    from prophet.serialize import model_to_json

    # PM2.5 is driven far more by ventilation (wind) and humidity than by
    # calendar seasonality alone, so those join ONI as regressors for the
    # pollution model. Temperature doesn't need them.
    extra_regressors = ["wind_kph", "humidity_pct"] if target == "pm25" else []

    df = add_features(load_joined(city_id))
    agg = {target: "mean", "oni": "last"}
    agg.update({r: "mean" for r in extra_regressors})
    daily = (df.set_index("ts")[[target, "oni"] + extra_regressors]
               .resample("D").agg(agg)
               .dropna().reset_index()
               .rename(columns={"ts": "ds", target: "y"}))
    if len(daily) < 200:
        raise RuntimeError(f"Only {len(daily)} daily rows - seed more history first")

    m = Prophet(seasonality_mode="multiplicative", changepoint_prior_scale=0.05,
                interval_width=0.80, yearly_seasonality=True,
                weekly_seasonality=True, daily_seasonality=False)
    m.add_seasonality(name="monsoon", period=91.25, fourier_order=5)
    m.add_regressor("oni")                       # <-- El Nino signal
    for r in extra_regressors:
        m.add_regressor(r)
    m.fit(daily)

    # Walk-forward validation
    from prophet.diagnostics import cross_validation, performance_metrics
    cv = cross_validation(m, initial="180 days", period="30 days",
                          horizon="7 days", parallel=None)
    perf = performance_metrics(cv)
    logger.info("Prophet[%s/%s] MAE=%.2f RMSE=%.2f",
                CITIES[city_id]["name"], target,
                perf["mae"].mean(), perf["rmse"].mean())

    path = os.path.join(MODEL_DIR, f"prophet_{target}_city{city_id}.json")
    with open(path, "w") as f:
        f.write(model_to_json(m))
    with open(path.replace(".json", "_metrics.json"), "w") as f:
        json.dump({"mae": float(perf["mae"].mean()),
                   "rmse": float(perf["rmse"].mean())}, f)
    return m


def forecast_prophet(city_id: int, target: str = "temp_c", days: int = 7) -> pd.DataFrame:
    from prophet.serialize import model_from_json
    path = os.path.join(MODEL_DIR, f"prophet_{target}_city{city_id}.json")
    with open(path) as f:
        m = model_from_json(f.read())
    from services.data_access import latest_enso, recent_means

    future = m.make_future_dataframe(periods=days, freq="D")
    # Latest ONI carried forward for the horizon (monthly signal; safe over 7 days)
    enso = latest_enso(limit=1)
    future["oni"] = float(enso.iloc[0]["oni"]) if not enso.empty else 0.0
    # Same carry-forward for any meteorological regressors this model was fit
    # with - recent-week means, which beat a single noisy last reading. Read
    # straight from SQL rather than materialising the city's whole history.
    extra = [r for r in getattr(m, "extra_regressors", {}) if r != "oni" and r not in future]
    for name, value in recent_means(city_id, extra).items():
        future[name] = value if value is not None else 0.0
    fc = m.predict(future).tail(days)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def forecast_metrics(city_id: int, target: str = "temp_c") -> dict | None:
    """MAE/RMSE recorded at training time, for display alongside a forecast."""
    path = os.path.join(MODEL_DIR, f"prophet_{target}_city{city_id}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------- XGBoost ------------------------------------
def train_xgboost(city_id: int):
    import xgboost as xgb

    df = training_frame(city_id)
    feats = [f for f in FEATURES if f in df.columns]
    X, y = df[feats], df["risk_tier"].astype(int)
    logger.info("XGBoost training rows=%d features=%d target=risk tier +%dh tiers=%s",
                len(X), len(feats), FORECAST_HORIZON_H, y.value_counts().to_dict())
    # Persistence baseline: "tomorrow looks like right now". The model has to
    # beat this to be worth anything - same-hour labels made that trivially
    # true, a +24h target does not. Scored on the SAME walk-forward folds as
    # the model, otherwise the comparison is meaningless.
    persistence = df["risk_tier_now"].astype(int).reset_index(drop=True)

    # objective/num_class are deliberately NOT pinned here: an early walk-forward
    # fold can legitimately contain fewer than 4 risk tiers (HIGH/CRITICAL are rare
    # events), and forcing num_class=4 while a fold's y has fewer classes makes
    # newer XGBoost sklearn-API predict() return raw per-class scores instead of
    # argmax'd labels, breaking f1_score. Letting XGBoost infer objective/num_class
    # per fit keeps predict() shape consistent with whatever labels that fit saw.
    tscv = TimeSeriesSplit(n_splits=5)
    scores, baseline_scores, model = [], [], None
    all_tiers = sorted(y.unique())
    tier_names = [["SAFE", "MODERATE", "HIGH", "CRITICAL"][t] for t in all_tiers]
    for tr, va in tscv.split(X):
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        # A chronologically early fold can be single-class (e.g. an all-SAFE
        # stretch) - not enough signal to fit or score a fold like that.
        if y_tr.nunique() < 2:
            continue
        # HIGH/CRITICAL are rare and can appear in the validation slice before
        # the training slice has ever seen them. Early stopping needs the eval
        # set's labels to fit inside the fitted class space, and "mlogloss" only
        # makes sense once XGBoost has actually inferred a multi-class objective
        # (a 2-class fold auto-picks binary:logistic, for which mlogloss hard-
        # crashes) - skip eval_set/early stopping whenever either doesn't hold.
        can_eval = set(y_va.unique()) <= set(y_tr.unique()) and y_tr.nunique() >= 3
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5, gamma=0.1,
            eval_metric="mlogloss" if can_eval else None,
            early_stopping_rounds=25 if can_eval else None, n_jobs=-1)
        if can_eval:
            model.fit(X.iloc[tr], y_tr, eval_set=[(X.iloc[va], y_va)], verbose=False)
        else:
            model.fit(X.iloc[tr], y_tr)
        scores.append(f1_score(y_va, model.predict(X.iloc[va]), average="weighted"))
        baseline_scores.append(f1_score(y_va, persistence.iloc[va], average="weighted"))
    if not scores:
        raise RuntimeError("Every walk-forward fold was single-class - seed more history")
    logger.info("Persistence baseline F1(weighted)=%.3f  (same folds)",
                np.mean(baseline_scores))
    logger.info("XGBoost walk-forward F1(weighted)=%.3f +/- %.3f  (lift vs baseline %+.3f)",
                np.mean(scores), np.std(scores), np.mean(scores) - np.mean(baseline_scores))
    logger.info("\n%s", classification_report(
        y.iloc[va], model.predict(X.iloc[va]),
        labels=all_tiers, target_names=tier_names, zero_division=0))

    final = xgb.XGBClassifier(**{k: v for k, v in model.get_params().items()
                                 if k not in ("early_stopping_rounds", "objective", "num_class")})
    final.fit(X, y)
    bundle = {"model": final, "features": feats,
              "horizon_h": FORECAST_HORIZON_H,
              "cv_f1_mean": float(np.mean(scores)), "cv_f1_std": float(np.std(scores)),
              "baseline_f1_mean": float(np.mean(baseline_scores))}
    joblib.dump(bundle, os.path.join(MODEL_DIR, f"xgb_risk_city{city_id}.pkl"))
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--city", type=int, default=1)
    p.add_argument("--models", nargs="+", default=["prophet", "xgboost"],
                   choices=["prophet", "xgboost"])
    args = p.parse_args()
    if "prophet" in args.models:
        train_prophet(args.city, "temp_c")
        train_prophet(args.city, "pm25")
    if "xgboost" in args.models:
        train_xgboost(args.city)
