"""Model training.

  python ml/train.py --city 1 --models prophet xgboost

Prophet
  - Daily-mean temp and PM2.5 forecasters per city.
  - Custom monsoon seasonality (period ~= 91 days).
  - ONI added as an *extra regressor* — the explicit El Nino signal. Because ONI is a
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
from ml.features import training_frame, add_features, load_joined, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------- Prophet ------------------------------------
def train_prophet(city_id: int, target: str = "temp_c"):
    from prophet import Prophet
    from prophet.serialize import model_to_json

    df = add_features(load_joined(city_id))
    daily = (df.set_index("ts")[[target, "oni"]]
               .resample("D").agg({target: "mean", "oni": "last"})
               .dropna().reset_index()
               .rename(columns={"ts": "ds", target: "y"}))
    if len(daily) < 200:
        raise RuntimeError(f"Only {len(daily)} daily rows — seed more history first")

    m = Prophet(seasonality_mode="multiplicative", changepoint_prior_scale=0.05,
                interval_width=0.80, yearly_seasonality=True,
                weekly_seasonality=True, daily_seasonality=False)
    m.add_seasonality(name="monsoon", period=91.25, fourier_order=5)
    m.add_regressor("oni")                       # <-- El Nino signal
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
    future = m.make_future_dataframe(periods=days, freq="D")
    # Latest ONI carried forward for the horizon (monthly signal; safe over 7 days)
    hist = add_features(load_joined(city_id))
    last_oni = float(hist["oni"].dropna().iloc[-1]) if hist["oni"].notna().any() else 0.0
    future["oni"] = last_oni
    fc = m.predict(future).tail(days)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ------------------------------- XGBoost ------------------------------------
def train_xgboost(city_id: int):
    import xgboost as xgb

    df = training_frame(city_id)
    feats = [f for f in FEATURES if f in df.columns]
    X, y = df[feats], df["risk_tier"].astype(int)
    logger.info("XGBoost training rows=%d features=%d tiers=%s",
                len(X), len(feats), y.value_counts().to_dict())

    tscv = TimeSeriesSplit(n_splits=5)
    scores, model = [], None
    for tr, va in tscv.split(X):
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5, gamma=0.1,
            objective="multi:softprob", num_class=4,
            eval_metric="mlogloss", early_stopping_rounds=25, n_jobs=-1)
        model.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        scores.append(f1_score(y.iloc[va], model.predict(X.iloc[va]), average="weighted"))
    logger.info("XGBoost walk-forward F1(weighted)=%.3f +/- %.3f",
                np.mean(scores), np.std(scores))
    logger.info("\n%s", classification_report(
        y.iloc[va], model.predict(X.iloc[va]),
        target_names=["SAFE", "MODERATE", "HIGH", "CRITICAL"], zero_division=0))

    final = xgb.XGBClassifier(**{k: v for k, v in model.get_params().items()
                                 if k != "early_stopping_rounds"})
    final.fit(X, y)
    bundle = {"model": final, "features": feats,
              "cv_f1_mean": float(np.mean(scores)), "cv_f1_std": float(np.std(scores))}
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
