#!/usr/bin/env python3
"""
Predict future packet loss from an extracted feature-only CSV.

Example:
    python predict_xgboost_model.py xgboost_features_only.csv -o predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def channel_state(loss_percent: float) -> str:
    return "Critical" if loss_percent >= 10.0 else "Good"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost packet-loss prediction on feature CSV.")
    parser.add_argument("features", help="Feature-only CSV from extract_xgboost_features.py.")
    parser.add_argument("-o", "--output", default="predictions.csv", help="Output prediction CSV.")
    parser.add_argument("--model", default="models/xgboost_packet_loss_pipeline.joblib", help="Saved pipeline path.")
    parser.add_argument("--config", default="models/model_config.json", help="Saved model config path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = joblib.load(args.model)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    feature_columns = config["feature_columns"]
    feature_defaults = config.get("feature_defaults", {})

    df = pd.read_csv(args.features)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        missing_without_default = [col for col in missing if col not in feature_defaults]
        if missing_without_default:
            raise SystemExit(f"Missing feature columns: {missing_without_default}")
        for col in missing:
            df[col] = feature_defaults[col]
    for col in feature_columns:
        if col in feature_defaults:
            df[col] = df[col].fillna(feature_defaults[col])

    pred = pipeline.predict(df[feature_columns])
    pred = [max(0.0, float(value)) for value in pred]

    output_cols = [col for col in ["scenario_id", "window_start", "window_end"] if col in df.columns]
    out = df[output_cols].copy()
    out["predicted_packet_loss_future_2s"] = pred
    out["predicted_state"] = [channel_state(value) for value in pred]
    out.to_csv(args.output, index=False)

    print(f"Wrote predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
