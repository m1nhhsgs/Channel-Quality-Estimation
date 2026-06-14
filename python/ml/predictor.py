"""Runtime XGBoost predictor used by GUI replay/live modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class XGBoostPacketLossPredictor:
    def __init__(self, model_path: str | Path, config_path: str | Path):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.pipeline: Any | None = None
        self.feature_columns: list[str] = []
        self.feature_defaults: dict[str, Any] = {}
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and not self.error

    def load(self) -> bool:
        try:
            import joblib

            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.feature_columns = list(config["feature_columns"])
            self.feature_defaults = dict(config.get("feature_defaults", {}))
            self.pipeline = joblib.load(self.model_path)
            self.error = None
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced in GUI status.
            self.pipeline = None
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def predict(self, features: dict[str, Any]) -> float | None:
        if not self.ready or self.pipeline is None:
            return None

        row = {
            column: features.get(column, self.feature_defaults.get(column))
            for column in self.feature_columns
        }
        missing = [column for column, value in row.items() if value is None]
        if missing:
            self.error = f"Missing feature columns: {missing}"
            return None

        try:
            import pandas as pd

            frame = pd.DataFrame([row])
            prediction = float(self.pipeline.predict(frame)[0])
            return max(0.0, prediction)
        except Exception as exc:  # noqa: BLE001 - surfaced in GUI status.
            self.error = f"{type(exc).__name__}: {exc}"
            return None
